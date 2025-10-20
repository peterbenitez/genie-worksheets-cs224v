from __future__ import annotations
import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import List, Optional, TYPE_CHECKING

import pandas as pd
import requests
from loguru import logger
from suql.agent import DialogueTurn as SUQLDialogueTurn

from langchain_core.output_parsers import StrOutputParser
from worksheets.agent.config import AzureModelConfig, OpenAIModelConfig
from worksheets.core.dialogue import CurrentDialogueTurn
from worksheets.core.runtime import GenieRuntime
from worksheets.core.worksheet import Answer
from worksheets.knowledge.base import SUQLKnowledgeBase
from worksheets.llm.llm import get_llm_client
from worksheets.llm.prompts import load_fewshot_prompt_template
from worksheets.llm.logging import LoggingHandler

from worksheets.utils.llm import extract_code_block_from_output

if TYPE_CHECKING:
    from worksheets.kraken.utils import DialogueTurn
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


@dataclass
class DatatalkDialogueTurn:
    question: str
    action_history: List[dict]
    entity_linking_results: dict
    response: str


class BaseKnowledgeParser(ABC):
    """Base class for parsers"""

    @abstractmethod
    def parse(
        self,
        query: str,
        dlg_history: List[CurrentDialogueTurn],
        *args,
        **kwargs,
    ):
        """Parse the query and return the output."""
        pass


class BaseSUQLParser(BaseKnowledgeParser):
    """Base class for SUQL parsers"""

    def __init__(self, model_config: AzureModelConfig | OpenAIModelConfig, **kwargs):
        super().__init__(**kwargs)

        # Name of the LLM model to use for the queries
        self.model_config = model_config

    async def parse(
        self,
        query: str,
        dlg_history: List[CurrentDialogueTurn],
        *args,
        **kwargs,
    ):
        raise NotImplementedError

    def convert_dlg_turn_to_suql_dlg_turn(self, dlg_history, db_results):
        # Convert the dialog history to the expected format for SUQL
        suql_dlg_history = []
        for i, turn in enumerate(dlg_history):
            user_target = turn.user_target_suql
            agent_utterance = turn.system_response
            user_utterance = turn.user_utterance

            if db_results is None:
                db_result = [
                    obj.result
                    for obj in turn.context.context.values()
                    if isinstance(obj, Answer)
                    and obj.query.value == turn.user_target_suql
                ]
            else:
                db_result = db_results[i]

            suql_dlg_history.append(
                SUQLDialogueTurn(
                    user_utterance=user_utterance,
                    db_results=db_result,
                    user_target=user_target,
                    agent_utterance=agent_utterance,
                )
            )

        db_result = None
        return suql_dlg_history, db_result, False


class SUQLParser(BaseSUQLParser):
    """Parser for SUQL queries"""

    def __init__(
        self,
        model_config: AzureModelConfig | OpenAIModelConfig,
        example_path: str = None,
        instruction_path: str = None,
        table_schema_path: str = None,
        examples: Optional[List[str]] = None,
        instructions: Optional[List[str]] = None,
        table_schema: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model_config=model_config, **kwargs)

        # Selector for the prompt to use for the queries
        self.prompt_selector = kwargs.get("prompt_selector", None)

        self.example_path = example_path
        self.instruction_path = instruction_path
        self.table_schema_path = table_schema_path
        self.examples = examples
        self.instructions = instructions
        self.table_schema = table_schema

        if self.examples is None and self.example_path is not None:
            self.examples = []
            with open(self.example_path, "r") as f:
                text = f.read()

            for example in text.split("--"):
                if example.strip():
                    self.examples.append(example.strip())

        if self.instructions is None and self.instruction_path is not None:
            with open(self.instruction_path, "r") as f:
                self.instructions = f.readlines()

        if self.table_schema is None and self.table_schema_path is not None:
            with open(self.table_schema_path, "r") as f:
                self.table_schema = f.read()

    async def parse(
        self,
        query: str,
        dlg_history: List[CurrentDialogueTurn],
        bot: GenieRuntime,
        db_results: List[str] | None = None,
    ):
        """
        A SUQL conversational semantic parser, with a pre-set prompt file.
        The function convets the List[CurrentDialogueTurn] to the expected format
        in SUQL (suql.agent.DialogueTurn) and calls the prompt file.

        # Parameters:

        `dlg_history` (List[CurrentDialogueTurn]): a list of past dialog turns.

        `query` (str): the current query to be parsed.

        # Returns:

        `parsed_output` (str): a parsed SUQL output
        """

        suql_dlg_history = self.convert_dlg_turn_to_suql_dlg_turn(
            dlg_history, db_results
        )

        # Use the prompt selector if available
        if self.prompt_selector:
            prompt_file = await self.prompt_selector(bot, dlg_history, query)
        else:
            prompt_file = "suql_parser.prompt"

        self.llm_client = get_llm_client(
            model=self.model_config.model_name,
            temperature=0.0,
            max_tokens=1024,
        )
        self.prompt_template = load_fewshot_prompt_template(
            prompt_file
        )
        self.chain = self.prompt_template | self.llm_client | StrOutputParser()

        logging_handler = LoggingHandler(
            prompt_file=prompt_file,
            metadata={
                "dlg": suql_dlg_history,
                "query": query,
            },
        )
        # Generate the SUQL output
        parsed_output = await self.chain.ainvoke(
            {
                "dlg": suql_dlg_history,
                "query": query,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "day": datetime.now().strftime("%A"),
                "day_tmr": (datetime.now() + timedelta(days=1)).strftime("%A"),
                "examples": "\n--\n".join(self.examples),
                "instructions": "\n".join([f"- {i}" for i in self.instructions]),
                "table_schema": self.table_schema,
            },
            config={"callbacks": [logging_handler]},
        )

        db_result = None
        return (
            extract_code_block_from_output(parsed_output, lang="sql"),
            db_result,
            False,
        )


class SUQLReActParser(BaseSUQLParser):
    """ReAct Parser for SUQL queries"""

    def __init__(
        self,
        model_config: AzureModelConfig | OpenAIModelConfig,
        example_path: str,
        instruction_path: str,
        table_schema_path: str,
        knowledge: SUQLKnowledgeBase,
        examples: Optional[List[str]] = None,
        instructions: Optional[List[str]] = None,
        table_schema: Optional[str] = None,
        conversation_history: Optional[List[DialogueTurn]] = None,
        **kwargs,
    ):
        super().__init__(model_config=model_config, **kwargs)

        self.example_path = example_path
        self.instruction_path = instruction_path
        self.table_schema_path = table_schema_path
        self.knowledge = knowledge
        self.examples = examples
        self.instructions = instructions
        self.table_schema = table_schema
        self.conversation_history = conversation_history

        if self.examples is None and self.example_path is not None:
            self.examples = []
            with open(self.example_path, "r") as f:
                text = f.read()

            for example in text.split("--"):
                if example.strip():
                    self.examples.append(example.strip())

        if self.instructions is None and self.instruction_path is not None:
            with open(self.instruction_path, "r") as f:
                self.instructions = f.readlines()

        if self.table_schema is None and self.table_schema_path is not None:
            with open(self.table_schema_path, "r") as f:
                self.table_schema = f.read()

    async def parse(
        self,
        query: str,
        dlg_history: List[CurrentDialogueTurn],
        runtime: GenieRuntime,
        db_results: List[str] | None = None,
    ):
        suql_dlg_history = self.convert_dlg_turn_to_suql_dlg_turn(
            dlg_history, db_results
        )

        self.conversation_history = suql_dlg_history

        output = await self.anext_turn(
            query,
            update_conversation_history=False,
            table_w_ids=self.knowledge.tables_with_primary_keys,
            database_name=self.knowledge.database_name,
            embedding_server_address=self.knowledge.embedding_server_address,
            source_file_mapping=self.knowledge.source_file_mapping,
        )

        # TODO: KeyError: 'final_sql'
        # happens when the action_counter limit is met without a final SQL being generated
        logger.info(f"SUQL output: {output}")
        try:
            final_output = output["final_sql"].sql
            final_result = output["final_sql"].execution_result
        except Exception as e:
            logger.error(f"Error in parsing output: {e}")
            final_output = None
            final_result = None
        return final_output, final_result, True

    async def anext_turn(
        self,
        user_input: str,
        update_conversation_history: bool = False,
        table_w_ids: dict = None,
        database_name: str = None,
        embedding_server_address: str = None,
        source_file_mapping: dict = None,
    ):
        from worksheets.kraken.agent import KrakenParser
        parser = KrakenParser()
        parser.initialize(
            engine=self.model_config.model_name,
            table_w_ids=table_w_ids,
            database_name=database_name,
            suql_model_name=self.knowledge.model_config.model_name,
            suql_api_base=os.getenv("LLM_API_ENDPOINT"),
            suql_api_version=os.getenv("LLM_API_VERSION"),
            suql_api_key=os.getenv("LLM_API_KEY"),
            embedding_server_address=embedding_server_address,
            db_host=self.knowledge.db_host,
            db_port=self.knowledge.db_port,
            db_username=self.knowledge.db_username,
            db_password=self.knowledge.db_password,
            source_file_mapping=source_file_mapping,
            domain_instructions=self.instructions,
            examples=self.examples,
            table_schema=self.table_schema,
        )

        output = await parser.arun(
            {
                "question": user_input,
                "conversation_history": self.conversation_history,
            }
        )

        if update_conversation_history:
            self.update_turn(self.conversation_history, output, response=None)

        return output

    def update_turn(self, conversation_history, output, response):
        from worksheets.kraken.utils import DialogueTurn
        turn = DialogueTurn(
            user_utterance=output["question"],
            agent_utterance=response,
            user_target=output["final_sql"].sql,
            db_results=output["final_sql"].execution_result,
        )

        conversation_history.append(turn)


class DatatalkParser(SUQLReActParser):
    """Datatalk Parser for SUQL queries"""

    def __init__(
        self,
        model_config: AzureModelConfig | OpenAIModelConfig,
        domain: str,
        api_key: str | None = None,
        **kwargs,
    ):
        super().__init__(
            model_config=model_config,
            table_schema_path=None,
            instruction_path=None,
            example_path=None,
            **kwargs,
        )
        self.domain = domain
        self.api_key = api_key

    async def anext_turn(
        self,
        user_input: str,
        update_conversation_history: bool = False,
        table_w_ids: dict = None,
        database_name: str = None,
        embedding_server_address: str = None,
        source_file_mapping: dict = None,
    ):
        # Define the API endpoint and the API key
        api_url = "http://localhost:8791/api"  # Adjust the URL if your server is running on a different host or port
        api_key = self.api_key if self.api_key else os.getenv("DATATALK_API")

        # Set up the parameters for the GET request
        conv_hist_serializable = [asdict(t) for t in self.conversation_history]
        params = {
            "question": user_input,
            "domain": self.domain,
            "api_key": api_key,
            "conversation_history": json.dumps(conv_hist_serializable),
            "file_path": "/home/oval/storm/datatalk/sql_results",
            "save_result_to_csv": True,
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(
            api_url,
            headers=headers,
            json=params,
            params={"api_key": api_key},
            timeout=600,
        )
        response = response.json()

        csv_path = response.get("csv_path", None)
        if csv_path is not None:
            df = pd.read_csv(response["csv_path"])
            json_data = df.to_dict(orient="records")
            response["sql_result"] = json_data
        else:
            response["sql_result"] = None
        response["questions"] = user_input

        if update_conversation_history:
            self.update_turn(self.conversation_history, response, response=None)

        return response

    def convert_dlg_turn_to_suql_dlg_turn(self, dlg_history, db_results):
        # Convert the dialog history to the expected format for SUQL
        suql_dlg_history = []
        for i, turn in enumerate(dlg_history):
            agent_utterance = turn.system_response
            user_utterance = turn.user_utterance

            if db_results is None:
                db_result = [
                    obj.result
                    for obj in turn.context.context.values()
                    if isinstance(obj, Answer)
                    and obj.query.value == turn.user_target_suql
                ]
            else:
                db_result = db_results[i]

            suql_dlg_history.append(
                DatatalkDialogueTurn(
                    question=user_utterance,
                    action_history=[],
                    entity_linking_results={},
                    response=agent_utterance,
                )
            )

        return suql_dlg_history

    def update_turn(self, conversation_history, output, response):
        from worksheets.kraken.utils import DialogueTurn
        turn = DialogueTurn(
            user_utterance=output["question"],
            agent_utterance=response,
            user_target=output["generated_sql"],
            db_results=output["sql_result"],
        )

        conversation_history.append(turn)

    async def parse(
        self,
        query: str,
        dlg_history: List[CurrentDialogueTurn],
        runtime: GenieRuntime,
        db_results: List[str] | None = None,
    ):
        suql_dlg_history = self.convert_dlg_turn_to_suql_dlg_turn(
            dlg_history, db_results
        )

        self.conversation_history = suql_dlg_history

        output = await self.anext_turn(
            query,
            update_conversation_history=False,
            table_w_ids=self.knowledge.tables_with_primary_keys,
            database_name=self.knowledge.database_name,
            embedding_server_address=self.knowledge.embedding_server_address,
            source_file_mapping=self.knowledge.source_file_mapping,
        )

        # TODO: KeyError: 'final_sql'
        # happens when the action_counter limit is met without a final SQL being generated
        logger.info(f"SUQL output: {output}")
        try:
            final_output = output["generated_sql"]
            final_result = output["sql_result"]
        except Exception as e:
            logger.error(f"Error in parsing output: {e}")
            final_output = None
            final_result = None
        return final_output, final_result, True
