from __future__ import annotations

import ast
import datetime
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from loguru import logger
from sql_metadata import Parser
from suql.sql_free_text_support.execute_free_text_sql import _check_required_params
from langchain_core.output_parsers import StrOutputParser
from worksheets.components.rewriter import rewrite_code_to_extract_funcs
from worksheets.core.dialogue import CurrentDialogueTurn
from worksheets.core.runtime import GenieRuntime
from worksheets.utils.annotation import prepare_semantic_parser_input
from worksheets.utils.field import get_genie_fields_from_ws
from worksheets.utils.llm import extract_code_block_from_output
from worksheets.utils.worksheet import count_worksheet_variables
from worksheets.llm.llm import get_llm_client
from worksheets.llm.prompts import load_fewshot_prompt_template
from worksheets.llm.logging import LoggingHandler

if TYPE_CHECKING:
    from worksheets.agent.agent import Agent
    from worksheets.knowledge.parser import BaseKnowledgeParser


class ContextualSemanticParser:
    """A class responsible for generating formal code representations from natural language using LLM.

    This class handles the contextual parsing of user utterances into executable code,
    taking into account dialogue history, available APIs, and system state.

    Attributes:
        runtime (GenieRuntime): The runtime instance for configuration and model access.
        agent (Agent): The agent instance for configuration and model access.
    """

    def __init__(self, runtime: GenieRuntime, agent):
        """Initialize the ContextualSemanticParser.

        Args:
            runtime (GenieRuntime): The runtime instance.
        """
        self.runtime = runtime
        self.agent = agent

        self.llm_client = get_llm_client(
            model=self.agent.config.semantic_parser.model_name,
            temperature=self.agent.config.semantic_parser.temperature,
            top_p=self.agent.config.semantic_parser.top_p,
            max_tokens=self.agent.config.semantic_parser.max_tokens,
        )

        self.prompt_template = load_fewshot_prompt_template(
            "semantic_parser.prompt"
        )
        self.chain = self.prompt_template | self.llm_client | StrOutputParser()

    async def generate_formal_representation(
        self,
        dlg_history: List[CurrentDialogueTurn],
        current_dlg_turn: CurrentDialogueTurn,
        state_schema: Optional[str],
        agent_acts: Optional[str],
        agent_utterance: Optional[str],
        available_worksheets_text: str,
        available_dbs_text: str,
    ) -> str:
        """Generate formal code representation from natural language input.

        Args:
            dlg_history (List[CurrentDialogueTurn]): The dialogue history.
            current_dlg_turn (CurrentDialogueTurn): The current dialogue turn.
            state_schema (Optional[str]): The state schema.
            agent_acts (Optional[str]): The agent actions.
            agent_utterance (Optional[str]): The agent utterance.
            available_worksheets_text (str): Available worksheets text.
            available_dbs_text (str): Available databases text.

        Returns:
            str: The generated formal code representation.
        """
        logging_handler = LoggingHandler(
            prompt_file="semantic_parser.prompt",
            metadata={
                "user_utterance": current_dlg_turn.user_utterance,
                "state_schema": state_schema,
                "agent_acts": agent_acts,
                "agent_utterance": agent_utterance,
                "available_worksheets_text": available_worksheets_text,
                "available_dbs_text": available_dbs_text,
            },
            session_id=self.agent.session_id,
        )
        prompt_inputs = self._prepare_prompt_inputs(
            dlg_history,
            current_dlg_turn,
            state_schema,
            agent_acts,
            agent_utterance,
            available_worksheets_text,
            available_dbs_text,
        )

        parsed_output = await self.chain.ainvoke(prompt_inputs, config={"callbacks": [logging_handler]})

        return extract_code_block_from_output(parsed_output, lang="python")

    def _prepare_prompt_inputs(
        self,
        dlg_history: List[CurrentDialogueTurn],
        current_dlg_turn: CurrentDialogueTurn,
        state_schema: Optional[str],
        agent_acts: Optional[str],
        agent_utterance: Optional[str],
        available_worksheets_text: str,
        available_dbs_text: str,
    ) -> Dict[str, Any]:
        """Prepare inputs for the LLM prompt.

        Args:
            Same as generate_formal_representation.

        Returns:
            Dict[str, Any]: The prepared prompt inputs.
        """
        current_date = datetime.datetime.now()
        refusal = getattr(self.agent.config, 'refusal', False)

        return {
            "user_utterance": current_dlg_turn.user_utterance,
            "dlg_history": dlg_history,
            "apis": available_worksheets_text,
            "database_tables": available_dbs_text,
            "date": current_date.strftime("%Y-%m-%d"),
            "day": current_date.strftime("%A"),
            "date_tmr": (current_date + datetime.timedelta(days=1)).strftime(
                "%Y-%m-%d"
            ),
            "yesterday_date": (current_date - datetime.timedelta(days=1)).strftime(
                "%Y-%m-%d"
            ),
            "state": state_schema,
            "agent_actions": agent_acts if agent_acts else "None",
            "agent_utterance": agent_utterance,
            "description": self.agent.description,
            "refusal": refusal,
        }

    def _get_model_args(self) -> Dict[str, Any]:
        """Get model arguments for LLM generation.

        Returns:
            Dict[str, Any]: The model arguments.
        """
        return self.agent.config.semantic_parser.model_dump()


class KnowledgeBaseParser:
    """A class responsible for processing and transforming answer queries into SUQL queries.

    This class handles the extraction and processing of answer queries, managing database table
    information, and generating SUQL queries with proper parameter handling.

    Attributes:
        runtime (GenieRuntime): The bot runtime instance for database access.
    """

    def __init__(self, runtime: GenieRuntime, parser):
        """Initialize the KnowledgeBaseParser.

        Args:
            runtime (GenieRuntime): The runtime instance.
            parser (BaseParser): The parser instance.
        """
        self.runtime = runtime
        self.parser = parser

    async def process_answer_queries(
        self,
        answer_queries: List[Tuple[str, Optional[str]]],
        dlg_history: List[CurrentDialogueTurn],
        user_target: str,
        pattern_type: str,
    ) -> Tuple[List[str], str]:
        """Process answer queries and generate SUQL queries.

        Args:
            answer_queries (List[Tuple[str, Optional[str]]]): List of (query, datatype) tuples to process.
            dlg_history (List[CurrentDialogueTurn]): The dialogue history.
            user_target (str): The user target code.
            pattern_type (str): The pattern type ("func" or "attr").

        Returns:
            Tuple[List[str], str]: Processed SUQL queries and updated user target.
        """
        suql_queries = []
        updated_target = user_target

        for answer_query, datatype in answer_queries:
            logger.info(f"Answer query: {answer_query}, datatype: {datatype}")
            parsing_sql_response = await self._parse_to_suql(dlg_history, answer_query)

            if parsing_sql_response:
                suql_query, db_result, db_result_exec = parsing_sql_response
            else:
                suql_query = None
                db_result = None
                db_result_exec = False

            if suql_query:
                tables, unfilled_params = self._process_suql_query(suql_query)
                suql_queries.append(suql_query)
            else:
                tables = []
                unfilled_params = {}

            updated_target = self._update_user_target(
                updated_target,
                answer_query,
                suql_query,
                tables,
                unfilled_params,
                pattern_type,
                db_result,
                db_result_exec,
                datatype,
            )

        return suql_queries, updated_target

    async def _parse_to_suql(
        self,
        dlg_history: List[CurrentDialogueTurn],
        answer_query: str,
    ) -> Tuple[Optional[str], Optional[str], bool]:
        """Parse an answer query to SUQL format.

        Args:
            dlg_history (List[CurrentDialogueTurn]): The dialogue history.
            answer_query (str): The answer query to parse.

        Returns:
            Tuple[Optional[str], Optional[str], bool]: The parsed SUQL query, database result, and database result
            execution is True if the query is executed.
        """
        suql_query, db_result, db_result_exec = await self.parser.parse(
            answer_query[1:-1], dlg_history, self.runtime
        )

        if suql_query is None:
            logger.error(f"SUQL parsing failed for {answer_query}")
            return ""

        return suql_query.replace("\*", "*"), db_result, db_result_exec

    def _process_suql_query(self, suql_query: str) -> Tuple[List[str], Dict[str, Any]]:
        """Process a SUQL query to extract tables and unfilled parameters.

        Args:
            suql_query (str): The SUQL query to process.

        Returns:
            Tuple[List[str], Dict[str, Any]]: Tables and unfilled parameters.
        """
        if "SELECT" not in suql_query:
            return [], {}

        tables = Parser(suql_query).tables
        table_req_params = {
            table: self._get_required_params_in_table(table)[0] for table in tables
        }

        _, unfilled_params = _check_required_params(suql_query, table_req_params)

        # Check if primary key is filled when there are unfilled params
        if unfilled_params:
            id_filled, _ = _check_required_params(
                suql_query, self._get_table_primary_keys()
            )
            if id_filled:
                unfilled_params = {}

        return tables, unfilled_params

    def _get_table_primary_keys(self) -> Dict[str, List[str]]:
        """Get the primary keys for all tables in the bot's database models.

        Returns:
            Dict[str, List[str]]: A dictionary mapping table names to their primary keys.
        """
        mapping = {}
        for db in self.runtime.genie_db_models:
            for field in get_genie_fields_from_ws(db):
                if field.primary_key:
                    mapping[db.__name__] = [field.name]
        return mapping

    def _get_required_params_in_table(self, table: str) -> Tuple[List[str], Any]:
        """Get the required parameters for a given table.

        Args:
            table (str): The name of the table.

        Returns:
            Tuple[List[str], Any]: A tuple containing required parameters and table class.
        """
        required_params = []
        table_class = None
        for db in self.runtime.genie_db_models:
            if db.__name__ == table:
                table_class = db
                for field in get_genie_fields_from_ws(db):
                    if not field.optional:
                        required_params.append(field.name)
        return required_params, table_class

    @staticmethod
    def _update_user_target(
        user_target: str,
        answer_query: str,
        suql_query: str,
        tables: List[str],
        unfilled_params: Dict[str, Any],
        pattern_type: str,
        db_result: Optional[str],
        db_result_exec: bool,
        datatype: Optional[str],
    ) -> str:
        """Update the user target with processed query information.

        Args:
            user_target (str): The current user target.
            answer_query (str): The original answer query.
            suql_query (str): The processed SUQL query.
            tables (List[str]): The tables used in the query.
            unfilled_params (Dict[str, Any]): The unfilled parameters.
            pattern_type (str): The pattern type ("func" or "attr").
            db_result (Optional[str]): The database result.
            db_result_exec (bool): Whether the database result was executed.
            datatype (Optional[str]): The datatype specified for the answer.

        Returns:
            str: The updated user target.
        """
        # TODO: Something is fucked up here.
        # we should be able to pass the db_result and db_result_exec to the answer worksheet
        # and use it to update the result
        # I don't want to execute it directly right now, that doesn't seem a good idea -- there can be random formatting
        # issues and other complications.
        if pattern_type == "func":
            # Include datatype in the Answer constructor if provided
            datatype_arg = f", datatype={repr(datatype)}" if datatype else ""
            answer_str = f"Answer({repr(suql_query)}, {unfilled_params}, {tables}, {repr(answer_query[1:-1])}{datatype_arg})"

            # Find and replace the original answer function call
            # The answer_query includes quotes, so we need to find the actual function call
            # query_without_quotes = answer_query[1:-1]  # Remove outer quotes

            # Try different patterns to find the original function call
            patterns = [
                # answer("query", datatype=SomeType)
                rf"answer\({re.escape(answer_query)},\s*datatype\s*=\s*[^,)]+\)",
                # answer(query="query", datatype=SomeType)
                rf"answer\(\s*query\s*=\s*{re.escape(answer_query)},\s*datatype\s*=\s*[^,)]+\)",
                # answer("query")
                rf"answer\({re.escape(answer_query)}\)",
                # answer(query="query")
                rf"answer\(\s*query\s*=\s*{re.escape(answer_query)}\)",
            ]

            for pattern in patterns:
                match = re.search(pattern, user_target)
                if match:
                    return user_target.replace(match.group(0), answer_str)

            # Fallback: if no pattern matches, try simple replacement
            return user_target.replace(f"answer({answer_query})", answer_str)
        else:
            answer_var = re.search(r"answer_(\d+)", user_target).group(0)
            # Include datatype in the update call if provided
            datatype_arg = f", datatype={repr(datatype)}" if datatype else ""
            answer_str = (
                f"{answer_var}.result = []\n"
                f"{answer_var}.update(query={repr(suql_query)}, "
                f"unfilled_params={unfilled_params}, tables={tables}, "
                f"query_str={repr(answer_query[1:-1])}{datatype_arg})"
            )
            return user_target.replace(
                f"{answer_var}.query = {answer_query}", answer_str
            )


class GenieParser:
    """A class that handles semantic parsing for the Genie conversational AI system.

    This class is responsible for converting natural language utterances into executable code,
    handling SQL queries, and managing the dialogue state.

    Attributes:
        bot (GenieRuntime): The bot runtime instance containing configuration and models.
        contextual_parser (ContextualSemanticParser): Parser for generating formal code.
        knowledge_parser (KnowledgeBaseParser): Parser for handling database queries.
    """

    def __init__(
        self,
        runtime: GenieRuntime,
        parser: "BaseKnowledgeParser",
        agent: "Agent",
    ):
        """Initialize the GenieParser.

        Args:
            runtime (GenieRuntime): The bot runtime instance.
        """
        self.runtime = runtime
        self.contextual_parser = ContextualSemanticParser(runtime, agent)
        self.knowledge_parser = KnowledgeBaseParser(runtime, parser)
        self.current_dir = os.path.dirname(__file__)
        self.agent = agent

    async def parse(
        self,
        current_dlg_turn: CurrentDialogueTurn,
        dlg_history: List[CurrentDialogueTurn],
    ) -> None:
        """Convert user utterance to worksheet representation and process it.

        Args:
            current_dlg_turn (CurrentDialogueTurn): The current dialogue turn.
            dlg_history (List[CurrentDialogueTurn]): The dialogue history.
        """
        # Convert the user utterance to worksheet representation
        user_formal, suql_target = await self._natural_language_to_formal(
            current_dlg_turn, dlg_history
        )

        current_dlg_turn.user_target_sp = user_formal
        current_dlg_turn.user_target_suql = "\n".join(suql_target)

        # Rewrite the code to extract function calls to variables
        genie_user_target = self._symbolic_code_rewrite(user_formal)

        current_dlg_turn.user_target = genie_user_target

    def _symbolic_code_rewrite(self, user_target: str) -> Optional[str]:
        """Extract function calls to variables using AST.

        Args:
            user_target (str): The user target code to rewrite.

        Returns:
            Optional[str]: The rewritten code with extracted function calls, or None if parsing fails.
        """
        valid_worksheets = [func.__name__ for func in self.runtime.genie_worksheets]
        valid_dbs = [func.__name__ for func in self.runtime.genie_db_models]

        valid_worksheets.extend(["Answer", "MoreFieldInfo"])
        var_counter = count_worksheet_variables(self.runtime.context.context)

        try:
            return rewrite_code_to_extract_funcs(
                user_target,
                valid_worksheets,
                valid_dbs,
                var_counter,
            )
        except SyntaxError as e:
            logger.info(f"SyntaxError: {e}")
            return None

    async def _natural_language_to_formal(
        self,
        current_dlg_turn: CurrentDialogueTurn,
        dlg_history: List[CurrentDialogueTurn],
    ) -> Tuple[str, List[str]]:
        """Convert natural language to executable code.

        Args:
            current_dlg_turn (CurrentDialogueTurn): The current dialogue turn.
            dlg_history (List[CurrentDialogueTurn]): The dialogue history.

        Returns:
            Tuple[str, List[str]]: A tuple containing the user target code and SUQL queries.
        """
        if current_dlg_turn.user_target_sp is None:
            input_data = prepare_semantic_parser_input(
                self.runtime, dlg_history, current_dlg_turn, self.agent.starting_prompt
            )
            (
                state_schema,
                agent_acts,
                agent_utterance,
                available_worksheets_text,
                available_dbs_text,
            ) = input_data

            user_target = await self.contextual_parser.generate_formal_representation(
                dlg_history,
                current_dlg_turn,
                state_schema,
                agent_acts,
                agent_utterance,
                available_worksheets_text,
                available_dbs_text,
            )
        else:
            user_target = current_dlg_turn.user_target_sp

        answer_queries, pattern_type = self._extract_answer_queries(user_target)
        suql_queries, user_target = await self.knowledge_parser.process_answer_queries(
            answer_queries, dlg_history, user_target, pattern_type
        )

        return user_target.strip(), suql_queries

    @staticmethod
    def _extract_answer_queries(
        text: str,
    ) -> Tuple[List[Tuple[str, Optional[str]]], str]:
        """Extract answer queries from the provided text using AST parsing.

        Args:
            text (str): The input text containing answer queries.

        Returns:
            Tuple[List[Tuple[str, Optional[str]]], str]: A tuple containing extracted (query, datatype) pairs and pattern type.
        """

        class AnswerQueryExtractor(ast.NodeVisitor):
            def __init__(self):
                self.queries = []
                self.pattern_type = "func"

            def visit_Call(self, node):
                """Visit function call nodes to find answer() calls."""
                if isinstance(node.func, ast.Name) and node.func.id == "answer":
                    # Extract string arguments from answer() calls
                    query = None
                    datatype = None

                    # Handle positional arguments: answer("query") or answer("query", datatype=...)
                    if node.args and isinstance(node.args[0], ast.Str):
                        query = node.args[0].s
                    elif (
                        node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                    ):
                        query = node.args[0].value

                    # Handle keyword arguments: answer(query="...", datatype="...")
                    for keyword in node.keywords:
                        if keyword.arg == "query":
                            if isinstance(keyword.value, ast.Str):
                                query = keyword.value.s
                            elif isinstance(keyword.value, ast.Constant) and isinstance(
                                keyword.value.value, str
                            ):
                                query = keyword.value.value
                        elif keyword.arg == "datatype":
                            if isinstance(keyword.value, ast.Str):
                                datatype = keyword.value.s
                            elif isinstance(keyword.value, ast.Constant) and isinstance(
                                keyword.value.value, str
                            ):
                                datatype = keyword.value.value
                            elif isinstance(keyword.value, ast.Name):
                                # Handle identifier names like Fund, str, int, etc.
                                datatype = keyword.value.id

                    if query:
                        self.queries.append((f'"{query}"', datatype))

                self.generic_visit(node)

            def visit_Assign(self, node):
                """Visit assignment nodes to find answer_X.query = "..." patterns."""
                if len(node.targets) == 1:
                    target = node.targets[0]

                    # Check if it's an attribute assignment: answer_X.query
                    if (
                        isinstance(target, ast.Attribute)
                        and target.attr == "query"
                        and isinstance(target.value, ast.Name)
                    ):
                        # Check if the variable name matches answer_\d+ pattern
                        var_name = target.value.id
                        if re.match(r"answer_\d+$", var_name):
                            # Extract the string value being assigned
                            if isinstance(node.value, ast.Str):
                                query = node.value.s
                                self.queries.append((f'"{query}"', None))
                                self.pattern_type = "attr"
                            elif isinstance(node.value, ast.Constant) and isinstance(
                                node.value.value, str
                            ):
                                query = node.value.value
                                self.queries.append((f'"{query}"', None))
                                self.pattern_type = "attr"

                self.generic_visit(node)

        try:
            # Parse the text as Python AST
            tree = ast.parse(text)
            extractor = AnswerQueryExtractor()
            extractor.visit(tree)

            return extractor.queries, extractor.pattern_type

        except SyntaxError:
            # Fallback to empty result if parsing fails
            logger.warning("Failed to parse code as AST, no answer queries extracted")
            return [], "func"
