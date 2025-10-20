
from __future__ import annotations
import json
import uuid
from typing import Optional, Type, TYPE_CHECKING

from worksheets.agent.config import Config
from worksheets.components.agent_policy import AgentPolicyManager
from worksheets.components.response_generator import ResponseGenerator
from worksheets.components.semantic_parser import GenieParser
from worksheets.core import GenieContext, GenieRuntime
from worksheets.core.dialogue import CurrentDialogueTurn
from worksheets.specification.from_spreadsheet import specification_to_genie

if TYPE_CHECKING:
    from worksheets.knowledge.base import BaseKnowledgeBase
    from worksheets.knowledge.parser import BaseKnowledgeParser


class Agent:
    """Agent setting for GenieWorksheets"""

    def __init__(
        self,
        # name of the agent
        botname: str,
        # description of the agent. This is used for generating response
        description: str,
        # starting prompt for the agent to ask the user
        starting_prompt: str,
        # arguments to pass to the agent for configuration
        config: Config,
        # list of functions that are available to the agent for execution
        api: list,
        # knowledge configuration for the agent to run queries and respond to the user
        knowledge_base: BaseKnowledgeBase,
        # semantic parser for knowledge queries
        knowledge_parser: BaseKnowledgeParser,
        # contextual semantic parser
        genie_parser_class: Optional[Type[GenieParser]] = GenieParser,
        # agent policy manager
        genie_agent_policy_class: Optional[
            Type[AgentPolicyManager]
        ] = AgentPolicyManager,
        # response generator
        genie_response_generator_class: Optional[
            Type[ResponseGenerator]
        ] = ResponseGenerator,
    ):
        self.botname = botname
        self.description = description
        self.starting_prompt = starting_prompt
        self.config = config
        self.api = api
        self.knowledge_base = knowledge_base
        self.knowledge_parser = knowledge_parser

        self.runtime = None
        self.dlg_history = []
        self.genie_parser_class = genie_parser_class
        self.genie_agent_policy_class = genie_agent_policy_class
        self.genie_response_generator_class = genie_response_generator_class

        self.session_id = str(uuid.uuid4())

    # ── context-manager hooks ─────────────────────────────────────────
    def __enter__(self):
        """Enter the agent context manager."""
        return self.enter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the agent context manager and save logs and conversation."""
        try:
            self.close()
        except Exception as e:
            # Log the error but don't suppress the original exception
            print(f"Warning: Failed to save logs during agent shutdown: {e}")
        # Don't suppress exceptions - re-raise any error that happened inside
        return False

    def enter(self):
        """Enter the agent context manager."""
        return self

    def close(self):
        """Close the agent and save logs and conversation."""
        self._save_conversation_json()


    def _save_conversation_json(self) -> None:
        """Save conversation history to JSON file.

        Args:
            path: Custom file path, defaults to {botname}_conversation.json
        """
        if self.config.conversation_log_path is not None:
            import os

            prompt_dir = os.path.dirname(self.config.conversation_log_path)
            if prompt_dir and not os.path.exists(prompt_dir):
                os.makedirs(prompt_dir, exist_ok=True)

        if self.config.conversation_log_path is None:
            import datetime

            self.config.conversation_log_path = f"{self.botname.lower().replace(' ', '_')}_conversation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        from worksheets.utils.interface import convert_to_json

        write_mode = "w"
        if self.config.append_to_conversation_log:
            write_mode = "a"

        with open(self.config.conversation_log_path, write_mode, encoding="utf-8") as f:
            json.dump(
                convert_to_json(self.dlg_history, self.session_id),
                f,
                indent=4,
                default=str,
            )

    def load_runtime_from_specification(
        self,
        csv_path: str | None = None,
        gsheet_id: str | None = None,
        json_path: str | None = None,
    ):
        """Load the agent configuration from the specification.

        Args:
            csv_path (str): The path to the CSV file.
            gsheet_id (str): The ID of the Google Sheet.
            json_path (str): The path to the JSON file.
        Returns:
            GenieRuntime: An instance of GenieRuntime configured with the loaded data.
        """

        # Load Genie worksheets, databases, and types from the Google Sheet
        genie_worsheets, genie_dbs, genie_types = specification_to_genie(
            csv_path=csv_path, gsheet_id=gsheet_id, json_path=json_path
        )

        # Create a SUQL runner if knowledge_base is provided. Suql runner is used by the
        # GenieRuntime to run queries against the knowledge base.
        if self.knowledge_base:

            def suql_runner(query, *args, **kwargs):
                return self.knowledge_base.run(query, *args, **kwargs)

        else:
            suql_runner = None

        # Create an instance of GenieRuntime with the loaded configuration
        runtime = GenieRuntime(
            config=self.config,
            api=self.api,
            suql_runner=suql_runner,
            agent=self,
        )

        # Add worksheets, databases, and types to the GenieRuntime instance
        for worksheet in genie_worsheets:
            runtime.add_worksheet(worksheet)

        for db in genie_dbs:
            runtime.add_db_model(db)

        for genie_type in genie_types:
            runtime.add_worksheet(genie_type)

        self.runtime = runtime

        self._initialize_modules()

    def _initialize_modules(self):
        self.genie_parser = self.genie_parser_class(
            self.runtime, self.knowledge_parser, self
        )
        self.genie_agent_policy_manager = self.genie_agent_policy_class(self.runtime)
        self.genie_response_generator = self.genie_response_generator_class(
            self.runtime, self
        )

    async def generate_next_turn(self, user_utterance: str):
        """Generate the next turn in the dialogue based on the user's utterance.

        Args:
            user_utterance (str): The user's input.
            bot (Agent): The bot instance handling the dialogue.
        """
        # instantiate a new dialogue turn
        current_dlg_turn = CurrentDialogueTurn()
        current_dlg_turn.user_utterance = user_utterance

        # initialize contexts
        current_dlg_turn.context = GenieContext()
        current_dlg_turn.global_context = GenieContext()

        # reset the agent acts
        self.runtime.context.reset_agent_acts()

        # process the dialogue turn to GenieWorksheets
        await self.genie_parser.parse(current_dlg_turn, self.dlg_history)

        # run the agent policy if user_target is not None
        if current_dlg_turn.user_target is not None:
            self.genie_agent_policy_manager.run_policy(current_dlg_turn)

        # generate a response based on the agent policy
        await self.genie_response_generator.generate_response(
            current_dlg_turn, self.dlg_history
        )
        self.dlg_history.append(current_dlg_turn)
