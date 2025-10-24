import datetime
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
from loguru import logger

from worksheets.core.dialogue import CurrentDialogueTurn
from worksheets.utils.annotation import get_agent_action_schemas, get_context_schema
from worksheets.llm.llm import get_llm_client
from worksheets.llm.prompts import load_fewshot_prompt_template
from worksheets.llm.logging import LoggingHandler
from langchain_core.output_parsers import StrOutputParser


class ResponsePromptManager:
    """Manages the preparation and handling of prompts for response generation.

    This class is responsible for gathering and organizing all necessary information
    for generating appropriate responses in the dialogue system.

    Attributes:
        runtime (GenieRuntime): The runtime runtime instance containing configuration and models.
    """

    def __init__(self, runtime, agent):
        """Initialize the ResponsePromptManager.

        Args:
            runtime (GenieRuntime): The runtime runtime instance.
            agent (Agent): The agent instance.
        """
        self.runtime = runtime
        self.agent = agent

    def _get_agent_acts(self, current_dlg_turn: CurrentDialogueTurn) -> List[Dict]:
        """Get the agent actions for the current turn.

        Args:
            current_dlg_turn (CurrentDialogueTurn): The current dialogue turn.

        Returns:
            List[Dict]: The agent actions.
        """
        if current_dlg_turn.system_action is None:
            return []
        return get_agent_action_schemas(
            current_dlg_turn.system_action, self.runtime.context
        )

    def prepare_prompt_inputs(
        self,
        current_dlg_turn: CurrentDialogueTurn,
        dlg_history: List[CurrentDialogueTurn],
    ) -> Dict[str, Any]:
        """Prepare all necessary inputs for the response generation prompt.

        Args:
            current_dlg_turn (CurrentDialogueTurn): The current dialogue turn.
            dlg_history (List[CurrentDialogueTurn]): The dialogue history.
            state_schema (str): The current state schema.
            agent_acts (List[Dict]): The agent actions.
            agent_utterance (str): The previous agent utterance.

        Returns:
            Dict[str, Any]: The prepared prompt inputs.
        """
        current_date = datetime.datetime.now()
        turns = dlg_history[-self.agent.config.rg_num_turns :]
        refusal = getattr(self.agent.config, 'refusal', False)

        return {
            "dlg_history": turns,
            "date": current_date.strftime("%Y-%m-%d"),
            "day": current_date.strftime("%A"),
            "state": get_context_schema(self.runtime.context, response_generator=True),
            "agent_acts": self._get_agent_acts(current_dlg_turn),
            "description": self.agent.description,
            "parsing": current_dlg_turn.user_target,
            "refusal": refusal,
        }


class ResponseSupervisor:
    """Supervises and validates generated responses.

    This class is responsible for checking the quality and appropriateness
    of generated responses, providing feedback and validation.
    """

    def __init__(self, agent):
        self.agent = agent

        self.validation_llm_client = get_llm_client(
            model=self.agent.config.response_generator.model_name,
            temperature=self.agent.config.response_generator.temperature,
            top_p=self.agent.config.response_generator.top_p,
            max_tokens=self.agent.config.response_generator.max_tokens,
        )
        self.validation_prompt_template = load_fewshot_prompt_template(
            "supervisor_response_generator.prompt"
        )
        self.validation_chain = self.validation_prompt_template | self.validation_llm_client

    async def validate_response(
        self,
        agent_response: str,
        prompt_inputs: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """Validate a generated response and provide feedback.

        Args:
            agent_response (str): The generated response to validate.
            prompt_inputs (Dict[str, Any]): The inputs used to generate the response.

        Returns:
            Tuple[bool, Optional[str]]: A tuple containing (is_valid, feedback).
        """
        logging_handler = LoggingHandler(
            prompt_file="supervisor_response_generator.prompt",
            metadata={
                "agent_response": agent_response,
                "prompt_inputs": prompt_inputs,
            },
            session_id=self.agent.session_id,
        )
        prompt_inputs["agent_response"] = agent_response

        validation_output = await self.validation_chain.ainvoke(prompt_inputs, config={"callbacks": [logging_handler]})

        return ResponseSupervisor._parse_validation_output(validation_output)

    @staticmethod
    def _parse_validation_output(validation_output: str) -> Tuple[bool, Optional[str]]:
        """Parse the validation output from the supervisor.

        Args:
            validation_output (str): The raw validation output.

        Returns:
            Tuple[bool, Optional[str]]: A tuple containing (is_valid, feedback).
        """
        bs = BeautifulSoup(validation_output, "html.parser")
        answer = bs.find("answer")
        feedback = bs.find("reasoning")

        is_valid = False
        feedback_text = None

        if answer is not None:
            answer_text = answer.text.lower().strip()
            is_valid = answer_text == "true"

        if feedback is not None:
            feedback_text = feedback.text.lower().strip()

        return is_valid, feedback_text


class ResponseGenerator:
    """Main class for generating dialogue system responses.

    This class orchestrates the process of generating appropriate responses
    based on the current dialogue state, history, and system actions.

    Attributes:
        runtime (GenieRuntime): The runtime instance.
        prompt_manager (ResponsePromptManager): Manager for prompt preparation.
        supervisor (ResponseSupervisor): Supervisor for response validation.
    """

    def __init__(self, runtime, agent):
        """Initialize the ResponseGenerator.

        Args:
            runtime (GenieRuntime): The runtime runtime instance.
            validate_response (bool): Whether to validate the response.
        """
        self.runtime = runtime
        self.agent = agent
        self.prompt_manager = ResponsePromptManager(runtime, agent)
        self.validate_response = agent.config.validate_response
        if self.validate_response:
            self.supervisor = ResponseSupervisor(agent)
        else:
            self.supervisor = None

        self.llm_client = get_llm_client(
            model=self.agent.config.response_generator.model_name,
            temperature=self.agent.config.response_generator.temperature,
            top_p=self.agent.config.response_generator.top_p,
            max_tokens=self.agent.config.response_generator.max_tokens,
        )
        self.prompt_template = load_fewshot_prompt_template(
            "response_generator.prompt"
        )
        self.chain = self.prompt_template | self.llm_client | StrOutputParser()

    async def generate_response(
        self,
        current_dlg_turn: CurrentDialogueTurn,
        dlg_history: List[CurrentDialogueTurn],
    ) -> None:
        """Generate a response for the current dialogue turn.

        Args:
            current_dlg_turn (CurrentDialogueTurn): The current dialogue turn.
            dlg_history (List[CurrentDialogueTurn]): The dialogue history.
        """
        # Prepare prompt inputs
        prompt_inputs = self.prompt_manager.prepare_prompt_inputs(
            current_dlg_turn,
            dlg_history,
        )
        logging_handler = LoggingHandler(
            prompt_file="response_generator.prompt",
            metadata={
                "prompt_inputs": prompt_inputs,
            },
            session_id=self.agent.session_id,
        )

        # Generate response
        response = await self.chain.ainvoke(prompt_inputs, config={"callbacks": [logging_handler]})

        # Update dialogue turn
        current_dlg_turn.system_response = response

        # Optionally validate response
        if self.validate_response:
            await self._validate_response(response, prompt_inputs)

    async def _validate_response(
        self,
        response: str,
        prompt_inputs: Dict[str, Any],
    ) -> None:
        """Validate the generated response.

        Args:
            response (str): The generated response.
            prompt_inputs (Dict[str, Any]): The prompt inputs used for generation.
        """
        is_valid, feedback = await self.supervisor.validate_response(
            response, prompt_inputs
        )

        if not is_valid:
            logger.warning(f"Response validation failed: {feedback}")
            # TODO: Implement response regeneration or fallback strategy
