import chainlit as cl

from worksheets.agent.agent import Agent
from worksheets.core import GenieContext
from worksheets.core.dialogue import CurrentDialogueTurn
from worksheets.utils.annotation import get_agent_action_schemas, get_context_schema


class ChainlitAgent(Agent):
    async def generate_next_turn(self, user_utterance: str):
        """Generate the next turn in the dialogue based on the user's utterance for chainlit frontend.

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
        async with cl.Step(
            name="Semantic Parsing",
            type="semantic_parser",
            language="python",
            show_input=True,
        ) as step:
            current_dlg_turn.context = GenieContext()
            current_dlg_turn.global_context = GenieContext()
            await self.genie_parser.parse(current_dlg_turn, self.dlg_history)
            step.output = current_dlg_turn.user_target_sp

        # run the agent policy
        async with cl.Step(
            name="Agent Policy",
            type="agent_policy",
            language="python",
            show_input=True,
        ) as step:
            await cl.make_async(self.genie_agent_policy_manager.run_policy)(
                current_dlg_turn
            )
            step.input = current_dlg_turn.user_target
            step.output = get_context_schema(self.runtime.context)

        # generate a response based on the agent policy
        async with cl.Step(
            name="Generating Response",
            type="response_generator",
            language="json",
            show_input=True,
        ) as step:
            await self.genie_response_generator.generate_response(
                current_dlg_turn, self.dlg_history
            )
            # step.output = get_context_schema(self.runtime.context)
            step.output = get_agent_action_schemas(
                current_dlg_turn.system_action, self.runtime.context
            )
            self.dlg_history.append(current_dlg_turn)
