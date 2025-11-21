import inspect
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from loguru import logger

from worksheets.core.agent_acts import ProposeAgentAct, ReportAgentAct
from worksheets.core.context import GenieContext
from worksheets.core.fields import GenieField, GenieValue
from worksheets.core.worksheet import GenieWorksheet
from worksheets.utils.field import get_genie_fields_from_ws

# Global reference to the current runtime (set by GenieRuntime.__init__)
_current_runtime: Optional['GenieRuntime'] = None  # type: ignore

# Tool discovery integration (optional - only loaded if available)
_tool_discovery_available = False
_data_availability_available = False
try:
    import sys
    from pathlib import Path

    # Add hallucinations directory to path if not already there
    hallucinations_path = Path(__file__).parent.parent.parent.parent / "experiments/domain_agents/investment_agent/hallucinations"
    if hallucinations_path.exists() and str(hallucinations_path) not in sys.path:
        sys.path.insert(0, str(hallucinations_path))

    from tool_discovery import ToolRegistry, discover_missing_tool, discover_missing_capability
    from data_availability import DataAvailabilityChecker
    _tool_discovery_available = True
    _data_availability_available = True
    logger.info("Tool discovery system with data availability checking available")
except ImportError:
    logger.debug("Tool discovery system not available (module not found)")
except Exception as e:
    logger.warning(f"Tool discovery system not available: {e}")


def propose(worksheet: GenieWorksheet, params: dict) -> ProposeAgentAct:
    """Create a proposal action.

    Args:
        worksheet (GenieWorksheet): The worksheet to propose values for.
        params (dict): The parameters to propose.

    Returns:
        ProposeAgentAct: The created proposal action.
    """
    return ProposeAgentAct(worksheet(**params), params)


def say(message: str) -> ReportAgentAct:
    """Create a message report action.

    Args:
        message (str): The message to report.

    Returns:
        ReportAgentAct: The created report action.
    """
    return ReportAgentAct(None, message)


def generate_clarification(worksheet: GenieWorksheet, field: str) -> str:
    """Generate clarification text for a field.

    Args:
        worksheet (GenieWorksheet): The worksheet containing the field.
        field (str): The name of the field.

    Returns:
        str: The generated clarification text.
    """
    for f in get_genie_fields_from_ws(worksheet):
        if f.name == field:
            if inspect.isclass(f.slottype) and issubclass(f.slottype, Enum):
                options = [x.name for x in list(f.slottype.__members__.values())]
                options = ", ".join(options)
                option_desc = f.description + f" Options are: {options}"
                return option_desc
            return f.description

    return ""


def no_response(message: str) -> ReportAgentAct:
    """
    Create a hallucinated response when agent lacks proper tools.

    Hybrid approach: Data availability checking + AI-powered capability gap analysis

    When the agent can't answer with existing tools, this function:
    Phase 1: Check if it's a data constraint issue (temporal, missing columns, etc.)
    Phase 2: Use AI to classify: missing tool, out of scope, or ambiguous
    Phase 3: Record the appropriate issue type and persist to disk

    Args:
        message (str): The hallucinated response from semantic parser

    Returns:
        ReportAgentAct: The report action with is_no_response=True flag
    """
    # ════════════════════════════════════════════════════
    # HYBRID TOOL DISCOVERY INTEGRATION
    # ════════════════════════════════════════════════════

    if _tool_discovery_available and _current_runtime is not None:
        try:
            # Get current user utterance from dialogue history
            user_utterance = _get_current_user_utterance()

            if user_utterance:
                # Get or initialize tool registry
                if not hasattr(_current_runtime, 'tool_registry'):
                    _current_runtime.tool_registry = ToolRegistry.load_or_create()
                    logger.info("Initialized tool registry for agent")

                # PHASE 1: Check data availability (NEW)
                data_result = None
                if _data_availability_available:
                    checker = DataAvailabilityChecker(_current_runtime.tool_registry, _current_runtime)
                    query_context = _get_query_context()
                    data_result = checker.check_availability(user_utterance, query_context)

                    if data_result.issue_type != "no_issue":
                        logger.info(f"Data constraint detected: {data_result.issue_type}")

                # PHASE 2: Capability gap analysis (ENHANCED)
                issue = discover_missing_capability(
                    user_utterance=user_utterance,
                    registry=_current_runtime.tool_registry,
                    data_availability_result=data_result
                )

                # Get additional context about the conversation
                turn_number = getattr(_current_runtime, 'current_turn_number', None)
                allow_hallucination = getattr(_current_runtime.config, 'allow_hallucination', None)
                investor_profile_name = getattr(_current_runtime, 'investor_profile_name', None)
                user_risk_profile = getattr(_current_runtime, 'user_risk_profile', None)

                context = {
                    'user_utterance': user_utterance,
                    'timestamp': datetime.now().isoformat(),
                    'hallucinated_response': message,
                    'turn_number': turn_number,
                    'allow_hallucination': allow_hallucination,
                    'investor_profile': investor_profile_name,
                    'user_risk_profile': user_risk_profile,
                }

                # PHASE 3: Record based on issue type
                if issue.issue_type == "missing_tool" and issue.tool_spec:
                    _current_runtime.tool_registry.record_missing_tool(issue.tool_spec, context)
                    logger.info(
                        f"Missing Tool: '{issue.tool_spec.tool_name}' (freq={issue.tool_spec.frequency})"
                    )
                elif issue.issue_type == "missing_data":
                    _current_runtime.tool_registry.record_data_constraint(issue, context)
                    logger.info(f"Data Constraint: {issue.data_constraint}")
                    # Use the specific data constraint explanation
                    if issue.user_facing_message:
                        message = issue.user_facing_message
                elif issue.issue_type == "out_of_scope":
                    _current_runtime.tool_registry.record_out_of_scope(issue, context)
                    logger.info(f"Out of Scope: {issue.scope_explanation}")
                    # Use the specific out-of-scope message
                    if issue.user_facing_message:
                        message = issue.user_facing_message
                elif issue.issue_type == "ambiguous":
                    # Don't record ambiguous queries, just log them
                    logger.info(f"Ambiguous query: {issue.technical_explanation}")
                    # Use the clarification message
                    if issue.user_facing_message:
                        message = issue.user_facing_message

                # Save to disk after each discovery
                _current_runtime.tool_registry.save_to_disk()

        except Exception as e:
            # Tool discovery failed - raise error as requested
            logger.error(f"Tool discovery failed: {e}")
            raise RuntimeError(f"Tool discovery system failed: {e}") from e
    
    # ════════════════════════════════════════════════════
    # NORMAL no_response BEHAVIOR
    # ════════════════════════════════════════════════════
    
    # Let the agent hallucinate instead of refusing
    # The LLM-generated message is passed through as the hallucinated response
    # This allows collecting potentially incorrect responses in ambiguous territory
    hallucinated_response = message

    action = ReportAgentAct(None, hallucinated_response, is_no_response=True)

    # Add action to runtime's agent_acts container
    if _current_runtime is not None:
        _current_runtime.context.agent_acts.add(action)
    
    return action


def _get_current_user_utterance() -> Optional[str]:
    """
    Helper to extract current user utterance from runtime.

    Returns:
        The user's current utterance, or None if not available
    """
    if _current_runtime is None:
        return None

    # Try to get from agent's dialogue history
    if hasattr(_current_runtime, 'agent') and hasattr(_current_runtime.agent, 'dlg_history'):
        dlg_history = _current_runtime.agent.dlg_history
        if dlg_history and len(dlg_history) > 0:
            # Get the most recent turn's user utterance
            last_turn = dlg_history[-1]
            if hasattr(last_turn, 'user_utterance'):
                return last_turn.user_utterance

    return None


def _get_query_context() -> dict:
    """
    Helper to extract query context from runtime for data availability checking.

    Returns:
        Dictionary with context information (attempted queries, tables accessed, etc.)
    """
    context = {}

    if _current_runtime is None:
        return context

    # Try to get attempted SQL queries or API calls from the last turn
    if hasattr(_current_runtime, 'agent') and hasattr(_current_runtime.agent, 'dlg_history'):
        dlg_history = _current_runtime.agent.dlg_history
        if dlg_history and len(dlg_history) > 0:
            last_turn = dlg_history[-1]

            # Extract any attempted queries or operations
            if hasattr(last_turn, 'intermediate_output'):
                context['intermediate_output'] = last_turn.intermediate_output

            if hasattr(last_turn, 'attempted_queries'):
                context['attempted_queries'] = last_turn.attempted_queries

    return context

def chitchat() -> ReportAgentAct:
    """Create a chitchat action.
    """
    return ReportAgentAct(None, "Chit chat with the user")


def state_response(message: str) -> ReportAgentAct:
    """Create a state answer action.

    Args:
        message (str): The message to report.
    """
    return ReportAgentAct(None, message)


def answer_clarification_question(
    worksheet: GenieField, field: GenieField, context: GenieContext
) -> ReportAgentAct:
    """Create a clarification answer action.

    Args:
        worksheet (GenieField): The worksheet field.
        field (GenieField): The field to clarify.
        context (GenieContext): The context.

    Returns:
        ReportAgentAct: The created clarification report action.
    """
    ws = context.context[worksheet.value]
    return ReportAgentAct(
        f"AskClarification({worksheet.value}, {field.value})",
        generate_clarification(ws, field.value),
    )


def confirm(value: Any) -> GenieValue:
    """Create a confirmed value.

    Args:
        value (Any): The value to confirm.

    Returns:
        GenieValue: The confirmed value instance.
    """
    if isinstance(value, GenieValue):
        return value.confirm()
    elif isinstance(value, GenieField):
        return GenieValue(value.value).confirm()
    return GenieValue(value).confirm()
