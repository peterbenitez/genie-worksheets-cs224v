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
try:
    import sys
    from pathlib import Path
    
    # Add hallucinations directory to path if not already there
    hallucinations_path = Path(__file__).parent.parent.parent.parent / "experiments/domain_agents/investment_agent/hallucinations"
    if hallucinations_path.exists() and str(hallucinations_path) not in sys.path:
        sys.path.insert(0, str(hallucinations_path))
    
    from tool_discovery import ToolRegistry, discover_missing_tool
    _tool_discovery_available = True
    logger.info("Tool discovery system available")
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
    
    Now with synchronous tool discovery!
    
    When the agent can't answer with existing tools, this function:
    1. Captures the user's query
    2. Uses AI to determine what missing tool would solve it
    3. Records the missing tool specification with frequency tracking
    4. Persists discoveries to disk for analysis
    
    Args:
        message (str): The hallucinated response from semantic parser
        
    Returns:
        ReportAgentAct: The report action with is_no_response=True flag
    """
    # ════════════════════════════════════════════════════
    # TOOL DISCOVERY INTEGRATION
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
                
                # Synchronous tool discovery
                tool_spec, is_new, is_update = discover_missing_tool(
                    user_utterance=user_utterance,
                    registry=_current_runtime.tool_registry
                )
                
                # Get additional context about the conversation
                turn_number = getattr(_current_runtime, 'current_turn_number', None)
                allow_hallucination = getattr(_current_runtime.config, 'allow_hallucination', None)
                investor_profile_name = getattr(_current_runtime, 'investor_profile_name', None)
                user_risk_profile = getattr(_current_runtime, 'user_risk_profile', None)  
                
                # Record the missing tool (increments frequency if existing, or updates version if modified)
                _current_runtime.tool_registry.record_missing_tool(
                    tool_spec=tool_spec,
                    context={
                        'user_utterance': user_utterance,
                        'timestamp': datetime.now().isoformat(),
                        'hallucinated_response': message,
                        'is_new_discovery': is_new,
                        'is_update': is_update,
                        'turn_number': turn_number,
                        'allow_hallucination': allow_hallucination,
                        'investor_profile': investor_profile_name,
                        'user_risk_profile': user_risk_profile,
                    },
                    is_update=is_update
                )
                
                # Save to disk after each discovery
                _current_runtime.tool_registry.save_to_disk()
                
                # Log with appropriate status
                if is_new:
                    status = 'Created'
                elif is_update:
                    status = f'Updated to v{tool_spec.version}'
                else:
                    status = 'Reused'
                
                logger.info(
                    f"Tool Discovery: {status} '{tool_spec.tool_name}' (freq={tool_spec.frequency})"
                )
            
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
