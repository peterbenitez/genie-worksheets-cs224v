import inspect
from enum import Enum
from typing import Any

from worksheets.core.agent_acts import ProposeAgentAct, ReportAgentAct
from worksheets.core.context import GenieContext
from worksheets.core.fields import GenieField, GenieValue
from worksheets.core.worksheet import GenieWorksheet
from worksheets.utils.field import get_genie_fields_from_ws


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
    """Create a cannot answer action.

    Args:
        message (str): The message to report.
    """
    return ReportAgentAct(None, "Refuse to answer the question")

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
