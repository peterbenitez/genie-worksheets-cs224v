"""Agent action classes for the Genie system.

This module provides classes that represent different types of agent actions
in the Genie system, such as reporting results, asking questions, and proposing values.
"""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any, Optional

from worksheets.core.fields import GenieField


class AgentAct:
    """Base class for agent actions.

    This class serves as the foundation for different types of agent actions
    in the system.
    """

    pass


class ReportAgentAct(AgentAct):
    """Action for reporting query results or messages.

    This class handles reporting of query results and system messages.

    Attributes:
        query: The query being reported
        message: The message or result to report
        query_var_name: Variable name for the query
        message_var_name: Variable name for the message
    """

    def __init__(
        self,
        query: Optional[GenieField],
        message: Any,
        query_var_name: Optional[str] = None,
        message_var_name: Optional[str] = None,
        is_no_response: bool = False,
    ):
        """Initialize a report action.

        Args:
            query: The query being reported
            message: The message or result to report
            query_var_name: Variable name for the query
            message_var_name: Variable name for the message
            is_no_response: Flag indicating this is from no_response() (hallucination territory)
        """
        self.query = query
        self.message = message
        self.query_var_name = query_var_name
        self.message_var_name = message_var_name
        self.is_no_response = is_no_response

    def __repr__(self) -> str:
        query_var_name = self.query_var_name or self.query
        message_var_name = self.message_var_name or self.message
        return f"Report({query_var_name}, {message_var_name})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ReportAgentAct):
            return self.query == other.query and self.message == other.message
        return False


class AskAgentAct(AgentAct):
    """Action for requesting information from users.

    This class handles user information requests.

    Attributes:
        ws: The worksheet context
        field: The field to ask about
        ws_name: Worksheet name override
    """

    def __init__(self, ws: Any, field: GenieField, ws_name: Optional[str] = None):
        """Initialize an ask action.

        Args:
            ws: The worksheet context
            field: The field to ask about
            ws_name: Worksheet name override
        """
        self.ws = ws
        self.field = field
        self.ws_name = ws_name

    def __repr__(self) -> str:
        description = self._get_field_description()
        ws_name = self.ws_name or self.ws.__class__.__name__
        return f"AskField({ws_name}, {self.field.name}, '{description}')"

    def _get_field_description(self) -> Optional[str]:
        """Get the field description with enum options if applicable.

        Returns:
            Field description string or None
        """
        if inspect.isclass(self.field.slottype) and issubclass(
            self.field.slottype, Enum
        ):
            options = [x.name for x in list(self.field.slottype.__members__.values())]
            options = ", ".join(options)
            return self.field.description + f" Options are: {options}"
        return self.field.description if self.field.description else None


class ProposeAgentAct(AgentAct):
    """Action for proposing worksheet values.

    This class handles proposals for worksheet field values.

    Attributes:
        ws: The worksheet context
        params: Proposed parameters
        ws_name: Worksheet name override
    """

    def __init__(self, ws: Any, params: dict, ws_name: Optional[str] = None):
        """Initialize a propose action.

        Args:
            ws: The worksheet context
            params: Proposed parameters
            ws_name: Worksheet name override
        """
        self.ws = ws
        self.params = params
        self.ws_name = ws_name

    def __repr__(self) -> str:
        ws_name = self.ws_name or self.ws.__class__.__name__
        return f"ProposeAgentAct({ws_name}, {self.params})"


class AskForConfirmationAgentAct(AgentAct):
    """Action for requesting user confirmation.

    This class handles confirmation requests for field values.

    Attributes:
        ws: The worksheet context
        field: The field to confirm
        ws_name: Worksheet name override
        field_name: Field name override
        value: Value to confirm
    """

    def __init__(
        self,
        ws: Any,
        field: GenieField,
        ws_name: Optional[str] = None,
        field_name: Optional[str] = None,
    ):
        """Initialize a confirmation request action.

        Args:
            ws: The worksheet context
            field: The field to confirm
            ws_name: Worksheet name override
            field_name: Field name override
        """
        self.ws = ws
        self.field = field
        self.ws_name = ws_name
        self.field_name = field_name
        self.value = None

    def __repr__(self) -> str:
        ws_name = self.ws_name or self.ws.__class__.__name__
        field_name = self.field_name or self.field.name
        return f"AskForFieldConfirmation({ws_name}, {field_name})"


class AgentActs:
    """Container for managing multiple agent actions.

    This class manages collections of agent actions, handling action ordering
    and compatibility.

    Attributes:
        args: Arguments for action management
        actions: List of agent actions
    """

    def __init__(self, args: Any):
        """Initialize an agent actions container.

        Args:
            args: Arguments for action management
        """
        self.args = args
        self.actions = []

    def add(self, action: AgentAct):
        """Add an action if compatible.

        Args:
            action: The action to add
        """
        self._add(action)

    def _add(self, action: AgentAct):
        """Internal method to add an action.

        Args:
            action: The action to add
        """
        if self.should_add(action):
            self.actions.append(action)

    def should_add(self, incoming_action: AgentAct) -> bool:
        """Check if an action can be added based on compatibility rules.

        Args:
            incoming_action: The action to check

        Returns:
            True if the action can be added, False otherwise
        """
        acts_to_action = self._group_actions_by_type()

        if isinstance(incoming_action, ReportAgentAct):
            return self._can_add_report(incoming_action, acts_to_action)
        elif isinstance(incoming_action, ProposeAgentAct):
            return self._can_add_propose(incoming_action, acts_to_action)
        elif isinstance(incoming_action, (AskAgentAct, AskForConfirmationAgentAct)):
            return self._can_add_ask(acts_to_action)

        return False

    def _group_actions_by_type(self) -> dict:
        """Group existing actions by their type.

        Returns:
            Dictionary mapping action types to lists of actions
        """
        acts_to_action = {}
        for action in self.actions:
            action_type = action.__class__.__name__
            if action_type in acts_to_action:
                acts_to_action[action_type].append(action)
            else:
                acts_to_action[action_type] = [action]
        return acts_to_action

    def _can_add_report(
        self, incoming_action: ReportAgentAct, acts_to_action: dict
    ) -> bool:
        """Check if a report action can be added.

        Args:
            incoming_action: The report action to check
            acts_to_action: Grouped existing actions

        Returns:
            True if the action can be added, False otherwise
        """
        for action in acts_to_action.get("ReportAgentAct", []):
            if (
                action.query == incoming_action.query
                and action.message == incoming_action.message
            ):
                return False
        return True

    def _can_add_propose(
        self, incoming_action: ProposeAgentAct, acts_to_action: dict
    ) -> bool:
        """Check if a propose action can be added.

        Args:
            incoming_action: The propose action to check
            acts_to_action: Grouped existing actions

        Returns:
            True if the action can be added, False otherwise
        """
        if (
            "AskAgentAct" in acts_to_action
            or "AskForConfirmationAgentAct" in acts_to_action
        ):
            return False

        from worksheets.utils.worksheet import same_worksheet

        for action in acts_to_action.get("ProposeAgentAct", []):
            if action.params == incoming_action.params and same_worksheet(
                action.ws, incoming_action.ws
            ):
                return False
        return True

    def _can_add_ask(self, acts_to_action: dict) -> bool:
        """Check if an ask action can be added.

        Args:
            acts_to_action: Grouped existing actions

        Returns:
            True if an ask action can be added, False otherwise
        """
        return not any(
            act in acts_to_action
            for act in ["ProposeAgentAct", "AskAgentAct", "AskForConfirmationAgentAct"]
        )

    def extend(self, actions: list[AgentAct]):
        """Add multiple actions.

        Args:
            actions: List of actions to add
        """
        for action in actions:
            self._add(action)

    def __iter__(self):
        return iter(self.actions)

    def __next__(self):
        return next(self.actions)

    def can_have_other_acts(self) -> bool:
        """Check if more actions can be added.

        Returns:
            True if more actions can be added, False otherwise
        """
        acts_to_action = self._group_actions_by_type()
        return not any(
            act in acts_to_action
            for act in ["ProposeAgentAct", "AskAgentAct", "AskForConfirmationAgentAct"]
        )
