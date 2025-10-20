"""Core base classes for the Genie worksheet system.

This module provides the foundational classes that form the backbone of the Genie worksheet system,
including base classes for worksheets, fields, and values.
"""

from __future__ import annotations

import inspect
from copy import deepcopy
from enum import Enum
from typing import Any, Optional, Tuple, Type, TYPE_CHECKING

from bs4 import BeautifulSoup
from loguru import logger

from worksheets.llm.llm import get_llm_client
from worksheets.llm.prompts import load_fewshot_prompt_template
from worksheets.llm.logging import LoggingHandler
from worksheets.utils.logging_config import log_validation_result


if TYPE_CHECKING:
    from worksheets.core.runtime import GenieRuntime
    from worksheets.core.context import GenieContext

class GenieValue:
    """A wrapper class for primitive values in Genie with confirmation tracking.

    This class wraps primitive values (string, int, float, etc.) and adds functionality
    like confirmation status tracking.

    Attributes:
        value: The wrapped primitive value
        confirmed (bool): Whether this value has been confirmed by the user
    """

    def __init__(self, value: Any):
        """Initialize a GenieValue.

        Args:
            value: The primitive value to wrap
        """
        # logger.debug(f"Creating GenieValue with value: {value}")
        self.value = value
        self.confirmed = False

    def __repr__(self) -> str:
        return f"{self.value}"

    def __eq__(self, other: Any) -> bool:
        # logger.debug(f"Comparing GenieValue {self.value} with {other}")
        if isinstance(other, GenieValue):
            return self.value == other.value
        return self.value == other

    def confirm(self, confirmed: bool = True) -> GenieValue:
        """Mark the value as confirmed.

        Args:
            confirmed: Whether to mark as confirmed. Defaults to True.

        Returns:
            The confirmed value instance
        """
        # logger.debug(
        #     f"Setting confirmation status to {confirmed} for value: {self.value}"
        # )
        self.confirmed = confirmed
        return self

    def __str__(self) -> str:
        return str(self.value)

    def __hash__(self) -> int:
        return hash(self.value)


class GenieResult(GenieValue):
    """A class to represent results from executions.

    This class extends GenieValue to store results from Answer executions or
    other actions, maintaining references to parent objects.

    Attributes:
        value: The result value
        parent: The parent object that produced this result
        parent_var_name: The variable name of the parent in the context
    """

    def __init__(self, value: Any, parent: Any, parent_var_name: str):
        """Initialize a GenieResult.

        Args:
            value: The result value
            parent: The parent object that produced this result
            parent_var_name: The variable name of the parent in the context
        """
        # logger.debug(f"Creating GenieResult with value: {value}")
        # logger.debug(f"Parent: {parent.__class__.__name__}")
        # logger.debug(f"Parent variable name: {parent_var_name}")

        super().__init__(value)
        self.parent = parent
        self.parent_var_name = parent_var_name

    def __getitem__(self, item: Any) -> Any:
        logger.debug(f"Accessing item {item} from GenieResult")
        return self.value[item]


class GenieField:
    """A class representing a field in a Genie worksheet.

    This class handles field definitions, validation, and value management for
    worksheet fields. It supports various field types, validation rules, and
    action triggers.

    Attributes:
        slottype: The type of the field
        name: The field name
        question: Question to ask when field needs filling
        description: Field description for LLM understanding
        predicate: Condition for field relevance
        ask: Whether to ask user for this field
        optional: Whether field is optional
        actions: Actions to perform when field is filled
        requires_confirmation: Whether field needs confirmation
        internal: Whether field is system-managed
        primary_key: Whether field is a primary key
        validation: Validation criteria
        parent: Parent worksheet
        bot: Associated bot instance
    """

    def __init__(
        self,
        slottype: Type | str,
        name: str,
        question: str = "",
        description: str = "",
        predicate: str = "",
        ask: bool = True,
        optional: bool = False,
        actions: Any = None,
        value: Any = None,
        requires_confirmation: bool = False,
        internal: bool = False,
        primary_key: bool = False,
        confirmed: bool = False,
        validation: Optional[str] = None,
        parent: Any = None,
        bot: Any = None,
        action_performed: bool = False,
        **kwargs: Any,
    ):
        """Initialize a GenieField.

        Args:
            slottype: The type of the field
            name: The field name
            question: Question to ask when field needs filling
            description: Field description for LLM understanding
            predicate: Condition for field relevance
            ask: Whether to ask user for this field
            optional: Whether field is optional
            actions: Actions to perform when field is filled
            value: Initial value
            requires_confirmation: Whether field needs confirmation
            internal: Whether field is system-managed
            primary_key: Whether field is a primary key
            confirmed: Whether field has been confirmed
            validation: Validation criteria
            parent: Parent worksheet
            bot: Associated bot instance
            action_performed: Whether action has been performed
            **kwargs: Additional keyword arguments
        """
        # logger.debug(f"Creating GenieField: {name}")
        # logger.debug(f"Type: {slottype}")
        # logger.debug(f"Initial value: {value}")

        self.predicate = predicate
        self.slottype = slottype
        self.name = name
        self.question = question
        self.ask = ask
        self.optional = optional if ask else True
        self.actions = actions
        self.requires_confirmation = requires_confirmation
        self.internal = internal
        self.description = description
        self.primary_key = primary_key
        self.validation = validation
        self.parent = parent
        self.bot = bot
        self.action_performed = action_performed
        self._value = self.init_value(value)
        self._confirmed = confirmed

        # logger.debug(f"Field {name} initialized with attributes:")
        # logger.debug(f"  Optional: {self.optional}")
        # logger.debug(f"  Requires confirmation: {self.requires_confirmation}")
        # logger.debug(f"  Internal: {self.internal}")
        # logger.debug(f"  Primary key: {self.primary_key}")
        # logger.debug(f"  Has validation: {self.validation is not None}")

    def __getattr__(self, item: str) -> Any:
        """Get an attribute of the field.

        Args:
            item: The attribute name

        Returns:
            The attribute value
        """
        from worksheets.core.worksheet import GenieWorksheet

        # Delegate to value if it’s a Field instance
        if isinstance(self.value, GenieWorksheet):
            # logger.debug(f"Getting attribute {item} from GenieWorksheet")
            x = getattr(self.value, item)
            return x
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{item}'"
        )

    def __setattr__(self, key: str, value: Any) -> None:
        """Set an attribute of the field.

        Args:
            key: The attribute name
            value: The attribute value
        """
        try:
            # hack to avoid recursion, check all the attributes.
            if key not in [
                "predicate",
                "slottype",
                "name",
                "question",
                "ask",
                "optional",
                "actions",
                "requires_confirmation",
                "internal",
                "description",
                "primary_key",
                "validation",
                "parent",
                "bot",
                "action_performed",
                "_value",
                "_confirmed",
                "value",
            ]:
                if isinstance(getattr(self, key), GenieField):
                    getattr(self, key).value = value
            else:
                super().__setattr__(key, value)
        except Exception as e:
            logger.error(
                f"Error setting attribute {key} for field {self.name}: {str(e)}"
            )
            raise

        # # Special case for initialization and for setting _value directly
        # # using __dict__ to avoid setattr and getattr recursion
        # if key == "_value" or "_value" not in self.__dict__:
        #     # Use parent's __setattr__ to avoid recursion
        #     super().__setattr__(key, value)
        #     return

        # if isinstance(getattr(self, key), GenieField):
        #     getattr(self, key).value = value
        # else:
        #     super().__setattr__(key, value)

    def __deepcopy__(self, memo: dict) -> GenieField:
        """create a deep copy of the field.

        Args:
            memo: Dictionary Instancedy copied objects

        Returns:
            A new GenieField instance
        """
        # logger.debug(f"Creating deep copy of field: {self.name}")
        try:
            new_field = GenieField(
                slottype=deepcopy(self.slottype, memo),
                name=deepcopy(self.name, memo),
                question=deepcopy(self.question, memo),
                description=deepcopy(self.description, memo),
                predicate=deepcopy(self.predicate, memo),
                ask=deepcopy(self.ask, memo),
                optional=deepcopy(self.optional, memo),
                actions=deepcopy(self.actions, memo),
                value=deepcopy(self.value, memo),
                requires_confirmation=deepcopy(self.requires_confirmation, memo),
                internal=deepcopy(self.internal, memo),
                primary_key=deepcopy(self.primary_key, memo),
                confirmed=deepcopy(self.confirmed, memo),
                validation=deepcopy(self.validation, memo),
                action_performed=deepcopy(self.action_performed, memo),
                parent=self.parent,
                bot=self.bot,
            )
            # logger.debug(f"Successfully created deep copy of field: {self.name}")
            return new_field
        except Exception as e:
            logger.error(f"Error creating deep copy of field {self.name}: {str(e)}")
            raise

    def __getitem__(self, item: Any) -> Any:
        """Get an item from the field.
        """
        if isinstance(self.value, list):
            return self.value[item]
        elif isinstance(self.value, dict):
            return self.value[item]
        else:
            raise ValueError(f"Cannot get item {item} from field {self.name}")

    def perform_action(self, runtime: GenieRuntime, local_context: GenieContext):
        """Perform the action associated with this field if it hasn't been performed yet.

        Args:
            bot (GenieRuntime): The bot instance.
            local_context (GenieContext): The local context for the action.

        Returns:
            list: A list of actions performed.
        """
        if self.action_performed:
            return []
        logger.info(f"Peforming action for {self.name}: {self.actions.action}")
        acts = []

        # If there are no actions, return an empty list
        if self.actions is None or len(self.actions) == 0:
            return acts

        # Perform the action
        acts = self.actions.perform(self, runtime, local_context)
        self.action_performed = True

        return acts

    def __repr__(self) -> str:
        return self.schema(value=True)

    def schema(self, value: bool = True) -> str:
        """Generate a schema representation of the field.

        Args:
            value: Whether to include the value in the schema

        Returns:
            The schema representation
        """
        # logger.debug(f"Generating schema for field: {self.name}")
        try:
            if isinstance(self.slottype, str) and self.slottype == "confirm":
                slottype = "bool"
            elif self.slottype.__name__ in ["List", "Dict"]:
                slottype = self.slottype.__name__ + "["
                if isinstance(self.slottype.__args__[0], str):
                    slottype += self.slottype.__args__[0]
                else:
                    slottype += self.slottype.__args__[0].__name__
                slottype += "]"
            elif inspect.isclass(self.slottype) and issubclass(self.slottype, Enum):
                options = ", ".join([repr(e.name) for e in self.slottype])
                slottype = "Enum[" + options + "]"
            else:
                slottype = self.slottype.__name__

            if value:
                if self.value is None:
                    val = "None"
                elif self.value == "":
                    val = '""'
                else:
                    val = self.value
                schema = f"{self.name}: {slottype} = {repr(val)}"
            else:
                schema = f"{self.name}: {slottype}"

            # logger.debug(f"Generated schema: {schema}")
            return schema
        except Exception as e:
            logger.error(f"Error generating schema for field {self.name}: {str(e)}")
            raise

    def schema_without_type(self, no_none: bool = False) -> Optional[str]:
        """Generate a schema representation without type information.

        Args:
            no_none: Whether to exclude None values

        Returns:
            The schema representation without type, or None if excluded
        """
        # logger.debug(f"Generating schema without type for field: {self.name}")
        try:
            if self.value is None:
                val = None
            elif self.value == "":
                val = '""'
            else:
                if isinstance(self.value, str):
                    val = f"{repr(self.value)}"
                else:
                    val = self.value

            if no_none and val == "None":
                # logger.debug(f"Skipping None value for field: {self.name}")
                return None

            schema = f"{self.name} = {repr(val)}"
            # logger.debug(f"Generated schema: {schema}")
            return schema
        except Exception as e:
            logger.error(
                f"Error generating schema without type for field {self.name}: {str(e)}"
            )
            return None

    @property
    def confirmed(self) -> bool:
        """Get the confirmation status of the field."""
        if hasattr(self, "_value") and isinstance(self._value, GenieValue):
            return self._value.confirmed
        return self._confirmed

    @confirmed.setter
    def confirmed(self, confirmed: bool):
        """Set the confirmation status of the field."""
        # logger.debug(
        #     f"Setting confirmation status to {confirmed} for field: {self.name}"
        # )
        self._confirmed = confirmed

    @property
    def value(self) -> Any:
        """Get the field value."""
        if isinstance(self._value, GenieValue):
            return self._value.value
        return self._value

    @value.setter
    def value(self, value: Any):
        """Set the field value."""
        # logger.debug(f"Setting value for field {self.name}: {value}")
        self.action_performed = False
        self._value = self.init_value(value)

    def init_value(self, value: Any) -> Optional[GenieValue]:
        """Initialize a field value with validation and wrapping.

        Args:
            value: The value to initialize

        Returns:
            The initialized value, or None if validation fails
        """
        logger.debug(f"Initializing value for field {self.name}: {value}")

        if value == "" or value is None:
            logger.debug(f"Empty or None value for field {self.name}")
            return None

        if self.slottype == "confirm":
            if not self._check_previous_confirm():
                logger.debug(
                    f"Previous confirmation check failed for field {self.name}"
                )
                return None

        valid = True
        # if self.validation:
        #     logger.debug(f"Validating value against criteria: {self.validation}")
        #     matches_criteria, reason = validation_check(
        #         self.name, value, self.validation
        #     )
        #     if not matches_criteria:
        #         if isinstance(value, GenieValue):
        #             value = value.value
        #         logger.warning(f"Validation failed for field {self.name}: {reason}")
        #         from worksheets.core.agent_acts import ReportAgentAct

        #         self.parent.bot.context.agent_acts.add(
        #             ReportAgentAct(
        #                 query=f"{self.name}={value}",
        #                 message=f"Invalid value for {self.name}: {value} - {reason}",
        #             )
        #         )
        #         valid = False

        if valid:
            if isinstance(value, GenieValue):
                logger.debug(f"Using existing GenieValue for field {self.name}")
                return value
            logger.debug(f"Creating new GenieValue for field {self.name}")
            return GenieValue(value)

        return None

    def _check_previous_confirm(self) -> bool:
        """Check if the previous action was a confirmation action."""
        logger.debug(f"Checking previous confirmation for field {self.name}")

        try:
            if self.bot.agent.dlg_history is not None and len(
                self.bot.agent.dlg_history
            ):
                if self.bot.agent.dlg_history[-1].system_action is not None:
                    for act in self.bot.agent.dlg_history[-1].system_action.actions:
                        from worksheets.core.agent_acts import AskAgentAct

                        if isinstance(act, AskAgentAct):
                            if act.field.slottype == "confirm":
                                logger.debug("Found previous confirmation action")
                                return True
            logger.debug("No previous confirmation action found")
            return False
        except Exception as e:
            logger.error(
                f"Error checking previous confirmation for field {self.name}: {str(e)}"
            )
            return False

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, GenieField):
            return self.name == other.name and self.value == other.value
        return False


"""Validation utilities for the Genie system.

This module provides functions for validating field values and handling
validation criteria using LLM-based validation.
"""


async def validation_check(
    name: str, value: Any, validation: str
) -> Tuple[bool, Optional[str]]:
    """Validate a value against specified criteria using LLM.

    This function uses a language model to validate field values against
    specified validation criteria.

    Args:
        name: The name of the field being validated
        value: The value to validate
        validation: The validation criteria

    Returns:
        Tuple containing:
            - bool: Whether the value is valid
            - str or None: The reason for invalidity, if any
    """
    logger.debug(f"Starting validation check for field {name}")
    logger.debug(f"Value to validate: {value}")
    logger.debug(f"Validation criteria: {validation}")

    prompt_path = "validation_check.prompt"

    if isinstance(value, GenieValue):
        val = str(value.value)
        logger.debug(f"Extracted value from GenieValue: {val}")
    else:
        val = str(value)

    try:
        logger.debug(f"Generating LLM response with prompt: {prompt_path}")
        llm_client = get_llm_client(
            model="azure/gpt-4.1-mini",
            temperature=0.0,
            max_tokens=1024,
        )
        prompt_template = load_fewshot_prompt_template(prompt_path)
        chain = prompt_template | llm_client
        
        logging_handler = LoggingHandler(
            prompt_file=prompt_path,
            metadata={
                "value": val,
                "criteria": validation,
                "name": name
            }
        )
        response = await chain.ainvoke(
            {
                "value": val,
                "criteria": validation,
                "name": name
            },
            config={"callbacks": [logging_handler]},
        )
        logger.debug(f"Received LLM response: {response}")

        bs = BeautifulSoup(response, "html.parser")
        reason = bs.find("reason")
        valid = bs.find("valid")

        if valid:
            is_valid = valid.text.strip().lower() == "true"
            reason_text = reason.text if reason else None
            log_validation_result(name, val, is_valid, reason_text)
            return is_valid, None

        logger.warning(f"Invalid response format from LLM for field {name}")
        log_validation_result(name, val, False, "Invalid response format from LLM")
        return False, reason.text if reason else None

    except Exception as e:
        logger.error(f"Error during validation check for field {name}: {str(e)}")
        log_validation_result(name, val, False, f"Validation error: {str(e)}")
        return False, f"Validation error: {str(e)}"
