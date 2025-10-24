from __future__ import annotations

import inspect
import re
from functools import partial
from typing import Any

from loguru import logger
from uvicorn import Config

from worksheets.core.builtin_functions import (
    answer_clarification_question,
    confirm,
    no_response,
    chitchat,
    propose,
    say,
    state_response,
)
from worksheets.core.context import GenieContext
from worksheets.core.fields import GenieValue
from worksheets.core.worksheet import Answer, GenieWorksheet, MoreFieldInfo
from worksheets.utils.code_execution import replace_undefined_variables
from worksheets.utils.field import get_genie_fields_from_ws
from worksheets.utils.predicates import eval_predicates
from worksheets.utils.rumtime import callable_name
from worksheets.utils.worksheet import collect_all_parents


class GenieRuntime:
    """Main runtime environment for Genie system.

    This class manages the execution environment, including worksheet registration,
    context management, and action execution.

    Attributes:
        name (str): Runtime instance name.
        prompt_dir (str): Directory for prompts.
        config: Additional arguments.
        genie_worksheets (list): Registered worksheets.
        genie_db_models (list): Registered database models.
        starting_prompt (str): Initial system prompt.
        description (str): Runtime description.
        suql_runner: SQL query runner.
        suql_parser: SQL query parser.
        context (GenieContext): Global context.
        dlg_history (list): Dialogue history.
    """

    def __init__(
        self,
        # Any additional arguments
        config: "Config",
        # Define the API to be used
        api=None,
        # The SUQL runner (SUQLKnowledgeBase)
        suql_runner=None,
        # The agent
        agent=None,
    ):
        self.config = config
        self.genie_worksheets = []
        self.genie_db_models = []
        self.suql_runner = suql_runner
        self.agent = agent
        self._interpreter = GenieInterpreter()
        self.context = GenieContext()
        self.local_context_init = GenieContext()

        # add the api to the context
        if api:
            if isinstance(api, list):
                apis = api
            else:
                api_funcs = inspect.getmembers(api, inspect.isfunction)
                apis = [func for name, func in api_funcs if not name.startswith("_")]
        else:
            apis = []

        self.order_of_actions = []

        apis.extend([self.suql_runner])

        Answer.bot = self

        # Set global runtime reference for builtin functions
        import worksheets.core.builtin_functions as builtin_functions
        builtin_functions._current_runtime = self

        # Add the predefined apis and functions
        apis.extend(
            [
                propose,
                confirm,
                GenieValue,
                partial(answer_clarification_question, context=self.context),
                Answer,
                MoreFieldInfo,
                say,
                no_response,
                state_response,
                chitchat,
            ]
        )
        for api in apis:
            self.add_api(api)

    def reset(self):
        """Reset the bot's context and state."""
        self.context.reset_agent_acts()
        to_delete = []
        for key, value in self.context.context.items():
            if isinstance(value, GenieWorksheet):
                to_delete.append(key)

        for key in to_delete:
            del self.context.context[key]
        self.dlg_history = None
        self.order_of_actions = []

    def add_worksheet(self, ws: type):
        """Add a worksheet to the bot's context.

        Args:
            ws (type): The worksheet class to add.
        """
        ws.bot = self
        for field in get_genie_fields_from_ws(ws):
            field.parent = ws
            field.bot = self
        self.genie_worksheets.append(ws)
        self.context.set(ws.__name__, ws)
        # self.context.update(self._grab_all_variables(ws))
        # self.local_context_init.update(self._grab_all_variables(ws))

    def add_db_model(self, db_model: type):
        """Add a database model to the bot's context.

        Args:
            db_model (type): The database model class to add.
        """
        db_model.bot = self
        for field in get_genie_fields_from_ws(db_model):
            field.parent = db_model
            field.bot = self
        self.genie_db_models.append(db_model)
        self.context.set(db_model.__name__, db_model)
        # self.context.update(self._grab_all_variables(db_model))
        # self.local_context_init.update(self._grab_all_variables(db_model))

    def add_api(self, api: Any):
        """Add an API function to the context.

        Args:
            api (Any): The API function or object to add.
        """
        self.context.set(callable_name(api), api)

    def geniews(
        self,
        predicates=None,
        outputs: GenieWorksheet | dict | None = None,
        backend_api=None,
        actions="",
    ):
        """Decorator to define a Genie worksheet."""

        def decorator(cls):
            cls.predicate = predicates
            cls.outputs = outputs
            cls.backend_api = backend_api
            cls.actions = actions
            self.add_worksheet(cls)
            return cls

        return decorator

    def genie_sql(
        self,
        outputs: GenieWorksheet | dict | None = None,
        actions="",
    ):
        """Decorator to define a Genie database model."""

        def decorator(cls):
            if outputs is None:
                d_outputs = {}
            else:
                d_outputs = outputs
            cls.outputs = d_outputs
            cls.actions = actions
            self.add_db_model(cls)
            return cls

        return decorator

    def execute(
        self, code: str, local_context: GenieContext | None = None, sp: bool = False
    ):
        """Execute the given code in the context of the bot.

        Args:
            code (str): The code to execute.
            local_context (GenieContext | None, optional): Local context to use. Defaults to None.
            sp (bool, optional): Whether this is a semantic parser execution. Defaults to False.
        """
        if local_context:
            local_context.update(
                {k: v for k, v in self.local_context_init.context.items()}
            )
        else:
            local_context = GenieContext(
                {k: v for k, v in self.local_context_init.context.items()}
            )
        self._interpreter.execute(
            code,
            self.context,
            local_context,
            sp=sp,
        )

        # Add the parents for all the objects in the local context
        collect_all_parents(local_context)

    def eval(self, code: str, local_context: GenieContext | None = None) -> Any:
        """Evaluate the given code in the context of the bot.

        Args:
            code (str): The code to evaluate.
            local_context (GenieContext | None, optional): Local context to use. Defaults to None.

        Returns:
            Any: The result of the evaluation.
        """
        if local_context:
            local_context.update(
                {k: v for k, v in self.local_context_init.context.items()}
            )
        else:
            local_context = GenieContext(
                {k: v for k, v in self.local_context_init.context.items()}
            )
        return self._interpreter.eval(
            code,
            self.context,
            local_context,
        )

    def update_from_context(self, context):
        """add new variables to the context"""
        self.context.update(context.context)

    def get_available_worksheets(self, context):
        """Get all available worksheets based on their predicates."""
        for ws in self.genie_worksheets:
            if ws.predicate:
                if eval_predicates(ws.predicate, None, self, context):
                    yield ws
            else:
                yield ws

    def get_available_dbs(self, context):
        """Get all available database models based on their predicates."""
        for db in self.genie_db_models:
            if db.predicate:
                if eval_predicates(db.predicate, None, self, context):
                    yield db
            else:
                yield db

    def get_all_variables(self):
        """Get all fields (variables) from all worksheets."""
        all_variables = []
        for ws in self.genie_worksheets:
            for field in get_genie_fields_from_ws(ws):
                all_variables.append(field.name)

        return all_variables


class GenieInterpreter:
    """Interpreter for executing Genie code.

    This class provides code execution capabilities within the Genie environment,
    handling variable resolution and context management.
    """

    def execute(self, code, global_context, local_context, sp=False):
        # There are some issues here. since there are no numbers now,
        # when we do courses_to_take = CoursesToTake(courses_0_details=course)
        # since courses_to_take is a field in main worksheet, the code gets modified to:
        # main.courses_to_take.value = CoursesToTake(courses_0_details=course)
        # One easy fix could be if you are setting a GenieWorksheet to a field, then
        # do not modify the code.

        # Another way is to have an, argument which mentions if the execution is from semantic parser
        # if it is, then do not modify the code.

        if not sp:
            # If the execution is for action then we replace the undefined variables
            code = replace_undefined_variables(code, local_context, global_context)
        try:
            try:
                exec(code, global_context.context, local_context.context)
            except NameError as e:
                local_context.set(e.name, None)
                # regex to catch the variable name. If the variable name is "user_task" then we want to find "user_task.value" as well until we hit a space.
                # This is just a hack ideally we should traverse the ast or at least use the tokenize module to find the variable name
                var_name = re.findall(rf"{e.name}\.\w+", code)
                if var_name:
                    code = code.replace(var_name[0], f"{e.name}")
                exec(code, global_context.context, local_context.context)
                local_context.delete(e.name)
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error(f"Code: {code}")

            local_context.context["__result"] = e

    def eval(self, code, global_context, local_context):
        # perform rewrite to update any variables that is not in the local context
        # by using the variable resolver
        code = replace_undefined_variables(code, local_context, global_context).strip()
        try:
            return eval(code, global_context.context, local_context.context)
        except (NameError, AttributeError):
            return False
