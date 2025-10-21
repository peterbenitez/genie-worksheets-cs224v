import json

from langgraph.graph import END, StateGraph
from loguru import logger
from langchain_core.runnables import chain
from worksheets.kraken.state import ParserAction, KrakenState, SqlQuery
from pydantic import BaseModel, Field
from worksheets.llm.prompts import load_fewshot_prompt_template
from worksheets.kraken.utils import (
    BaseParser,
    execute_sql_object,
    format_table_schema,
    get_relevant_examples,
    get_relevant_table_schema,
    sql_string_to_sql_object,
)
from worksheets.llm.llm import get_llm_client

class Action(BaseModel):
    thought: str = Field(description="The thought of the action")
    action_name: str = Field(description="The name of the action")
    action_argument: str = Field(description="The argument of the action")

@chain
async def json_to_string(j: dict) -> str:
    return json.dumps(j, indent=2, ensure_ascii=False)


@chain
async def action_to_dict(action: Action) -> dict:
    """Convert Pydantic Action to dict for downstream processing."""
    return action.model_dump()


@chain
async def json_to_action(action_dict: dict) -> ParserAction:
    thought = action_dict["thought"]
    action_name = action_dict["action_name"]
    action_argument = action_dict["action_argument"]

    if action_name == "execute_sql":
        assert action_argument, action_dict

    return ParserAction(
        thought=thought,
        action_name=action_name,
        action_argument=action_argument,
    )


class KrakenParser(BaseParser):
    @classmethod
    def initialize(
        cls,
        engine: str,
        table_w_ids: dict,
        database_name: str,
        suql_model_name: str,
        embedding_server_address: str = "http://127.0.0.1:8509",
        db_host: str = "127.0.0.1",
        db_port: str = "5432",
        db_username: str = "select_user",
        db_password: str = "select_user",
        source_file_mapping: dict = {},
        domain_instructions: str | None = None,
        examples: list | None = None,
        table_schema: list | None = None,
        suql_api_base: str = None,
        suql_api_version: str = None,
        suql_api_key: str = None,
    ):
        @chain
        async def initialize_state(_input):
            return KrakenState(
                question=_input["question"],
                conversation_history=_input["conversation_history"],
                engine=engine,
                action_counter=0,
                actions=[],
                examples=examples or [],
                table_schemas=table_schema or [],
                domain_instructions=domain_instructions,
                api_base=suql_api_base,
                api_version=suql_api_version,
                api_key=suql_api_key,
                generated_sqls=[],
            )

        # build the graph
        graph = StateGraph(KrakenState)
        graph.add_node("start", lambda x: {})
        graph.add_node("controller", KrakenParser.controller)
        graph.add_node("execute_sql", KrakenParser.execute_sql)
        graph.add_node("get_tables_schema", KrakenParser.get_tables_schema)
        # graph.add_node("get_examples", PartToWholeParser.get_examples)

        graph.add_node("stop", KrakenParser.stop)

        graph.set_entry_point("start")

        graph.add_edge("start", "controller")
        graph.add_conditional_edges(
            "controller",
            KrakenParser.router,  # the function that will determine which node is called next.
        )
        for n in [
            "execute_sql",
            "get_tables_schema",
            # "get_examples",
        ]:
            graph.add_edge(n, "controller")

        graph.add_edge("stop", END)

        cls.llm_client = get_llm_client(
            model=engine,
            temperature=1.0,
            top_p=0.9,
            max_tokens=700,
        )

        cls.controller_prompt_template = load_fewshot_prompt_template(
            "controller.prompt"
        )
        cls.controller_chain = cls.controller_prompt_template | cls.llm_client.with_structured_output(Action) | action_to_dict | json_to_action

        cls.sql_chain = sql_string_to_sql_object | execute_sql_object.bind(
            table_w_ids=table_w_ids,
            database_name=database_name,
            suql_model_name=suql_model_name,
            embedding_server_address=embedding_server_address,
            source_file_mapping=source_file_mapping,
            api_base=suql_api_base,
            api_version=suql_api_version,
            api_key=suql_api_key,
            db_host=db_host,
            db_port=db_port,
            db_username=db_username,
            db_password=db_password,
        )

        compiled_graph = graph.compile()
        cls.runnable = initialize_state | compiled_graph
        logger.info("Finished initializing the graph.")
        # compiled_graph.get_graph().print_ascii()  # requies grandalf
        # sys.stdout.flush()

    @staticmethod
    def get_current_action(state):
        return state["actions"][-1]

    @staticmethod
    async def router(state):
        move_back_on_duplicate_action = 2
        current_action = KrakenParser.get_current_action(state)
        if current_action in state["actions"][-5:-1]:
            logger.warning(
                "Took duplicate action %s, going back %d steps.",
                current_action.action_name,
                move_back_on_duplicate_action,
            )
            # current_action.observation = "I have already taken this action. I should not repeat the same action twice."
            # Remove generated_sparqls as well
            if len(state["actions"]) - 2 >= 0:
                for i in range(len(state["actions"]) - 2, -1, -1):
                    if state["actions"][i].action_name == "execute_sql":
                        state["generated_sqls"] = state["generated_sqls"][:-1]
            state["actions"] = state["actions"][:-move_back_on_duplicate_action]
            state["action_counter"] -= move_back_on_duplicate_action
            return "controller"

        if state["action_counter"] >= 15:
            if (
                len(state["generated_sqls"]) == 0
                or not state["generated_sqls"][-1].has_results()
            ):
                logger.warning(
                    "Reached action_counter limit without a good Sql. Starting over."
                )
                state["generated_sqls"] = []
                state["actions"] = []
                state["action_counter"] = 0

            return END

        return state["actions"][-1].action_name

    @staticmethod
    @chain
    async def controller(state):
        # if state["actions"]:
        #     print("last action = ", state["actions"][-1])
        #     sys.stdout.flush()

        # make the history shorter
        actions = state["actions"][-10:]
        action_history = []
        for i, a in enumerate(actions):
            include_observation = True
            if i < len(actions) - 2 and a.action_name in [
                "get_tables_schema",
                "get_examples",
                "get_feedback_on_result",
            ]:
                include_observation = False
            action_history.append(a.to_jinja_string(include_observation))

        action = await KrakenParser.controller_chain.ainvoke(
            {
                "question": state["question"],
                "action_history": action_history,
                "conversation_history": state["conversation_history"],
                "instructions": state["domain_instructions"],
            }
        )
        logger.debug(f"Generated action: {action}")
        return {
            "actions": state["actions"] + [action],
            "action_counter": state["action_counter"] + 1,
        }

    @staticmethod
    @chain
    async def execute_sql(state):
        current_action = KrakenParser.get_current_action(state)
        assert current_action.action_name == "execute_sql"
        logger.debug(f"executing {current_action.action_argument}")
        sql = await KrakenParser.sql_chain.ainvoke(current_action.action_argument)
        current_action.action_argument = (
            sql.sql
        )  # update it with the cleaned and optimized SQL
        # print(sql.has_results())
        # print(sql.execution_result)
        if sql.has_results():
            # current_action.observation = sql.results_in_table_format()
            current_action.observation = sql.execution_result
            # print("SUQL execution result:", current_action.observation)
        else:
            current_action.observation = sql.execution_status

        return {"generated_sqls": state["generated_sqls"] + [sql]}

    @staticmethod
    @chain
    async def get_tables_schema(state):
        current_action = KrakenParser.get_current_action(state)
        assert current_action.action_name == "get_tables_schema"
        action_results = get_relevant_table_schema(
            current_action.action_argument, state["table_schemas"]
        )
        current_action.observation = format_table_schema(action_results)

    @staticmethod
    @chain
    async def get_examples(state):
        current_action = KrakenParser.get_current_action(state)
        assert current_action.action_name == "get_examples"
        action_result = await get_relevant_examples(
            current_action.action_argument, state["examples"]
        )
        current_action.observation = "\n".join(action_result)

    @staticmethod
    @chain
    async def stop(state):
        current_action = KrakenParser.get_current_action(state)
        assert current_action.action_name == "stop"
        # print("generated_sqls = ", state["generated_sqls"])
        for s in state["generated_sqls"]:
            assert isinstance(s, SqlQuery)
        if (
            len(state["generated_sqls"]) == 0
            or not state["generated_sqls"][-1].has_results()
        ):
            logger.warning("Stop() was called without a good Sql. Starting over.")
            state["generated_sqls"] = []
            state["actions"] = []
            state["action_counter"] = 0
            return

        logger.info("Finished run for question %s", state["question"])
        final_sql = state["generated_sqls"][-1]
        if final_sql.sql.strip().endswith("LIMIT 10"):
            logger.info("Removing LIMIT 10 from the final Sql")
            s = final_sql.sql.strip()[: -len("LIMIT 10")]
            final_sql = SqlQuery(sql=s)
            final_sql.execute()

        return {"final_sql": final_sql}
