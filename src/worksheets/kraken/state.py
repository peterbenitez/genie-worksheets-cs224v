import json
import re
from typing import Optional, TypedDict

import pandas as pd

from worksheets.kraken.sql_utils import execute_sql


def convert_sql_result_to_dict(results, column_names):
    data = []
    for row in results:
        row_data = {}
        for col_index, col_value in enumerate(row):
            row_data[column_names[col_index]] = col_value
        data.append(row_data)

    return data


def convert_json_to_table_format(data):
    # Load the JSON data
    if isinstance(data, str):
        data = json.loads(data)

    # Convert the JSON data to a Pandas DataFrame
    df = pd.DataFrame(data)

    # Convert the DataFrame to a table format
    table = df.to_markdown(index=False)

    return table


class SqlQuery:
    def __init__(
        self,
        sql: Optional[str] = None,
        table_w_ids: dict = None,
        database_name: str = None,
        embedding_server_address: str = "http://127.0.0.1:8509",
        source_file_mapping: dict = {},
    ):
        if "SELECT" not in sql:
            self.is_valid = False
        self.sql = SqlQuery.clean_sql(sql)
        self.table_w_ids = table_w_ids
        self.database_name = database_name
        self.embedding_server_address = embedding_server_address
        self.source_file_mapping = source_file_mapping
        self.is_valid = True

        self.execution_result = None
        self.execution_status = None

    @staticmethod
    def clean_sql(sql: str):
        if sql is None:
            return sql
        cleaned_sql = sql.strip()

        cleaned_sql = re.sub(r"#.*", "", cleaned_sql).strip()  # Remove comments
        # cleaned_sql = try_to_optimize_query(cleaned_sql)
        # cleaned_sql = re.sub(
        #     r"\s+", " ", cleaned_sql
        # )  # Remove line breaks and other extra whitespaces

        # Might want to prettify the sql
        # cleaned_sql = prettify_sql(cleaned_sql)

        return cleaned_sql

    def execute(
        self,
        table_w_ids,
        database_name,
        suql_model_name,
        embedding_server_address,
        source_file_mapping,
        api_base=None,
        api_version=None,
        api_key=None,
        db_host=None,
        db_port=None,
        db_username=None,
        db_password=None,
    ):
        # TODO: probably need to perform some post processing by using column_names with self.execution_result
        execution_result, column_names, self.execution_status = execute_sql(
            self.sql,
            table_w_ids,
            database_name,
            suql_model_name,
            embedding_server_address,
            source_file_mapping,
            api_base=api_base,
            api_version=api_version,
            api_key=api_key,
            db_host=db_host,
            db_port=db_port,
            db_username=db_username,
            db_password=db_password,
        )

        if execution_result is not None:
            self.execution_result = convert_sql_result_to_dict(
                execution_result, column_names
            )

    def has_results(self) -> bool:
        return self.execution_result is not None

    def results_in_table_format(self):
        return convert_json_to_table_format(self.execution_result)

    def __repr__(self):
        return f"Sql({self.sql})"

    def __hash__(self):
        return hash(self.sql)


def merge_dictionaries(dictionary_1: dict, dictionary_2: dict) -> dict:
    """
    Merges two dictionaries, combining their key-value pairs.
    If a key exists in both dictionaries, the value from dictionary_2 will overwrite the value from dictionary_1.

    Parameters:
        dictionary_1 (dict): The first dictionary.
        dictionary_2 (dict): The second dictionary.

    Returns:
        dict: A new dictionary containing the merged key-value pairs.
    """
    merged_dict = dictionary_1.copy()  # Start with a copy of the first dictionary
    merged_dict.update(
        dictionary_2
    )  # Update with the second dictionary, overwriting any duplicates
    return merged_dict


def merge_sets(set_1: set, set_2: set) -> set:
    return set_1 | set_2


def add_item_to_list(_list: list, item) -> list:
    ret = _list.copy()
    # if item not in ret:
    ret.append(item)
    return ret


class ParserAction:
    possible_actions = [
        "get_tables_schema",
        "execute_sql",
        "get_examples",
        "get_feedback_on_result",
        "stop",
    ]

    # All actions have a single input parameter for now
    def __init__(self, thought: str, action_name: str, action_argument: str):
        self.thought = thought
        self.action_name = action_name
        self.action_argument = action_argument
        self.observation = None

        assert self.action_name in ParserAction.possible_actions

    def to_jinja_string(self, include_observation: bool) -> str:
        if not self.observation:
            observation = "Did not find any results."
        else:
            observation = self.observation
        ret = f"Thought: {self.thought}\nAction: {self.action_name}({self.action_argument})\n"
        if include_observation:
            ret += f"Observation: {observation}\n"
        return ret

    def __repr__(self) -> str:
        if not self.observation:
            observation = "Did not find any results."
        else:
            observation = self.observation
        return f"Thought: {self.thought}\nAction: {self.action_name}({self.action_argument})\nObservation: {observation}"

    def __eq__(self, other):
        if not isinstance(other, ParserAction):
            return NotImplemented
        return (
            self.action_name == other.action_name
            and self.action_argument == other.action_argument
        )

    def __hash__(self):
        return hash((self.action_name, self.action_argument))


class KrakenState(TypedDict):
    question: str
    engine: str
    generated_sqls: list[SqlQuery]
    final_sql: SqlQuery
    action_counter: int
    examples: list[str]
    table_schemas: list[dict[str, str]]
    conversation_history: list
    domain_instructions: str
    api_base: str = None
    api_version: str = None
    api_key: str = None
    actions: list[ParserAction]
