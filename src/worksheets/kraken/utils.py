import asyncio
import warnings
from dataclasses import dataclass
from typing import Dict


from langchain_core.runnables import chain
from json_repair import repair_json
from loguru import logger

from worksheets.kraken.state import SqlQuery
from worksheets.llm.logging import LoggingHandler
from worksheets.llm.prompts import load_fewshot_prompt_template
from worksheets.llm.llm import get_llm_client


warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)  # from ReFinED


class BaseParser:
    @classmethod
    def initialize(engine: str):
        raise NotImplementedError("Subclasses should implement this method")

    @classmethod
    def run_batch(cls, questions: list[dict]):
        return asyncio.run(
            cls.runnable.with_config(
                {"recursion_limit": 60, "max_concurrency": 50}
            ).abatch(questions)
        )

    @classmethod
    async def arun_batch(cls, questions: list[dict]):
        return await cls.runnable.with_config(
            {"recursion_limit": 60, "max_concurrency": 50}
        ).abatch(questions)

    @classmethod
    async def arun(cls, question: dict):
        logger.info(f"Running question: {question}")
        return await cls.runnable.with_config(
            {"recursion_limit": 60, "max_concurrency": 50}
        ).ainvoke(question)


@chain
async def parse_string_to_json(output: str) -> dict:
    return repair_json(output, return_objects=True)


@chain
def extract_code_block_from_output(llm_output: str, code_block: str) -> str:
    code_block = code_block.lower()
    if f"```{code_block}" in llm_output.lower():
        start_idx = llm_output.lower().rfind(f"```{code_block}") + len(
            f"```{code_block}"
        )
        end_idx = llm_output.lower().rfind("```", start_idx)
        if end_idx < 0:
            # because llm_generation_chain does not include the stop token
            end_idx = len(llm_output)
        extracted_block = llm_output[start_idx:end_idx].strip()
        return extracted_block
    else:
        raise ValueError(f"Expected a code block, but llm output is {llm_output}")


@chain
def sql_string_to_sql_object(sql: str) -> SqlQuery:
    return SqlQuery(sql=sql)


@chain
def execute_sql_object(
    sql: SqlQuery,
    table_w_ids: Dict,
    database_name: str,
    embedding_server_address,
    db_host,
    db_port,
    db_username,
    db_password,
    source_file_mapping,
    suql_model_name,
    api_base=None,
    api_version=None,
    api_key=None,
) -> str:
    sql.execute(
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
    return sql


def format_table_schema(schema: str) -> str:
    # TODO: Used to format the table schemas from sql, right now we just return the schema as is.
    return schema


async def get_relevant_examples(utterance: str, examples: list[str]) -> list[str]:
    top_examples = await rerank_list.bind(top_k=3).ainvoke(
        # {
        #     "query_text": utterance,
        #     "item_list": examples,
        # }
        utterance,
        examples,
    )
    return top_examples


def get_relevant_table_schema(utterance: str, schema: str) -> dict:
    # TODO: Select relevant table schemas. Right now we are just returning the schema
    return schema


def process_reranking_output(response):
    new_response = ""
    for character in response:
        if not character.isdigit():
            new_response += " "
        else:
            new_response += character
    response = new_response.strip()
    response = [int(x) - 1 for x in response.split()]

    # deduplicate
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)

    return new_response


async def llm_rerank_window(query_text, retrieval_results):
    llm_client = get_llm_client(
        model="gpt-4o",
        temperature=1.0,
        top_p=0.9,
        max_tokens=700,
    )
    reranking_prompt_template = load_fewshot_prompt_template(
        "rerank_list.prompt"
    )
    logging_handler = LoggingHandler(
        prompt_file="rerank_list.prompt",
        metadata={
            "query_text": query_text,
            "retrieval_results": retrieval_results,
        }
    )
    reranking_prompt_chain = reranking_prompt_template | llm_client
    reranking_prompt_output = await reranking_prompt_chain.ainvoke(
        {
            "query_text": query_text,
            "retrieval_results": retrieval_results,
        },
        config={"callbacks": [logging_handler]},
    )

    reranked_indices = process_reranking_output(reranking_prompt_output)
    reranked_indices = [
        i for i in reranked_indices if i < len(retrieval_results)
    ]  # remove numbers that are outside the range
    logger.debug("reranked_indices = %s", str(reranked_indices))
    return reranked_indices, reranking_prompt_output


@chain
async def rerank_list(query_text, item_list, top_k=3):
    llm_reranker_sliding_window_size = 20
    llm_reranker_sliding_window_step = 10

    end_index = len(item_list)
    start_index = end_index - llm_reranker_sliding_window_size
    original_rank = list(range(len(item_list)))
    while True:
        if start_index < 0:
            start_index = 0
        reranked_indices, reranking_prompt_output = await llm_rerank_window(
            query_text, item_list[start_index:end_index]
        )

        if len(reranked_indices) != (end_index - start_index):
            missing_indices = set(range(end_index - start_index))
            for found_index in reranked_indices:
                if found_index in missing_indices:
                    missing_indices.remove(found_index)

            logger.warning(
                "LLM reranking should return the same number of outputs as inputs. Adding missing indices: %s. Prompt output was %s",
                str(missing_indices),
                reranking_prompt_output,
            )

            # TODO instead of adding missing indices, shift everything and continue
            # Add missing indices to the end so that we don't crash
            # This is reasonable assuming that if the LLM did not output an index, it probably was not that relevant to the query to begin with
            reranked_indices = reranked_indices + list(missing_indices)

            assert len(reranked_indices) == (end_index - start_index)

        item_list[start_index:end_index] = [
            item_list[start_index + reranked_indices[i]]
            for i in range(len(reranked_indices))
        ]
        original_rank[start_index:end_index] = [
            original_rank[start_index + reranked_indices[i]]
            for i in range(len(reranked_indices))
        ]

        if start_index == 0:
            break
        end_index -= llm_reranker_sliding_window_step
        start_index -= llm_reranker_sliding_window_step

    return item_list[:top_k]


@dataclass
class DialogueTurn:
    user_utterance: str
    agent_utterance: str
    user_target: str
    db_results: list
