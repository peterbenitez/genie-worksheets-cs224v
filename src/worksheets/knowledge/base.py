import os
from abc import ABC, abstractmethod
from typing import Callable, Optional

from suql import suql_execute

from worksheets.agent.config import AzureModelConfig, OpenAIModelConfig

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


class BaseKnowledgeBase(ABC):
    """Base class for knowledge bases"""

    @abstractmethod
    def run(self, query, *args, **kwargs):
        """Run the knowledge base query and return the result."""
        pass


class SUQLKnowledgeBase(BaseKnowledgeBase):
    """Knowledge base for SUQL queries"""

    def __init__(
        self,
        model_config: AzureModelConfig | OpenAIModelConfig,
        tables_with_primary_keys: Optional[dict] = None,
        database_name: Optional[str] = None,
        embedding_server_address: Optional[str] = None,
        source_file_mapping: Optional[dict] = None,
        postprocessing_fn: Optional[Callable] = None,
        result_postprocessing_fn: Optional[Callable] = None,
        max_rows: int = 3,
        db_username: Optional[str] = None,
        db_password: Optional[str] = None,
        db_host: str = "127.0.0.1",
        db_port: str = "5432",
    ):
        self.model_config = model_config
        self.tables_with_primary_keys = tables_with_primary_keys
        self.database_name = database_name
        self.embedding_server_address = embedding_server_address
        self.source_file_mapping = source_file_mapping
        self.postprocessing_fn = postprocessing_fn
        self.result_postprocessing_fn = result_postprocessing_fn
        self.max_rows = max_rows

        self.db_username = db_username
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port

    def run(self, query, *args, **kwargs):
        """Run the SUQL query and return the result."""

        if self.postprocessing_fn:
            query = self.postprocessing_fn(query)

        query = query.strip().replace("\\'", "'")

        if self.model_config.config_name == "azure":
            api_base = os.getenv("LLM_API_ENDPOINT")
            api_version = os.getenv("LLM_API_VERSION")
        else:
            api_base = os.getenv("LLM_API_BASE_URL")
            api_version = None

        results, column_names, _ = suql_execute(
            query,
            table_w_ids=self.tables_with_primary_keys,
            database=self.database_name,
            llm_model_name=self.model_config.model_name,
            embedding_server_address=self.embedding_server_address,
            source_file_mapping=self.source_file_mapping,
            select_username=self.db_username,
            select_userpswd=self.db_password,
            host=self.db_host,
            port=self.db_port,
            api_base=api_base,
            api_version=api_version,
            api_key=os.getenv("LLM_API_KEY"),
        )

        # Convert the results to a list of dictionaries for genie worksheets
        results = [dict(zip(column_names, result)) for result in results]

        if self.result_postprocessing_fn:
            results = self.result_postprocessing_fn(results, column_names)

        return results[: self.max_rows]
