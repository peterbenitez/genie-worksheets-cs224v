"""
Data Availability Checking System

This module distinguishes between "missing tool" and "missing data" issues.
Before triggering tool discovery, we check if the query failure is due to:
- Missing columns (querying fields that don't exist)
- Missing tables (querying tables that don't exist)

This prevents generating "missing tool" specs when we have the capability
but lack the data.

Usage:
    from data_availability import DataAvailabilityChecker

    # In no_response() function:
    checker = DataAvailabilityChecker(registry, runtime)
    result = checker.check_availability(user_utterance, query_context)

    if result.issue_type != "no_issue":
        # Handle data constraint issue
        return result.explanation
    else:
        # Proceed with tool discovery
        ...

Schema:
    Schema is parsed from table_schema.txt (single source of truth).
    No separate config file needed.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

from loguru import logger
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

try:
    from .schema_parser import parse_table_schema, get_table_keywords
except ImportError:
    from schema_parser import parse_table_schema, get_table_keywords


# Load environment variables
ENV_PATH = Path(__file__).parent.parent.parent.parent.parent / ".env"
load_dotenv(ENV_PATH)


@dataclass
class DataAvailabilityResult:
    """Result of data availability checking"""
    issue_type: str  # "no_issue", "temporal_constraint", "missing_column", "missing_table", "insufficient_data"
    has_capability: bool  # Can we theoretically do this with existing tools?
    explanation: str  # User-facing explanation
    technical_details: Dict = field(default_factory=dict)  # For logging/debugging
    suggested_workaround: Optional[str] = None  # Alternative approach if available


class DataAvailabilityChecker:
    """
    Checks if query failures are due to data constraints rather than missing tools.

    This class implements programmatic checking of hard constraints (schema)
    before invoking LLM analysis.

    Schema is loaded from table_schema.txt (single source of truth).
    """

    def __init__(self, registry: Any, runtime: Any = None, schema_path: Path = None):
        """
        Initialize the data availability checker.

        Args:
            registry: ToolRegistry instance with existing tools/tables info
            runtime: GenieRuntime instance (optional, for live DB queries)
            schema_path: Path to table_schema.txt (optional, auto-detected)
        """
        self.registry = registry
        self.runtime = runtime

        if schema_path is None:
            schema_path = Path(__file__).parent.parent / "table_schema.txt"

        self.schema = parse_table_schema(schema_path)
        self.known_tables = list(self.schema.keys())

        # Initialize Azure OpenAI for semantic analysis of queries
        self.client = AzureOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            api_version=os.getenv("LLM_API_VERSION", "2024-10-21"),
            azure_endpoint=os.getenv("LLM_API_ENDPOINT"),
        )

    def check_availability(self, user_utterance: str, query_context: Dict = None) -> DataAvailabilityResult:
        """
        Main entry point: Check if query failure is due to data constraints.

        Args:
            user_utterance: The user's question that triggered no_response()
            query_context: Optional context (attempted SQL, API calls, etc.)

        Returns:
            DataAvailabilityResult indicating the type of issue
        """
        if query_context is None:
            query_context = {}

        logger.debug(f"Checking data availability for: {user_utterance}")

        # Check missing columns (deterministic)
        column_result = self._check_missing_columns(user_utterance, query_context)
        if column_result:
            logger.info(f"Missing column detected: {column_result.issue_type}")
            return column_result

        # Check missing tables (deterministic)
        table_result = self._check_missing_tables(user_utterance, query_context)
        if table_result:
            logger.info(f"Missing table detected: {table_result.issue_type}")
            return table_result

        # No data constraint detected - might be a genuine tool gap
        logger.debug("No data constraint detected")
        return DataAvailabilityResult(
            issue_type="no_issue",
            has_capability=False,
            explanation="",
            technical_details={"checked": ["columns", "tables"]}
        )

    def _check_missing_columns(self, utterance: str, context: Dict) -> Optional[DataAvailabilityResult]:
        """
        Check if query needs a column that doesn't exist in the target table.
        """
        # Identify target table
        target_table = self._identify_target_table(utterance)

        if target_table not in self.schema:
            return None  # Unknown table, can't check schema

        available_columns = self.schema[target_table]["columns"]

        # Use LLM to identify what column/field is being requested
        requested_column = self._identify_requested_column(utterance, target_table, available_columns)

        if not requested_column or requested_column in available_columns:
            return None  # No specific column requested, or column exists

        similar_columns = self._find_similar_columns(requested_column, available_columns)

        explanation = (
            f"I can query {target_table} data, but the field '{requested_column}' "
            f"is not available in our database."
        )

        if similar_columns:
            explanation += f" Available similar fields: {', '.join(similar_columns)}."

        return DataAvailabilityResult(
            issue_type="missing_column",
            has_capability=True,  # We CAN query the table, just not this specific field
            explanation=explanation,
            technical_details={
                "table": target_table,
                "requested_column": requested_column,
                "available_columns": available_columns,
                "similar_columns": similar_columns
            },
            suggested_workaround=f"Try asking about: {', '.join(similar_columns[:3])}" if similar_columns else None
        )

    def _check_missing_tables(self, utterance: str, context: Dict) -> Optional[DataAvailabilityResult]:
        """
        Check if query needs a table/data source that doesn't exist.
        """
        # Use LLM to identify what table/data source is being requested
        requested_table = self._identify_requested_table(utterance)

        if requested_table and requested_table not in self.known_tables:
            explanation = (
                f"I don't have access to {requested_table} data. "
                f"Available data sources: {', '.join(self.known_tables)}."
            )

            return DataAvailabilityResult(
                issue_type="missing_table",
                has_capability=False,  # Can't query table that doesn't exist
                explanation=explanation,
                technical_details={
                    "requested_table": requested_table,
                    "available_tables": self.known_tables
                },
                suggested_workaround=f"Try asking about: {', '.join(self.known_tables)}" if self.known_tables else None
            )

        return None  # Table exists or couldn't identify requested table

    def _identify_target_table(self, utterance: str) -> Optional[str]:
        """
        Identify target table using keywords derived from table names.
        Keywords are auto-generated from table names.
        """
        lowercase = utterance.lower()

        for table_name in self.schema.keys():
            keywords = get_table_keywords(table_name)
            if any(kw in lowercase for kw in keywords):
                return table_name

        return list(self.schema.keys())[0] if self.schema else None

    def _identify_requested_column(self, utterance: str, table: str, available_columns: List[str]) -> Optional[str]:
        """
        Use LLM to identify what column/field is being requested.
        """
        try:
            prompt = f"""Query: "{utterance}"
Table: {table}
Columns: {', '.join(available_columns)}

Return ONLY the column name that best matches the query, or NONE if no specific column is requested."""

            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50
            )

            result = response.choices[0].message.content.strip().strip('"\'').split()[0]
            return None if result == "NONE" else result

        except Exception as e:
            logger.warning(f"Error identifying requested column: {e}")
            return None

    def _identify_requested_table(self, utterance: str) -> Optional[str]:
        """
        Use LLM to identify what table/data source is being requested.
        """
        try:
            table_options = ", ".join(self.known_tables)

            prompt = f"""Query: "{utterance}"
Available tables:
{self._format_table_descriptions()}

Return ONLY the table name, or OTHER if not in any table."""

            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50
            )

            result = response.choices[0].message.content.strip().lower()

            if result in self.known_tables:
                return result
            elif result != "other":
                return result  # Unknown data source

            return None

        except Exception as e:
            logger.warning(f"Error identifying requested table: {e}")
            return None

    def _find_similar_columns(self, target: str, available: List[str], threshold: int = 3) -> List[str]:
        """
        Find columns similar to the target using simple string matching.

        Args:
            target: The column name being searched for
            available: List of available column names
            threshold: Maximum number of similar columns to return

        Returns:
            List of similar column names
        """
        target_lower = target.lower()
        similar = []

        for column_name in available:
            column_lowercase = column_name.lower()

            # Check if target is substring of column or vice versa
            if target_lower in column_lowercase or column_lowercase in target_lower:
                similar.append(column_name)
            # Check if they share common words
            elif any(word in column_lowercase for word in target_lower.split('_')):
                similar.append(column_name)

        return similar[:threshold]

    def _format_table_descriptions(self) -> str:
        """Format table descriptions for the prompt."""
        descriptions = []
        for table_name, table_info in self.schema.items():
            columns = table_info.get("columns", [])
            sample_cols = ", ".join(columns[:5])
            if len(columns) > 5:
                sample_cols += f", ... ({len(columns)} total)"
            descriptions.append(f"- {table_name}: {sample_cols}")
        return "\n".join(descriptions) if descriptions else "No tables available"


# Utility function for testing
def test_data_availability_checker():
    """Test the data availability checker with sample queries."""

    class MockRegistry:
        pass

    checker = DataAvailabilityChecker(MockRegistry())

    print(f"Loaded schema with tables: {checker.known_tables}")
    for table, info in checker.schema.items():
        print(f"  {table}: {len(info['columns'])} columns")

    # Test with a generic query
    result = checker.check_availability("What data do you have?")
    print(f"\nTest result: {result.issue_type}")
    print(f"  Explanation: {result.explanation or 'No issue detected'}\n")


if __name__ == "__main__":
    test_data_availability_checker()
