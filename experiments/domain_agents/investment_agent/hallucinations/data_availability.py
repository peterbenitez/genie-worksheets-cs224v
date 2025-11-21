"""
Data Availability Checking System

This module distinguishes between "missing tool" and "missing data" issues.
Before triggering tool discovery, we check if the query failure is due to:
- Temporal constraints (data outside available date range)
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
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from loguru import logger
from openai import AzureOpenAI
import os
from dotenv import load_dotenv


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

    This class implements Phase 1 of the hybrid approach: programmatic checking
    of hard constraints (schema, temporal ranges) before invoking expensive LLM analysis.
    """

    # Known schema for our tables
    SCHEMA = {
        "fidelity_funds": {
            "columns": [
                "id", "symbol", "name", "summary", "inceptionDate", "expenseRatio",
                "assets", "updated", "address_line1", "address_line2", "address_line3",
                "price_nav", "price_currency", "price_fiftyTwoWeek_low", "price_fiftyTwoWeek_high",
                "price_fiftyTwoWeek_changePct", "yields_distribution", "yields_dividendRate",
                "returns_oneYear", "returns_threeYear", "returns_fiveYear", "returns_tenYear",
                "returns_calendar_2015", "returns_calendar_2016", "returns_calendar_2017",
                "returns_calendar_2018", "returns_calendar_2019", "returns_calendar_2020",
                "returns_calendar_2021", "returns_calendar_2022", "returns_calendar_2023",
                "returns_calendar_2024", "ratings_morningstarOverall", "ratings_morningstarRisk",
                "ratings_beta3Year"
            ],
            "temporal_range": {
                "earliest_year": 2015,  # Based on returns_calendar_2015
                "latest_year": 2024,     # Based on returns_calendar_2024
                "note": "Calendar year returns available 2015-2024"
            }
        },
        "fidelity_bonds": {
            "columns": [
                "Description", "Coupon", "Coupon Frequency", "Maturity Date",
                "Moody's Rating", "S&P Rating", "Expected Price", "Expected Yield",
                "Call Protected", "Offering Period", "Settlement Date", "Attributes"
            ],
            "temporal_range": None  # No explicit temporal columns
        }
    }

    def __init__(self, registry: Any, runtime: Any = None):
        """
        Initialize the data availability checker.

        Args:
            registry: ToolRegistry instance with existing tools/tables info
            runtime: GenieRuntime instance (optional, for live DB queries)
        """
        self.registry = registry
        self.runtime = runtime
        self.known_tables = list(self.SCHEMA.keys())

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

        # Phase 1: Check temporal constraints (fast, deterministic)
        temporal_result = self._check_temporal_constraints(user_utterance, query_context)
        if temporal_result:
            logger.info(f"Temporal constraint detected: {temporal_result.issue_type}")
            return temporal_result

        # Phase 2: Check missing columns (fast, deterministic)
        column_result = self._check_missing_columns(user_utterance, query_context)
        if column_result:
            logger.info(f"Missing column detected: {column_result.issue_type}")
            return column_result

        # Phase 3: Check missing tables (fast, deterministic)
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
            technical_details={"checked": ["temporal", "columns", "tables"]}
        )

    def _check_temporal_constraints(self, utterance: str, context: Dict) -> Optional[DataAvailabilityResult]:
        """
        Check if query asks for data outside available time range.

        Examples:
        - "What were FXAIX returns in 1990?" → 1990 < 2015 (earliest data)
        - "Show me 2025 performance" → 2025 > 2024 (latest data)
        """
        # Extract years from utterance using regex
        year_pattern = r'\b(19\d{2}|20\d{2})\b'
        years_mentioned = [int(year) for year in re.findall(year_pattern, utterance)]

        if not years_mentioned:
            return None  # No year mentioned, no temporal constraint

        # Check which table is being queried (use LLM for semantic understanding)
        target_table = self._identify_target_table(utterance)

        if target_table not in self.SCHEMA:
            return None  # Unknown table, can't check constraints

        temporal_range = self.SCHEMA[target_table].get("temporal_range")
        if not temporal_range:
            return None  # Table has no temporal constraints

        earliest = temporal_range["earliest_year"]
        latest = temporal_range["latest_year"]

        # Check if any mentioned year is out of range
        out_of_range_years = [y for y in years_mentioned if y < earliest or y > latest]

        if out_of_range_years:
            # Generate user-facing explanation
            if all(y < earliest for y in out_of_range_years):
                explanation = (
                    f"I have data for {target_table}, but only from {earliest} onwards. "
                    f"Historical data from {', '.join(map(str, out_of_range_years))} is not available in our database."
                )
            elif all(y > latest for y in out_of_range_years):
                explanation = (
                    f"I have data for {target_table} up to {latest}. "
                    f"Future data for {', '.join(map(str, out_of_range_years))} is not yet available."
                )
            else:
                explanation = (
                    f"I have data for {target_table} from {earliest} to {latest}. "
                    f"Data for {', '.join(map(str, out_of_range_years))} is outside this range."
                )

            return DataAvailabilityResult(
                issue_type="temporal_constraint",
                has_capability=True,  # We CAN query the table, just not this time range
                explanation=explanation,
                technical_details={
                    "table": target_table,
                    "available_range": f"{earliest}-{latest}",
                    "requested_years": out_of_range_years
                },
                suggested_workaround=f"Try asking about data from {earliest} to {latest}."
            )

        return None  # Years mentioned are within range

    def _check_missing_columns(self, utterance: str, context: Dict) -> Optional[DataAvailabilityResult]:
        """
        Check if query needs a column that doesn't exist in the target table.

        Examples:
        - "What's the expense ratio for bond XYZ?" → fidelity_bonds lacks expense_ratio
        - "Show me the beta for bonds" → fidelity_bonds lacks ratings_beta3Year
        """
        # Identify target table
        target_table = self._identify_target_table(utterance)

        if target_table not in self.SCHEMA:
            return None  # Unknown table, can't check schema

        available_columns = self.SCHEMA[target_table]["columns"]

        # Use LLM to identify what column/field is being requested
        requested_column = self._identify_requested_column(utterance, target_table, available_columns)

        if requested_column and requested_column not in available_columns:
            # Find similar columns that DO exist (for suggestions)
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

        return None  # Column exists or couldn't identify requested column

    def _check_missing_tables(self, utterance: str, context: Dict) -> Optional[DataAvailabilityResult]:
        """
        Check if query needs a table/data source that doesn't exist.

        Examples:
        - "Show me stock prices" → No stocks table
        - "What ETFs do you have?" → No ETF table
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
                suggested_workaround=f"Try asking about funds or bonds instead."
            )

        return None  # Table exists or couldn't identify requested table

    def _identify_target_table(self, utterance: str) -> Optional[str]:
        """
        Use keyword matching to identify which table is being queried.

        Simple heuristic approach (can be enhanced with LLM if needed).
        """
        lowercase_utterance = utterance.lower()

        # Keyword matching
        if any(keyword in lowercase_utterance for keyword in ["fund", "mutual fund", "equity", "etf"]):
            return "fidelity_funds"
        elif any(keyword in lowercase_utterance for keyword in ["bond", "cd", "certificate", "deposit"]):
            return "fidelity_bonds"

        # Default to fidelity_funds (most common)
        return "fidelity_funds"

    def _identify_requested_column(self, utterance: str, table: str, available_columns: List[str]) -> Optional[str]:
        """
        Use LLM to identify what column/field is being requested.

        This uses semantic understanding to map natural language to column names.
        """
        try:
            prompt = f"""Given this user query about {table} data:
"{utterance}"

What specific column or field is the user asking about? Identify the field name they want, even if it might not exist in the database.

Respond with ONLY the field name (use snake_case or camelCase as appropriate), or "NONE" if the query is too general and doesn't ask for a specific field.

Examples:
Query: "What's the expense ratio for fund XYZ?" → expense_ratio
Query: "Show me the beta" → beta
Query: "What's the Moody's rating?" → moodys_rating
Query: "Tell me about this fund" → NONE (too general)
Query: "What's the dividend yield?" → dividend_yield

For reference, available columns in {table} are:
{', '.join(available_columns)}

But respond with what the USER is asking for, not what's available.

Field name:"""

            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50
            )

            identified_column = response.choices[0].message.content.strip()

            if identified_column == "NONE" or not identified_column:
                return None

            # Clean up the response (remove quotes, extra text)
            cleaned_column_name = identified_column.strip('"\'').split()[0]

            return cleaned_column_name if cleaned_column_name != "NONE" else None

        except Exception as e:
            logger.warning(f"Error identifying requested column: {e}")
            return None

    def _identify_requested_table(self, utterance: str) -> Optional[str]:
        """
        Use LLM to identify what table/data source is being requested.
        """
        try:
            prompt = f"""Given this user query:
"{utterance}"

What data source or table is the user asking about? Choose from:
- fidelity_funds (mutual funds, ETFs, equity funds)
- fidelity_bonds (bonds, CDs, fixed income)
- OTHER (if asking about something we don't have, like stocks, ETFs, real estate, etc.)

Respond with ONLY the table name or "OTHER".

Table name:"""

            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50
            )

            identified_table = response.choices[0].message.content.strip().lower()

            if identified_table == "other":
                # Ask follow-up to identify what they're looking for
                return self._extract_data_source_from_query(utterance)
            elif identified_table in self.known_tables:
                return identified_table

            return None

        except Exception as e:
            logger.warning(f"Error identifying requested table: {e}")
            return None

    def _extract_data_source_from_query(self, utterance: str) -> Optional[str]:
        """Extract the data source name from the query (e.g., 'stocks', 'ETFs')."""
        # Simple keyword extraction for common data sources
        keywords = {
            "stock": "stocks",
            "etf": "ETFs",
            "equity": "equities",
            "real estate": "real estate",
            "reit": "REITs",
            "commodity": "commodities",
            "crypto": "cryptocurrency",
            "option": "options",
        }

        lowercase_utterance = utterance.lower()
        for search_keyword, data_source_name in keywords.items():
            if search_keyword in lowercase_utterance:
                return data_source_name

        return "unknown_data_source"

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


# Utility function for testing
def test_data_availability_checker():
    """Test the data availability checker with sample queries."""

    class MockRegistry:
        pass

    checker = DataAvailabilityChecker(MockRegistry())

    # Test 1: Temporal constraint
    result1 = checker.check_availability("What were FXAIX returns in 1990?")
    print(f"Test 1 (Temporal): {result1.issue_type}")
    print(f"  Explanation: {result1.explanation}\n")

    # Test 2: Missing column
    result2 = checker.check_availability("What's the expense ratio for bond CUSIP 12345?")
    print(f"Test 2 (Missing Column): {result2.issue_type}")
    print(f"  Explanation: {result2.explanation}\n")

    # Test 3: No issue (valid query)
    result3 = checker.check_availability("What are the returns for FXAIX in 2023?")
    print(f"Test 3 (Valid Query): {result3.issue_type}")
    print(f"  Explanation: {result3.explanation or 'No issue detected'}\n")


if __name__ == "__main__":
    test_data_availability_checker()
