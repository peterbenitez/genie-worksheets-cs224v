"""
AI-Powered Tool Discovery System for Hallucination Analysis

This module automatically identifies and tracks missing agent capabilities when
hallucinations occur. When the agent can't answer with existing tools, an LLM
analyzes what tool would be needed and generates a specification for it.

Key Features:
- Synchronous tool discovery during no_response() calls
- Smart deduplication: reuses existing missing tools instead of creating duplicates
- Frequency tracking: prioritizes which tools to implement based on usage
- Persistent storage: saves discoveries across conversations
- Auto-generated implementation reports

Usage:
    from tool_discovery import ToolRegistry, discover_missing_tool
    
    # In no_response() function:
    registry = ToolRegistry.load_or_create()
    tool_spec, is_new = discover_missing_tool(
        user_utterance="What tech ETFs have high returns?",
        registry=registry
    )
    registry.record_missing_tool(tool_spec, context={...})
    registry.save_to_disk()
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from loguru import logger
from openai import AzureOpenAI


# Load environment variables
ENV_PATH = Path(__file__).parent.parent.parent.parent.parent / ".env"
load_dotenv(ENV_PATH)

# Constants
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_TEMPERATURE = 0.0  # Low temperature for consistent tool generation
REGISTRY_FILENAME = "tool_registry.json"


@dataclass
class ToolParameter:
    """Single parameter specification for a tool"""
    name: str
    type: str  # e.g., "str", "float", "List[str]"
    description: str
    required: bool = True


@dataclass
class HypotheticalTool:
    """A tool that SHOULD exist but doesn't (discovered through hallucinations)"""
    tool_name: str
    description: str
    parameters: List[ToolParameter]
    return_type: str
    category: str  # e.g., "api_function", "kb_table_column", "new_api"

    # Tracking metadata
    frequency: int = 0
    first_seen: str = ""
    last_seen: str = ""
    conversation_contexts: List[Dict] = field(default_factory=list)
    
    # Version history tracking
    version: int = 1
    version_history: List[Dict] = field(default_factory=list)  # List of ToolVersionHistory as dicts


@dataclass
class ToolRegistry:
    """
    Tracks existing and missing tools with disk persistence.

    This registry maintains a catalog of what tools the agent has and what
    tools it needs (discovered through hallucinations). It persists to disk
    after each discovery.
    """

    # What we have
    existing_apis: Dict[str, Dict] = field(default_factory=dict)
    existing_kb_tables: Dict[str, List[str]] = field(default_factory=dict)

    # What's missing (discovered through hallucinations)
    missing_tools: Dict[str, HypotheticalTool] = field(default_factory=dict)
    
    # Metadata
    registry_version: str = "2.0"  # Bumped for new schema
    last_updated: str = ""
    
    @classmethod
    def load_or_create(cls, registry_path: Optional[Path] = None) -> 'ToolRegistry':
        """
        Load existing registry from disk, or create new one.
        
        Args:
            registry_path: Path to registry file (uses default if None)
            
        Returns:
            ToolRegistry instance
        """
        if registry_path is None:
            registry_path = cls._get_default_registry_path()
        
        if registry_path.exists():
            # Load existing
            with open(registry_path, 'r') as registry_file:
                registry_data = json.load(registry_file)
            registry = cls._from_dict(registry_data)
            logger.info(f"Loaded tool registry with {len(registry.missing_tools)} missing tools")
        else:
            # Create new
            registry = cls()
            registry._initialize_existing_tools()
            logger.info("Created new tool registry")
        
        return registry
    
    def _initialize_existing_tools(self):
        """
        Populate existing_apis and existing_kb_tables from actual agent tools.
        
        These are the GenieWorksheets the investment agent has access to.
        """
        # GenieWorksheets (agent's available worksheets/forms) with semantic metadata
        self.existing_apis = {
            "UserProfile": {
                "description": "User profile with ID and risk profile",
                "signature": "UserProfile(user_id: str, risk_profile: str)",
                "returns": {
                    "type": "Dict",
                    "fields": ["user_id", "risk_profile", "age", "goals"],
                    "description": "User demographic and investment preferences"
                },
                "capabilities": {
                    "can_do": [
                        "Get user's risk tolerance (conservative, moderate, aggressive)",
                        "Retrieve user age and investment goals",
                        "Access user preferences"
                    ],
                    "cannot_do": [
                        "Calculate or modify risk profiles",
                        "Provide investment recommendations",
                        "Access account balances or holdings"
                    ]
                },
                "data_dependencies": [],
                "when_to_use": "When you need user preferences before making recommendations",
                "constraints": [
                    "Read-only - cannot modify user profile",
                    "Requires valid user_id"
                ],
                "example_queries": [
                    "What's my risk profile?",
                    "What are my investment goals?",
                    "Am I conservative or aggressive?"
                ]
            },
            "GetRecommendation": {
                "description": "Get investment recommendations based on risk profile",
                "signature": "GetRecommendation(value_to_invest: float)",
                "returns": {
                    "type": "Dict",
                    "fields": ["recommended_funds", "allocation_percentages", "risk_level", "rationale"],
                    "description": "Investment recommendations with suggested allocation"
                },
                "capabilities": {
                    "can_do": [
                        "Suggest funds based on user's risk profile",
                        "Provide allocation percentages across multiple funds",
                        "Match investment goals to appropriate products",
                        "Recommend diversified portfolio"
                    ],
                    "cannot_do": [
                        "Calculate required monthly/annual contributions",
                        "Project future returns or values",
                        "Perform portfolio optimization calculations",
                        "Provide tax optimization advice",
                        "Calculate compound interest or time-value of money"
                    ]
                },
                "data_dependencies": ["UserProfile", "fidelity_funds DB"],
                "when_to_use": "When user needs product recommendations or allocation advice for a specific investment amount",
                "constraints": [
                    "Requires UserProfile to be set first",
                    "Only recommends from funds available in fidelity_funds DB",
                    "Does not perform mathematical projections or calculations",
                    "Recommendations are static, not optimized dynamically"
                ],
                "example_queries": [
                    "What funds should I invest in?",
                    "Recommend an aggressive portfolio",
                    "How should I allocate $10,000?",
                    "What's a good diversified investment?"
                ]
            },
            "GetAccountBalance": {
                "description": "Get the user's current account balance",
                "signature": "GetAccountBalance()",
                "returns": {
                    "type": "float",
                    "description": "Current account balance in dollars"
                },
                "capabilities": {
                    "can_do": [
                        "Retrieve current account balance",
                        "Check available funds for investment"
                    ],
                    "cannot_do": [
                        "Show historical balance changes",
                        "Calculate net worth or total assets",
                        "Access external account balances",
                        "Show pending transactions"
                    ]
                },
                "data_dependencies": [],
                "when_to_use": "When user asks about current balance or available funds",
                "constraints": [
                    "Read-only",
                    "Shows only this account, not total assets"
                ],
                "example_queries": [
                    "What's my account balance?",
                    "How much money do I have?",
                    "What funds are available to invest?"
                ]
            },
            "UsersInvestmentPortfolio": {
                "description": "Get user's current investment holdings",
                "signature": "UsersInvestmentPortfolio()",
                "returns": {
                    "type": "List[Dict]",
                    "fields": ["symbol", "shares", "current_value", "cost_basis"],
                    "description": "List of current investment holdings"
                },
                "capabilities": {
                    "can_do": [
                        "Show current holdings with symbols and quantities",
                        "Display current market value of holdings",
                        "List all invested funds and bonds"
                    ],
                    "cannot_do": [
                        "Calculate portfolio performance or returns",
                        "Show historical holdings",
                        "Analyze portfolio risk or diversification",
                        "Suggest rebalancing actions",
                        "Calculate gains/losses"
                    ]
                },
                "data_dependencies": [],
                "when_to_use": "When user asks about current holdings or portfolio composition",
                "constraints": [
                    "Read-only",
                    "Shows snapshot, not historical data"
                ],
                "example_queries": [
                    "What do I currently own?",
                    "Show my portfolio",
                    "What funds am I invested in?",
                    "What are my holdings?"
                ]
            },
            "CertificateDepositInvestment": {
                "description": "Process a certificate deposit (CD/bond) investment transaction",
                "signature": "CertificateDepositInvestment(certificate_allocation: CertificateDepositAllocation, confirm: bool)",
                "returns": {
                    "type": "Dict",
                    "fields": ["status", "transaction_id", "confirmation"],
                    "description": "Transaction result"
                },
                "capabilities": {
                    "can_do": [
                        "Execute bond/CD purchase",
                        "Process investment transaction",
                        "Confirm or preview transaction"
                    ],
                    "cannot_do": [
                        "Recommend which bonds to buy",
                        "Calculate optimal bond allocation",
                        "Analyze bond risk or yields"
                    ]
                },
                "data_dependencies": ["CertificateDepositAllocation"],
                "when_to_use": "When user wants to execute a bond/CD investment (not for recommendations)",
                "constraints": [
                    "Requires explicit confirmation",
                    "Execution only, no analysis or recommendations"
                ],
                "example_queries": [
                    "Buy this bond",
                    "Invest in this CD",
                    "Execute bond purchase"
                ]
            },
            "FundInvestment": {
                "description": "Process a mutual fund investment transaction",
                "signature": "FundInvestment(fund_allocations: FundAllocation, confirm: bool)",
                "returns": {
                    "type": "Dict",
                    "fields": ["status", "transaction_id", "confirmation"],
                    "description": "Transaction result"
                },
                "capabilities": {
                    "can_do": [
                        "Execute fund purchase",
                        "Process investment transaction",
                        "Confirm or preview transaction"
                    ],
                    "cannot_do": [
                        "Recommend which funds to buy",
                        "Calculate optimal fund allocation",
                        "Analyze fund risk or returns"
                    ]
                },
                "data_dependencies": ["FundAllocation"],
                "when_to_use": "When user wants to execute a fund investment (not for recommendations)",
                "constraints": [
                    "Requires explicit confirmation",
                    "Execution only, no analysis or recommendations"
                ],
                "example_queries": [
                    "Buy this fund",
                    "Invest in FXAIX",
                    "Execute fund purchase"
                ]
            },
            "CertificateDepositAllocation": {
                "description": "Allocation details for certificate deposit investment",
                "signature": "CertificateDepositAllocation(fund_to_invest_in: CertificateDeposit, investment_amount: float)",
                "returns": {
                    "type": "Dict",
                    "description": "Allocation specification for CD/bond"
                },
                "capabilities": {
                    "can_do": [
                        "Specify bond/CD and investment amount"
                    ],
                    "cannot_do": [
                        "Recommend bonds",
                        "Calculate allocation percentages"
                    ]
                },
                "data_dependencies": [],
                "when_to_use": "When constructing CD/bond investment parameters",
                "constraints": ["Data structure only, not an action"],
                "example_queries": []
            },
            "FundAllocation": {
                "description": "Allocation details for fund investment",
                "signature": "FundAllocation(fund_to_invest_in: Fund, investment_amount: float)",
                "returns": {
                    "type": "Dict",
                    "description": "Allocation specification for fund"
                },
                "capabilities": {
                    "can_do": [
                        "Specify fund and investment amount"
                    ],
                    "cannot_do": [
                        "Recommend funds",
                        "Calculate allocation percentages"
                    ]
                },
                "data_dependencies": [],
                "when_to_use": "When constructing fund investment parameters",
                "constraints": ["Data structure only, not an action"],
                "example_queries": []
            }
        }
        
        # Database tables with structured schemas and semantic metadata
        self.existing_kb_tables = {
            "fidelity_funds": {
                "columns": [
                    "id", "symbol", "name", "summary", "inceptionDate", "expenseRatio",
                    "assets", "updated", "address_line1", "address_line2", "address_line3",
                    "price_nav", "price_currency", "price_fiftyTwoWeek_low",
                    "price_fiftyTwoWeek_high", "price_fiftyTwoWeek_changePct",
                    "yields_distribution", "yields_dividendRate",
                    "returns_oneYear", "returns_threeYear", "returns_fiveYear", "returns_tenYear",
                    "returns_calendar_2015", "returns_calendar_2016", "returns_calendar_2017",
                    "returns_calendar_2018", "returns_calendar_2019", "returns_calendar_2020",
                    "returns_calendar_2021", "returns_calendar_2022", "returns_calendar_2023",
                    "returns_calendar_2024",
                    "ratings_morningstarOverall", "ratings_morningstarRisk", "ratings_beta3Year"
                ],
                "column_groups": {
                    "identifiers": ["id", "symbol", "name"],
                    "performance": ["returns_oneYear", "returns_threeYear", "returns_fiveYear", "returns_tenYear"],
                    "calendar_returns": ["returns_calendar_2015", "returns_calendar_2016", "returns_calendar_2017",
                                        "returns_calendar_2018", "returns_calendar_2019", "returns_calendar_2020",
                                        "returns_calendar_2021", "returns_calendar_2022", "returns_calendar_2023",
                                        "returns_calendar_2024"],
                    "risk_ratings": ["ratings_morningstarOverall", "ratings_morningstarRisk", "ratings_beta3Year"],
                    "pricing": ["price_nav", "price_fiftyTwoWeek_low", "price_fiftyTwoWeek_high"],
                    "costs": ["expenseRatio"],
                    "yields": ["yields_distribution", "yields_dividendRate"]
                },
                "temporal_coverage": {
                    "earliest_year": 2015,
                    "latest_year": 2024,
                    "note": "Calendar year returns available from 2015-2024 only",
                    "missing_before": "No historical data before 2015",
                    "missing_after": "Future data not available"
                },
                "available_operations": [
                    "SELECT with WHERE clauses (filter funds)",
                    "Filter by risk rating (ratings_morningstarRisk: conservative/moderate/aggressive)",
                    "Sort by performance metrics (returns_*, expenseRatio)",
                    "Filter by expense ratio or other costs",
                    "Query specific calendar year returns (2015-2024)",
                    "Compare multiple funds"
                ],
                "semantics": {
                    "can_answer": [
                        "Find funds matching specific risk profile (conservative/moderate/aggressive)",
                        "Get historical returns for specific years between 2015-2024",
                        "Filter by expense ratio, ratings, or yields",
                        "Compare fund performance across different time periods",
                        "Find low-cost or high-yield funds",
                        "Query fund details like symbol, name, summary",
                        "Sort funds by 1-year, 3-year, 5-year, or 10-year returns"
                    ],
                    "cannot_answer": [
                        "Historical data before 2015 (e.g., 1990s, 2000s data)",
                        "Future projections or predictions",
                        "Calculate required contributions to reach goals",
                        "Perform portfolio optimization (modern portfolio theory)",
                        "Calculate compound interest or time-value of money",
                        "Tax implications or tax-advantaged strategies",
                        "Real-time price updates (data may be delayed)",
                        "Fund holdings or underlying securities details",
                        "Expense ratios for bonds (only available for funds)"
                    ]
                },
                "common_use_cases": [
                    "Query: 'Find aggressive funds' → WHERE ratings_morningstarRisk = 'high' or 'aggressive'",
                    "Query: 'Best performing funds in 2023' → ORDER BY returns_calendar_2023 DESC LIMIT 10",
                    "Query: 'Low expense ratio funds' → WHERE expenseRatio < 0.5",
                    "Query: 'Funds with high 5-year returns' → WHERE returns_fiveYear > 10.0",
                    "Query: 'Compare FXAIX vs VTSAX' → WHERE symbol IN ('FXAIX', 'VTSAX')"
                ],
                "data_quality_notes": [
                    "Returns are percentages (e.g., 10.5 means 10.5%)",
                    "Some funds may have NULL values for older calendar years if inception was recent",
                    "expense_ratio is annual fee as percentage of assets"
                ]
            },
            "fidelity_bonds": {
                "columns": [
                    "Description", "Coupon", "Coupon Frequency", "Maturity Date",
                    "Moody's Rating", "S&P Rating", "Expected Price", "Expected Yield",
                    "Call Protected", "Offering Period", "Settlement Date", "Attributes"
                ],
                "column_groups": {
                    "identifiers": ["Description"],
                    "pricing": ["Expected Price", "Expected Yield"],
                    "ratings": ["Moody's Rating", "S&P Rating"],
                    "terms": ["Coupon", "Coupon Frequency", "Maturity Date"],
                    "features": ["Call Protected", "Attributes"]
                },
                "temporal_coverage": None,  # No temporal constraints for bonds
                "available_operations": [
                    "SELECT with WHERE clauses (filter bonds)",
                    "Filter by credit rating (Moody's, S&P)",
                    "Sort by yield or price",
                    "Filter by maturity date",
                    "Query bond details"
                ],
                "semantics": {
                    "can_answer": [
                        "Find bonds by credit rating (Moody's or S&P)",
                        "Filter by expected yield or price",
                        "Query bonds with specific maturity dates",
                        "Find call-protected bonds",
                        "Filter by coupon rate or frequency",
                        "List available bonds with their characteristics"
                    ],
                    "cannot_answer": [
                        "Expense ratios for bonds (bonds don't have expense ratios - that's a fund concept)",
                        "Historical bond performance or returns",
                        "Calculate bond yield-to-maturity or duration",
                        "Recommend optimal bond allocation",
                        "Tax implications of bond investments",
                        "Compare bond risk vs bond funds"
                    ]
                },
                "common_use_cases": [
                    "Query: 'High-rated bonds' → WHERE \"Moody's Rating\" IN ('Aaa', 'Aa1', 'Aa2')",
                    "Query: 'Bonds with yield > 5%' → WHERE \"Expected Yield\" > 5.0",
                    "Query: 'Short-term bonds' → WHERE \"Maturity Date\" < '2026-01-01'",
                    "Query: 'Investment grade bonds' → WHERE \"S&P Rating\" IN ('AAA', 'AA', 'A', 'BBB')"
                ],
                "data_quality_notes": [
                    "Expected Yield is annualized percentage",
                    "Ratings may be NULL if bond is unrated",
                    "Call Protected indicates if bond can be called early by issuer",
                    "NO expense_ratio column - bonds don't have expense ratios"
                ]
            }
        }
    
    def record_missing_tool(self, tool_spec: HypotheticalTool, context: Dict):
        """
        Record when a tool is needed but doesn't exist.
        
        If the tool already exists in missing_tools, increments its frequency.
        Otherwise, adds it as a new discovery.
        
        Args:
            tool_spec: The hypothetical tool specification
            context: Context dict with user_utterance, timestamp, etc.
        """
        if tool_spec.tool_name in self.missing_tools:
            if is_update:
                # Replace with updated version but preserve frequency and contexts
                existing_freq = self.missing_tools[tool_spec.tool_name].frequency
                existing_contexts = self.missing_tools[tool_spec.tool_name].conversation_contexts
                existing_first_seen = self.missing_tools[tool_spec.tool_name].first_seen
                
                # Update the tool
                self.missing_tools[tool_spec.tool_name] = tool_spec
                self.missing_tools[tool_spec.tool_name].frequency = existing_freq + 1
                self.missing_tools[tool_spec.tool_name].first_seen = existing_first_seen
                self.missing_tools[tool_spec.tool_name].last_seen = context['timestamp']
                self.missing_tools[tool_spec.tool_name].conversation_contexts = existing_contexts + [context]
                
                logger.info(
                    f"Tool '{tool_spec.tool_name}' updated to v{tool_spec.version} "
                    f"(freq={self.missing_tools[tool_spec.tool_name].frequency})"
                )
            else:
                # Existing missing tool - increment frequency (reuse without changes)
                self.missing_tools[tool_spec.tool_name].frequency += 1
                self.missing_tools[tool_spec.tool_name].last_seen = context['timestamp']
                self.missing_tools[tool_spec.tool_name].conversation_contexts.append(context)
                logger.info(
                    f"Tool '{tool_spec.tool_name}' reused (freq={self.missing_tools[tool_spec.tool_name].frequency})"
                )
        else:
            # New missing tool
            tool_spec.frequency = 1
            tool_spec.first_seen = context['timestamp']
            tool_spec.last_seen = context['timestamp']
            tool_spec.conversation_contexts = [context]
            tool_spec.version = 1
            
            # Add creation entry to version history
            creation_entry = {
                "version": 1,
                "timestamp": context['timestamp'],
                "change_type": "created",
                "change_description": "Initial tool discovery",
                "parameters_snapshot": [asdict(p) for p in tool_spec.parameters],
                "description_snapshot": tool_spec.description
            }
            tool_spec.version_history = [creation_entry]
            
            self.missing_tools[tool_spec.tool_name] = tool_spec
            logger.info(f"New missing tool discovered: '{tool_spec.tool_name}'")
    
    def save_to_disk(self, registry_path: Optional[Path] = None):
        """
        Persist registry to disk using atomic write.
        
        Args:
            registry_path: Path to save to (uses default if None)
        """
        if registry_path is None:
            registry_path = self._get_default_registry_path()
        
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.last_updated = datetime.now().isoformat()
        
        # Atomic write: write to temp file first, then rename
        temp_path = registry_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as temp_file:
                json.dump(self._to_dict(), temp_file, indent=2)
            # Only replace the real file if write succeeded
            temp_path.replace(registry_path)
            logger.debug(f"Saved tool registry to {registry_path}")
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            logger.error(f"Failed to save tool registry: {e}")
            raise
    
    @staticmethod
    def _get_default_registry_path() -> Path:
        """Get default path to registry file"""
        return Path(__file__).parent / REGISTRY_FILENAME
    
    def _to_dict(self) -> dict:
        """Convert to JSON-serializable dict with tools grouped by action type"""
        # Group tools by action type
        grouped_tools = self._group_tools_by_action()
        
        return {
            'registry_version': self.registry_version,
            'last_updated': self.last_updated,
            'existing_apis': self.existing_apis,
            'existing_kb_tables': self.existing_kb_tables,
            'missing_tools': {
                name: self._tool_to_dict(tool) 
                for name, tool in self.missing_tools.items()
            }
        }
    
    @staticmethod
    def _tool_to_dict(tool: HypotheticalTool) -> dict:
        """Convert HypotheticalTool to dict"""
        tool_dict = asdict(tool)
        # Convert ToolParameter objects to dicts
        tool_dict['parameters'] = [
            asdict(p) if isinstance(p, ToolParameter) else p
            for p in tool_dict['parameters']
        ]
        return tool_dict
    
    @classmethod
    def _from_dict(cls, data: dict) -> 'ToolRegistry':
        """Load from dict with grouped format (missing_tools_by_action)"""
        registry = cls()
        registry.registry_version = data.get('registry_version', '1.0')
        registry.last_updated = data.get('last_updated', '')
        registry.existing_apis = data.get('existing_apis', {})
        registry.existing_kb_tables = data.get('existing_kb_tables', {})
        
        # Reconstruct HypotheticalTool objects
        for name, tool_data in data.get('missing_tools', {}).items():
            # Convert parameter dicts to ToolParameter objects
            tool_data['parameters'] = [
                ToolParameter(**p) for p in tool_data['parameters']
            ]
            registry.missing_tools[name] = HypotheticalTool(**tool_data)
        
        return registry
    
    def get_priority_tools(self, min_frequency: int = 2) -> List[HypotheticalTool]:
        """
        Get missing tools sorted by frequency (highest priority first).
        
        Args:
            min_frequency: Minimum frequency threshold
            
        Returns:
            List of HypotheticalTool sorted by frequency descending
        """
        return sorted(
            [t for t in self.missing_tools.values() if t.frequency >= min_frequency],
            key=lambda x: x.frequency,
            reverse=True
        )
    
    def generate_implementation_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate markdown report of tools to implement.
        
        Args:
            output_path: Path to save report (only saves if provided)
            
        Returns:
            The report as a string
        """
        priority_tools = self.get_priority_tools(min_frequency=2)
        
        report = f"""# Tool Implementation Priority Report
Generated: {datetime.now().isoformat()}

Total missing tools discovered: {len(self.missing_tools)}
High priority tools (frequency ≥ 2): {len(priority_tools)}

## Priority Tools to Implement

"""
        for i, tool in enumerate(priority_tools, 1):
            report += f"""
### {i}. `{tool.tool_name}` (Used {tool.frequency} times)

**Description:** {tool.description}

**Category:** {tool.category}

**Parameters:**
"""
            for param in tool.parameters:
                required = "Required" if param.required else "Optional"
                report += f"- `{param.name}` ({param.type}, {required}): {param.description}\n"
            
            report += f"""
**Returns:** {tool.return_type}

**Example User Queries:**
"""
            # Show first 3 queries from conversation contexts
            for ctx in tool.conversation_contexts[:3]:
                query = ctx.get('user_utterance', '')
                report += f'- "{query}"\n'
            
            report += "\n---\n"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Implementation report saved to {output_path}")
        
        return report


class ToolAnalyzer:
    """
    AI-powered analyzer that determines what tool is needed for a query.
    
    Uses Azure OpenAI to intelligently decide whether to reuse an existing
    missing tool or create a new one.
    """
    
    def __init__(self, model: str = DEFAULT_MODEL):
        """
        Initialize the tool analyzer.
        
        Args:
            model: Azure OpenAI model deployment name
        """
        self.model = model
        
        # Initialize Azure OpenAI client
        self._client = AzureOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            api_version=os.getenv("LLM_API_VERSION"),
            azure_endpoint=os.getenv("LLM_API_ENDPOINT"),
        )
    
    def analyze(
        self,
        user_utterance: str,
        registry: ToolRegistry,
        data_availability_result: Optional[Any] = None
    ) -> 'HypotheticalIssue':
        """
        Analyze what capability gap exists for the user's query.

        Args:
            user_utterance: The user's question that couldn't be answered
            registry: ToolRegistry with existing and missing tools
            data_availability_result: Result from data availability checking (Phase 1)

        Returns:
            HypotheticalIssue indicating the type of issue (tool, data, out_of_scope, ambiguous)

        Raises:
            Exception: If LLM API call fails or response parsing fails
        """
        # Phase 1 already found data constraint
        if data_availability_result and data_availability_result.issue_type != "no_issue":
            return HypotheticalIssue(
                issue_type="missing_data",
                data_constraint=data_availability_result.issue_type,
                affected_entity=data_availability_result.technical_details.get("table", "unknown"),
                user_facing_message=data_availability_result.explanation,
                technical_explanation=data_availability_result.technical_details
            )

        prompt = self._build_prompt(user_utterance, registry, data_availability_result)

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Analyze what tools or data sources would help answer the user's investment planning question based on the system prompt context."}
        ]
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=DEFAULT_TEMPERATURE,
                response_format={"type": "json_object"}
            )
        except Exception as e:
            error_str = str(e)
            # Don't crash on Azure content filter blocks - just skip tool discovery
            if any(keyword in error_str.lower() for keyword in ["content_filter", "jailbreak", "responsibleai"]):
                logger.warning(f"Content filter blocked tool discovery (skipping): {error_str[:200]}")
                # Return a generic "unknown_tool" to avoid breaking the flow
                return HypotheticalTool(tool_name="unknown_tool", description="Tool discovery blocked by content filter"), False
            logger.error(f"Tool analyzer LLM call failed: {e}")
            raise
        
        response_text = response.choices[0].message.content.strip()
        
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tool analyzer response: {response_text}")
            raise
        
        return self._process_response(parsed, registry)
    
    def _build_prompt(self, user_utterance: str, registry: ToolRegistry, data_availability_result: Optional[Any] = None) -> str:
        """Build the analysis prompt for the LLM"""

        # Format existing APIs
        apis_text = self._format_existing_apis(registry.existing_apis)

        # Format DB schema
        db_text = self._format_db_schema(registry.existing_kb_tables)

        # Format missing tools
        missing_text = self._format_missing_tools(registry.missing_tools)
        
        prompt = f"""You are analyzing a conversational AI investment agent's tool gaps.

AVAILABLE TOOLS:
{apis_text}
{db_text}

MISSING TOOLS (discovered but not implemented):
{missing_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USER QUERY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"{user_utterance}"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The agent CANNOT answer the user's query with the CURRENTLY AVAILABLE TOOLS.

Determine which missing tool would solve this:

1. If one of the PREVIOUSLY DISCOVERED MISSING TOOLS would work AS-IS, return:
{{
    "action": "reuse_existing",
    "tool_name": "exact_name_from_missing_tools",
    "rationale": "why this existing missing tool fits"
}}

2. If one of the PREVIOUSLY DISCOVERED MISSING TOOLS is SUBSTANTIALLY SIMILAR but needs minor changes, return:
{{
    "action": "update_existing",
    "tool_name": "name_of_tool_to_update",
    "updated_description": "enhanced description (if needed, otherwise copy existing)",
    "new_parameters": [
        {{
            "name": "new_param_name",
            "type": "python_type",
            "description": "what this new parameter does",
            "required": true|false
        }}
    ],
    "parameter_changes": "explanation of what parameters to add/modify/make optional",
    "rationale": "why updating is better than creating new tool"
}}

SUBSTANTIALLY SIMILAR means:
- Same core purpose/domain (e.g., both about calculating returns)
- Overlapping functionality with minor variation (e.g., needs one more filter parameter)
- Same general workflow but different specifics (e.g., time period differs)
- Would confuse users if both existed as separate tools

3. If you need to CREATE A NEW missing tool (genuinely different functionality), return:
{{
    "action": "create_new",
    "tool_name": "descriptive_snake_case_name",
    "description": "what this tool does",
    "category": "api_function|kb_table_column|new_api",
    "parameters": [
        {{
            "name": "param_name",
            "type": "python_type",
            "description": "what this means",
            "required": true
        }}
    ],
    "return_type": "what it returns",
    "rationale": "why existing missing tools don't work and why this is distinct"
}}

DECISION PRIORITY:
1. FIRST, check if existing tool works as-is (reuse_existing)
2. SECOND, check if existing tool can be enhanced (update_existing) 
3. LAST, create new tool only if functionality is genuinely different

IMPORTANT:
- Prefer updating over creating when tools are substantially similar
- Avoid tool proliferation - merge similar functionality
- Be specific about parameter types (str, float, int, List[str], etc.)
- Use snake_case for tool and parameter names
- Return valid JSON only"""
        
        return prompt
    
    @staticmethod
    def _format_existing_apis(apis: Dict[str, Dict]) -> str:
        """Format existing APIs for prompt (STUB)"""
        if not apis:
            return "None"
        
        lines = []
        for name, info in apis.items():
            lines.append(f"- {info['signature']}")
            lines.append(f"  Description: {info['description']}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_db_schema(tables: Dict[str, List[str]]) -> str:
        """Format database schema for prompt (STUB)"""
        if not tables:
            return "None"
        
        lines = []
        for table_name, columns in tables.items():
            lines.append(f"- {table_name}: {', '.join(columns[:10])}")  # Show first 10 columns
            if len(columns) > 10:
                lines.append(f"  ... and {len(columns) - 10} more columns")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_missing_tools(missing_tools: Dict[str, HypotheticalTool]) -> str:
        """Format missing tools for prompt"""
        if not missing_tools:
            return "None discovered yet"
        
        lines = []
        for name, tool in missing_tools.items():
            params = ", ".join([f"{p.name}: {p.type}" for p in tool.parameters])
            lines.append(f"- {name}({params}) -> {tool.return_type}")
            lines.append(f"  Description: {tool.description}")
            lines.append(f"  Used {tool.frequency} times")
        
        return "\n".join(lines)
    
    def _merge_tool_parameters(
        self,
        existing_tool: HypotheticalTool,
        update_request: dict,
        registry: ToolRegistry
    ) -> HypotheticalTool:
        """
        Use LLM to intelligently merge parameters from existing tool and update request.
        
        Args:
            existing_tool: The tool to update
            update_request: The parsed update request from initial LLM call
            registry: ToolRegistry for context
            
        Returns:
            Updated HypotheticalTool with merged parameters and version history
        """
        # Build merge prompt
        merge_prompt = self._build_merge_prompt(existing_tool, update_request)
        
        messages = [
            {"role": "system", "content": merge_prompt},
            {"role": "user", "content": "Intelligently merge the parameters and determine which should be required vs optional."}
        ]
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=DEFAULT_TEMPERATURE,
                response_format={"type": "json_object"}
            )
        except Exception as e:
            logger.error(f"Parameter merge LLM call failed: {e}")
            raise
        
        response_text = response.choices[0].message.content.strip()
        
        try:
            merge_result = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse merge response: {response_text}")
            raise
        
        # Create version history entry for the current state (before update)
        current_version_entry = {
            "version": existing_tool.version,
            "timestamp": existing_tool.last_seen,
            "change_type": "snapshot_before_update",
            "change_description": f"State before v{existing_tool.version + 1} update",
            "parameters_snapshot": [asdict(p) for p in existing_tool.parameters],
            "description_snapshot": existing_tool.description
        }
        
        # Update the tool
        existing_tool.description = merge_result.get("merged_description", existing_tool.description)
        existing_tool.parameters = [ToolParameter(**p) for p in merge_result["merged_parameters"]]
        existing_tool.version += 1
        existing_tool.version_history.append(current_version_entry)
        
        # Add new version entry
        new_version_entry = {
            "version": existing_tool.version,
            "timestamp": datetime.now().isoformat(),
            "change_type": merge_result.get("change_type", "updated"),
            "change_description": merge_result.get("change_summary", "Tool updated with new parameters"),
            "parameters_snapshot": [asdict(p) for p in existing_tool.parameters],
            "description_snapshot": existing_tool.description
        }
        existing_tool.version_history.append(new_version_entry)
        
        return existing_tool
    
    def _build_merge_prompt(self, existing_tool: HypotheticalTool, update_request: dict) -> str:
        """Build prompt for parameter merging"""
        
        existing_params_text = "\n".join([
            f"  - {p.name} ({p.type}, {'required' if p.required else 'optional'}): {p.description}"
            for p in existing_tool.parameters
        ])
        
        new_params_text = "\n".join([
            f"  - {p['name']} ({p['type']}, {'required' if p['required'] else 'optional'}): {p['description']}"
            for p in update_request.get("new_parameters", [])
        ])
        
        prompt = f"""You are merging parameters for a tool that is being updated.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXISTING TOOL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name: {existing_tool.tool_name}
Description: {existing_tool.description}
Parameters:
{existing_params_text}
Return Type: {existing_tool.return_type}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UPDATE REQUEST:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
New/Updated Description: {update_request.get('updated_description', 'No change')}
New Parameters to Add:
{new_params_text if new_params_text else '  None specified'}
Reasoning: {update_request.get('parameter_changes', 'Not specified')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Intelligently merge these parameters into a unified parameter list. Consider:

1. **Core vs Optional Parameters:**
   - Parameters that are ESSENTIAL to the tool's primary function should be required=true
   - Parameters that add flexibility or filtering should be required=false
   - If a parameter was required before and is still core, keep it required
   - If a new parameter is for specialized use cases, make it optional

2. **Parameter Deduplication:**
   - If new parameter is essentially the same as existing, keep one version with best description
   - If parameters overlap in purpose, merge them with a broader type/description

3. **Parameter Enhancement:**
   - If new parameter adds to existing parameter (e.g., adding filter options), enhance the existing one
   - Update descriptions to be clearer and more comprehensive

4. **Description Update:**
   - Merge the descriptions to cover all use cases
   - Keep it concise but comprehensive

Return JSON in this format:
{{
    "merged_description": "comprehensive description covering all use cases",
    "merged_parameters": [
        {{
            "name": "parameter_name",
            "type": "python_type",
            "description": "what it does",
            "required": true|false
        }}
    ],
    "change_type": "parameter_added|parameter_modified|description_updated|comprehensive_update",
    "change_summary": "brief summary of what changed and why"
}}

IMPORTANT:
- Return ALL parameters (existing + new), not just changes
- Be thoughtful about required vs optional
- Avoid parameter bloat - merge similar concepts
- Return valid JSON only"""
        
        return prompt
    
    def _process_response(
        self,
        parsed: dict,
        registry: ToolRegistry
    ) -> Tuple[HypotheticalTool, bool, bool]:
        """Process the LLM's response and return tool spec with flags"""
        
        if parsed["action"] == "reuse_existing":
            # Find and return existing missing tool
            tool_name = parsed["tool_name"]
            
            if tool_name not in registry.missing_tools:
                raise ValueError(
                    f"LLM suggested reusing '{tool_name}' but it doesn't exist in missing_tools"
                )
            
            existing_tool = registry.missing_tools[tool_name]
            logger.info(f"Reusing existing missing tool: {tool_name}")
            return existing_tool, False, False  # Not new, not updated
        
        elif parsed["action"] == "update_existing":
            # Update an existing missing tool with enhanced parameters
            tool_name = parsed["tool_name"]
            
            if tool_name not in registry.missing_tools:
                raise ValueError(
                    f"LLM suggested updating '{tool_name}' but it doesn't exist in missing_tools"
                )
            
            existing_tool = registry.missing_tools[tool_name]
            
            # Use LLM to intelligently merge parameters
            updated_tool = self._merge_tool_parameters(
                existing_tool=existing_tool,
                update_request=parsed,
                registry=registry
            )
            
            logger.info(f"Updated existing missing tool: {tool_name} (v{updated_tool.version})")
            return updated_tool, False, True  # Not new, but is updated
        
        elif parsed["action"] == "create_new":
            # Create new HypotheticalTool
            new_tool = HypotheticalTool(
                tool_name=parsed["tool_name"],
                description=parsed["description"],
                parameters=[ToolParameter(**p) for p in parsed["parameters"]],
                return_type=parsed["return_type"],
                category=parsed["category"]
            )
            logger.info(f"Creating new missing tool: {new_tool.tool_name}")
            return new_tool, True, False  # Is new, not updated
        
        else:
            raise ValueError(f"Unknown classification in LLM response: {classification}")


def discover_missing_capability(
    user_utterance: str,
    registry: ToolRegistry,
    data_availability_result: Optional[Any] = None
) -> HypotheticalIssue:
    """
    Main entry point for capability gap discovery (Phase 2 of hybrid approach).

    Analyzes a user query that the agent couldn't answer and determines
    what type of issue it is: missing tool, out of scope, or ambiguous.

    Args:
        user_utterance: The user's question
        registry: ToolRegistry with current tool state
        data_availability_result: Result from Phase 1 data checking (optional)

    Returns:
        HypotheticalIssue indicating the type of gap

    Raises:
        Exception: If analysis fails
    """
    analyzer = ToolAnalyzer()
    return analyzer.analyze(user_utterance, registry, data_availability_result)


# Backward compatibility alias
def discover_missing_tool(
    user_utterance: str,
    registry: ToolRegistry
) -> Tuple[HypotheticalTool, bool]:
    """
    DEPRECATED: Use discover_missing_capability() instead.

    This function is kept for backward compatibility but will be removed in a future version.
    """
    logger.warning("discover_missing_tool() is deprecated. Use discover_missing_capability() instead.")
    issue = discover_missing_capability(user_utterance, registry, None)

    if issue.issue_type == "missing_tool" and issue.tool_spec:
        # Return old format: (tool_spec, is_new)
        # We can't determine is_new here, so default to True
        return issue.tool_spec, True
    else:
        # For non-tool issues, return a dummy tool
        dummy_tool = HypotheticalTool(
            tool_name="unknown_capability",
            description="Unknown capability gap",
            parameters=[],
            return_type="unknown",
            category="unknown"
        )
        return dummy_tool, False


