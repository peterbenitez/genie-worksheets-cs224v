"""
AI-Powered Tool Discovery System for Hallucination Analysis

This module automatically identifies and tracks missing agent capabilities when
hallucinations occur. When the agent can't answer with existing tools, an LLM
analyzes what tool would be needed and generates a specification for it.

Key Features:
- Synchronous tool discovery during no_response() calls
- Smart deduplication: reuses existing missing tools instead of creating duplicates
- Intelligent updates: merges similar tools rather than creating duplicates
- Version history: tracks how tools evolve over time with parameter changes
- LLM-based parameter merging: intelligently determines core vs optional parameters
- Frequency tracking: prioritizes which tools to implement based on usage
- Persistent storage: saves discoveries across conversations
- Auto-generated implementation reports

Usage:
    from tool_discovery import ToolRegistry, discover_missing_tool
    
    # In no_response() function:
    registry = ToolRegistry.load_or_create()
    tool_spec, is_new, is_update = discover_missing_tool(
        user_utterance="What tech ETFs have high returns?",
        registry=registry
    )
    registry.record_missing_tool(tool_spec, context={...}, is_update=is_update)
    registry.save_to_disk()
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
class ToolVersionHistory:
    """Version history entry for a tool"""
    version: int
    timestamp: str
    change_type: str  # "created", "parameter_added", "parameter_modified", "description_updated"
    change_description: str
    parameters_snapshot: List[ToolParameter]
    

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
    conversation_contexts: List[Dict] = field(default_factory=list)  # Each context has semantic_mapping and example_usage
    
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
    registry_version: str = "1.0"
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
            with open(registry_path, 'r') as f:
                data = json.load(f)
            registry = cls._from_dict(data)
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
        # GenieWorksheets (agent's available worksheets/forms)
        self.existing_apis = {
            "UserProfile": {
                "description": "User profile with ID and risk profile",
                "signature": "UserProfile(user_id: str, risk_profile: str)"
            },
            "GetRecommendation": {
                "description": "Get investment recommendations",
                "signature": "GetRecommendation(value_to_invest: float)"
            },
            "GetAccountBalance": {
                "description": "Get the user's account balance",
                "signature": "GetAccountBalance()"
            },
            "UsersInvestmentPortfolio": {
                "description": "Get user's current investment portfolio",
                "signature": "UsersInvestmentPortfolio()"
            },
            "CertificateDepositInvestment": {
                "description": "Process a certificate deposit (CD/bond) investment",
                "signature": "CertificateDepositInvestment(certificate_allocation: CertificateDepositAllocation, confirm: bool)"
            },
            "FundInvestment": {
                "description": "Process a fund investment",
                "signature": "FundInvestment(fund_allocations: FundAllocation, confirm: bool)"
            },
            "CertificateDepositAllocation": {
                "description": "Allocation details for certificate deposit investment",
                "signature": "CertificateDepositAllocation(fund_to_invest_in: CertificateDeposit, investment_amount: float)"
            },
            "FundAllocation": {
                "description": "Allocation details for fund investment",
                "signature": "FundAllocation(fund_to_invest_in: Fund, investment_amount: float)"
            }
        }
        
        # Database tables with all available columns (from genie_db_models)
        self.existing_kb_tables = {
            "fidelity_funds": [
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
            "fidelity_bonds": [
                "Description", "Coupon", "Coupon Frequency", "Maturity Date",
                "Moody's Rating", "S&P Rating", "Expected Price", "Expected Yield",
                "Call Protected", "Offering Period", "Settlement Date", "Attributes"
            ]
        }
    
    def record_missing_tool(self, tool_spec: HypotheticalTool, context: Dict, is_update: bool = False):
        """
        Record when a tool is needed but doesn't exist.
        
        If the tool already exists in missing_tools, increments its frequency.
        Otherwise, adds it as a new discovery.
        
        Args:
            tool_spec: The hypothetical tool specification
            context: Context dict with user_utterance, timestamp, etc.
            is_update: If True, this is an updated version of an existing tool
        """
        # Add semantic_mapping and example_usage to context if they exist (from new tool creation)
        if hasattr(tool_spec, '_query_mapping'):
            context['semantic_mapping'] = tool_spec._query_mapping.get('semantic_mapping', '')
            context['example_usage'] = tool_spec._query_mapping.get('example_usage', '')
            delattr(tool_spec, '_query_mapping')  # Remove temporary attribute
        
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
            # New missing tool - add version history entry for creation
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
            with open(temp_path, 'w') as f:
                json.dump(self._to_dict(), f, indent=2)
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
            'missing_tools_by_action': grouped_tools
        }
    
    def _group_tools_by_action(self) -> dict:
        """Group tools by their primary action for better readability"""
        from collections import defaultdict
        
        groups = defaultdict(dict)
        
        for name, tool in self.missing_tools.items():
            # Extract action from tool name (first word before underscore)
            # e.g., "explain_allocation" -> "explain", "calculate_returns" -> "calculate"
            action = self._extract_action(name)
            groups[action][name] = self._tool_to_dict(tool)
        
        # Sort groups by action name and convert to regular dict
        return dict(sorted(groups.items()))
    
    @staticmethod
    def _extract_action(tool_name: str) -> str:
        """Extract primary action from tool name"""
        # Common action verbs (ordered by priority)
        actions = [
            'compare', 'explain', 'calculate', 'compute', 'retrieve', 'get',
            'evaluate', 'assess', 'justify', 'recommend', 'list', 'find',
            'analyze', 'identify', 'determine', 'generate', 'create', 'build',
            'validate', 'check', 'verify', 'transform', 'convert', 'format'
        ]
        
        tool_lower = tool_name.lower()
        
        # Check if tool starts with any known action
        for action in actions:
            if tool_lower.startswith(action):
                return action
        
        # If no known action, put in misc
        return 'misc'
    
    @staticmethod
    def _tool_to_dict(tool: HypotheticalTool) -> dict:
        """Convert HypotheticalTool to dict"""
        tool_dict = asdict(tool)
        # Convert ToolParameter objects to dicts
        tool_dict['parameters'] = [
            asdict(p) if isinstance(p, ToolParameter) else p
            for p in tool_dict['parameters']
        ]
        # version_history is already serialized as dicts in our code
        return tool_dict
    
    @classmethod
    def _from_dict(cls, data: dict) -> 'ToolRegistry':
        """Load from dict with grouped format (missing_tools_by_action)"""
        registry = cls()
        registry.registry_version = data.get('registry_version', '1.0')
        registry.last_updated = data.get('last_updated', '')
        registry.existing_apis = data.get('existing_apis', {})
        registry.existing_kb_tables = data.get('existing_kb_tables', {})
        
        # Load tools from grouped format
        for action_group in data.get('missing_tools_by_action', {}).values():
            for name, tool_data in action_group.items():
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
### {i}. `{tool.tool_name}` (Used {tool.frequency} times, Version {tool.version})

**Description:** {tool.description}

**Category:** {tool.category}

**Parameters:**
"""
            for param in tool.parameters:
                required = "Required" if param.required else "Optional"
                report += f"- `{param.name}` ({param.type}, {required}): {param.description}\n"
            
            report += f"""
**Returns:** {tool.return_type}

"""
            # Show version history if tool has been updated
            if tool.version > 1 and tool.version_history:
                report += "**Version History:**\n"
                for version_entry in tool.version_history:
                    report += f"- v{version_entry['version']} ({version_entry['change_type']}): {version_entry['change_description']}\n"
                report += "\n"
            
            report += "**Example User Queries:**\n"
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
        registry: ToolRegistry
    ) -> Tuple[HypotheticalTool, bool, bool]:
        """
        Analyze what tool is needed for the user's query.
        
        Args:
            user_utterance: The user's question that couldn't be answered
            registry: ToolRegistry with existing and missing tools
            
        Returns:
            Tuple of (tool_spec, is_new, is_update) where:
                - tool_spec is the HypotheticalTool (existing, updated, or new)
                - is_new is True if this is a newly created tool
                - is_update is True if this is an updated existing tool
                
        Raises:
            Exception: If LLM API call fails or response parsing fails
        """
        prompt = self._build_prompt(user_utterance, registry)

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
                return HypotheticalTool(tool_name="unknown_tool", description="Tool discovery blocked by content filter"), False, False
            logger.error(f"Tool analyzer LLM call failed: {e}")
            raise
        
        response_text = response.choices[0].message.content.strip()
        
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tool analyzer response: {response_text}")
            raise
        
        return self._process_response(parsed, registry)
    
    def _build_prompt(self, user_utterance: str, registry: ToolRegistry) -> str:
        """Build the analysis prompt for the LLM"""
        
        # Format existing APIs
        apis_text = self._format_existing_apis(registry.existing_apis)
        
        # Format DB schema
        db_text = self._format_db_schema(registry.existing_kb_tables)
        
        # Format missing tools
        missing_text = self._format_missing_tools(registry.missing_tools)
        
        prompt = f"""You are analyzing a conversational AI agent's tool gaps.

AVAILABLE TOOLS:
{apis_text}
{db_text}

MISSING TOOLS (discovered but not implemented):
{missing_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK: The agent cannot answer this query with available tools.
USER QUERY: "{user_utterance}"

TOOL GRANULARITY PRINCIPLE:
A tool = ONE specific capability (like a function) that:
1. Has ONE primary action (compute, explain, compare, justify, retrieve, list, evaluate, transform)
2. Returns ONE type of output (number, text explanation, comparison, data list, judgment)
3. Can be parameterized for variations of the SAME task

SIMILARITY MATCHING RULES (CRITICAL):
When comparing query to existing tools, IGNORE minor wording differences!

THESE ARE THE SAME:
- "compare X on metrics" = "compare X head-to-head" = "compare X directly"
- "get data about X" = "retrieve X info" = "fetch X details"
- "explain why X" = "justify X" = "provide reasoning for X"
- "key metrics" = "main factors" = "important criteria"

DO NOT create separate tools for:
- Synonyms in description ("head-to-head" vs "on metrics")
- Paraphrasing ("data-driven justification" vs "concrete comparison")
- Different wordings of same request

DECISION FRAMEWORK:
Step 1: Extract functional signature
- PRIMARY ACTION: What operation? (compare, explain, calculate, get, etc.)
- RETURN TYPE: What output format?
- SUBJECT: What entity?
- PARAMETERS: What varies?

Step 2: Check EACH existing missing tool
For EACH tool, ask:
1. Same PRIMARY ACTION verb? (compare=compare, explain=explain, calculate=calculate)
2. Same RETURN TYPE structure? (comparison_table=comparison_table, explanation=explanation)
3. Same SUBJECT entity? (ETFs=ETFs, funds=funds, bonds=bonds)
4. Same PARAMETERS (ignoring entity names/values)?

IF ALL 4 match -> REUSE existing tool (don't create new!)
IF 1-3 match but need new optional param -> UPDATE existing
IF ANY of 1-3 differ -> CREATE new

CONCRETE EXAMPLES:
Query A: "Compare BOTZ vs AIQ on fees, holdings, performance"
Query B: "Compare BOTZ vs AIQ head-to-head with numbers"
-> SAME TOOL (both compare ETFs, return comparison, just synonym phrasing)
-> ACTION: reuse_existing

Query A: "Calculate 5-year returns"
Query B: "Calculate 10-year returns"  
-> SAME TOOL (same action, return, subject - only year value differs)
-> ACTION: reuse_existing

Query A: "Compare fund performance"
Query B: "Explain why fund performed well"
-> DIFFERENT TOOLS (compare != explain, comparison != explanation)
-> ACTION: create_new

Query A: "Get ETF expense ratios"
Query B: "Get ETF expense ratios for tech sector"
-> SAME TOOL but needs sector filter
-> ACTION: update_existing (add sector param)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT:

BEFORE deciding, check EVERY existing missing tool carefully!
Don't create duplicates just because of synonym wording!

1. REUSE (same action, return, subject, params - just different wording or entity values):
{{
    "action": "reuse_existing",
    "tool_name": "exact_tool_name",
    "reasoning": "why this matches (be specific about action/return/subject match)"
}}

2. UPDATE (same action + return + subject, but needs additional optional param):
{{
    "action": "update_existing",
    "tool_name": "tool_to_update",
    "new_parameters": [{{"name": "param", "type": "type", "description": "desc", "required": false}}],
    "parameter_changes": "what new capability this adds",
    "reasoning": "why updating instead of reusing or creating new"
}}

3. CREATE (different action OR return type OR subject):
{{
    "action": "create_new",
    "tool_name": "specific_action_name",
    "description": "one specific thing this does",
    "category": "api_function|kb_table_column|new_api",
    "parameters": [{{"name": "param", "type": "type", "description": "desc", "required": true|false}}],
    "return_type": "output format",
    "semantic_mapping_for_this_query": "For THIS query, extract param1 from [where]. Returns [format] to answer [what].",
    "example_usage_for_this_query": "tool_name(param1=value_from_query)",
    "reasoning": "why this needs separate tool (which of action/return/subject differs)"
}}

REMEMBER: Minor description wording changes DO NOT justify separate tools!
Return valid JSON only."""
        
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
        """Format missing tools for prompt, grouped by action type"""
        if not missing_tools:
            return "None discovered yet"
        
        # Group tools by action
        from collections import defaultdict
        grouped = defaultdict(list)
        
        for name, tool in missing_tools.items():
            # Extract action from tool name
            action = name.split('_')[0] if '_' in name else name
            grouped[action].append((name, tool))
        
        # Format by group
        lines = []
        for action in sorted(grouped.keys()):
            lines.append(f"\n{action.upper()} TOOLS:")
            for name, tool in grouped[action]:
                params = ", ".join([f"{p.name}: {p.type}" for p in tool.parameters])
                lines.append(f"  - {name}({params}) -> {tool.return_type}")
                lines.append(f"    Description: {tool.description}")
                lines.append(f"    Used {tool.frequency} times")
        
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
        
        prompt = f"""Merge parameters for tool update.

EXISTING: {existing_tool.tool_name}
{existing_params_text}

NEW PARAMETERS:
{new_params_text if new_params_text else 'None'}

TASK: Merge into unified parameter list.
- Essential params -> required=true
- Filters/options -> required=false
- Deduplicate similar params
- Return ALL params (existing + new)

Return JSON:
{{
    "merged_description": "brief description",
    "merged_parameters": [
        {{
            "name": "parameter_name",
            "type": "python_type",
            "description": "what it does",
            "required": true|false
        }}
    ],
    "change_type": "parameter_added|parameter_modified|description_updated",
    "change_summary": "what changed"
}}"""
        
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
            reasoning = parsed.get("reasoning", "No reasoning provided")
            logger.info(f"Reusing existing missing tool: {tool_name}")
            logger.debug(f"Reuse reasoning: {reasoning}")
            return existing_tool, False, False  # Not new, not updated
        
        elif parsed["action"] == "update_existing":
            # Update an existing missing tool with enhanced parameters
            tool_name = parsed["tool_name"]
            
            if tool_name not in registry.missing_tools:
                raise ValueError(
                    f"LLM suggested updating '{tool_name}' but it doesn't exist in missing_tools"
                )
            
            existing_tool = registry.missing_tools[tool_name]
            reasoning = parsed.get("reasoning", "No reasoning provided")
            logger.debug(f"Update reasoning: {reasoning}")
            
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
            reasoning = parsed.get("reasoning", "No reasoning provided")
            logger.debug(f"Create new reasoning: {reasoning}")
            
            new_tool = HypotheticalTool(
                tool_name=parsed["tool_name"],
                description=parsed["description"],
                parameters=[ToolParameter(**p) for p in parsed["parameters"]],
                return_type=parsed["return_type"],
                category=parsed["category"]
            )
            # Store semantic_mapping and example_usage for THIS query in a tuple to be added to context
            new_tool._query_mapping = {
                "semantic_mapping": parsed.get("semantic_mapping_for_this_query", ""),
                "example_usage": parsed.get("example_usage_for_this_query", "")
            }
            logger.info(f"Creating new missing tool: {new_tool.tool_name}")
            return new_tool, True, False  # Is new, not updated
        
        else:
            raise ValueError(f"Unknown action in LLM response: {parsed.get('action')}")


def discover_missing_tool(
    user_utterance: str,
    registry: ToolRegistry
) -> Tuple[HypotheticalTool, bool, bool]:
    """
    Main entry point for tool discovery.
    
    Analyzes a user query that the agent couldn't answer and determines
    what tool would be needed.
    
    Args:
        user_utterance: The user's question
        registry: ToolRegistry with current tool state
        
    Returns:
        Tuple of (tool_spec, is_new, is_update) where:
            - tool_spec is the HypotheticalTool
            - is_new is True if this is a newly created tool
            - is_update is True if this is an updated existing tool
        
    Raises:
        Exception: If analysis fails
    """
    analyzer = ToolAnalyzer()
    return analyzer.analyze(user_utterance, registry)


