"""
Tool Registry Refiner - Post-processing layer for tool discovery

This module refines a tool registry by merging overly specific tools and splitting
overly broad tools. It operates domain-agnostically using LLM-based analysis.

Key Features:
- Merges tools with same action + semantic return type
- Ignores entity-type, time-period, and metric specificity
- Rewrites semantic mappings for merged tools
- Preserves all conversation contexts and version history
- Standardizes parameters across similar tools
- Detects and flags overly broad tools

Usage:
    python3 tool_registry_refiner.py
    
    Reads: tool_registry.json
    Writes: tool_registry_refined.json
"""

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from loguru import logger
from openai import AzureOpenAI

from tool_discovery import HypotheticalTool, ToolParameter, ToolRegistry


# Load environment variables
ENV_PATH = Path(__file__).parent.parent.parent.parent.parent / ".env"
load_dotenv(ENV_PATH)

# Constants
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_TEMPERATURE = 0.0
MERGE_CONFIDENCE_THRESHOLD = 0.9
REGISTRY_INPUT = "tool_registry.json"
REGISTRY_OUTPUT = "tool_registry_refined.json"


@dataclass
class SemanticReturnType:
    """Semantic interpretation of a tool's return type"""
    python_type: str  # e.g., "Dict", "List", "str"
    semantic_meaning: str  # e.g., "comparison_metrics", "text_explanation"
    original: str  # Original return_type string


@dataclass
class ToolCluster:
    """A cluster of tools that potentially should be merged"""
    action: str  # e.g., "compare", "explain"
    return_semantic: str  # Semantic meaning of return type
    tools: List[HypotheticalTool]
    
    def __len__(self):
        return len(self.tools)


@dataclass
class MergeDecision:
    """Decision about whether and how to merge a cluster of tools"""
    should_merge: bool
    confidence: float
    merged_tool_name: str
    merged_description: str
    merged_parameters: List[Dict]
    merged_return_type: str
    rationale: str
    tool_names_to_merge: List[str]


@dataclass
class RewrittenContext:
    """A conversation context with updated semantic mapping"""
    user_utterance: str
    timestamp: str
    hallucinated_response: str
    turn_number: int
    allow_hallucination: bool
    investor_profile: str
    user_risk_profile: str
    semantic_mapping: str  # Updated to reference new tool
    example_usage: str  # Updated to reference new tool
    is_new_discovery: Optional[bool] = None
    is_update: Optional[bool] = None


class ReturnTypeAnalyzer:
    """Analyzes tool return types to extract semantic meaning"""
    
    def __init__(self, client: AzureOpenAI, model: str = DEFAULT_MODEL):
        self.client = client
        self.model = model
    
    def analyze_batch(self, tools: List[HypotheticalTool]) -> Dict[str, SemanticReturnType]:
        """
        Analyze return types for a batch of tools.
        
        Args:
            tools: List of tools to analyze
            
        Returns:
            Dict mapping tool_name to SemanticReturnType
        """
        if not tools:
            return {}
        
        logger.info(f"Analyzing return types for {len(tools)} tools")
        
        prompt = self._build_analysis_prompt(tools)
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Analyze the return types and extract semantic meanings."}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=DEFAULT_TEMPERATURE,
                response_format={"type": "json_object"}
            )
        except Exception as e:
            logger.error(f"Return type analysis failed: {e}")
            raise
        
        response_text = response.choices[0].message.content.strip()
        
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse return type analysis: {response_text}")
            raise
        
        results = {}
        for tool_name, analysis in parsed.get("tools", {}).items():
            results[tool_name] = SemanticReturnType(
                python_type=analysis["python_type"],
                semantic_meaning=analysis["semantic_meaning"],
                original=analysis["original"]
            )
        
        return results
    
    def _build_analysis_prompt(self, tools: List[HypotheticalTool]) -> str:
        """Build prompt for return type analysis"""
        
        tools_text = "\n".join([
            f"- {tool.tool_name}: {tool.return_type}"
            for tool in tools
        ])
        
        return f"""You are analyzing return types of API functions to extract their semantic meaning.

TOOLS TO ANALYZE:
{tools_text}

TASK:
For each tool, extract:
1. Python type (Dict, List, str, int, float, bool, etc.)
2. Semantic meaning (what the data represents conceptually)

SEMANTIC MEANINGS (examples, not exhaustive):
- comparison_metrics: Numeric comparison data
- text_explanation: Human-readable explanation
- ranked_list: Ordered list of items
- entity_details: Information about specific entities
- allocation_specification: How to distribute resources
- performance_data: Historical or current performance metrics
- recommendation: Suggested action or choice
- calculation_result: Computed numeric value
- validation_result: Success/failure status

Return JSON in this format:
{{
  "tools": {{
    "tool_name_1": {{
      "python_type": "Dict",
      "semantic_meaning": "comparison_metrics",
      "original": "Dict[str, Any] (performance metrics for each fund)"
    }},
    "tool_name_2": {{
      "python_type": "str",
      "semantic_meaning": "text_explanation",
      "original": "str (text explanation)"
    }}
  }}
}}

IMPORTANT:
- Focus on WHAT the data represents, not HOW it's structured
- Use general categories, not domain-specific terms
- Be consistent with semantic meanings across similar tools
- Return valid JSON only"""


class ToolClusterer:
    """Groups tools by action and semantic return type"""
    
    def __init__(self, return_type_analyzer: ReturnTypeAnalyzer):
        self.return_type_analyzer = return_type_analyzer
    
    def cluster_tools(self, tools: Dict[str, HypotheticalTool]) -> List[ToolCluster]:
        """
        Cluster tools by (action + semantic return type).
        
        Args:
            tools: Dict of tool_name -> HypotheticalTool
            
        Returns:
            List of ToolCluster objects
        """
        if not tools:
            return []
        
        logger.info(f"Clustering {len(tools)} tools")
        
        # Analyze return types
        return_types = self.return_type_analyzer.analyze_batch(list(tools.values()))
        
        # Group by (action, semantic_return)
        clusters_dict = defaultdict(list)
        
        for tool_name, tool in tools.items():
            action = self._extract_action(tool.tool_name, tool.description)
            
            return_semantic = "unknown"
            if tool_name in return_types:
                return_semantic = return_types[tool_name].semantic_meaning
            
            key = (action, return_semantic)
            clusters_dict[key].append(tool)
        
        # Convert to ToolCluster objects
        clusters = []
        for (action, return_semantic), tool_list in clusters_dict.items():
            if len(tool_list) >= 1:  # Include all clusters, even single tools
                clusters.append(ToolCluster(
                    action=action,
                    return_semantic=return_semantic,
                    tools=tool_list
                ))
        
        logger.info(f"Created {len(clusters)} clusters")
        return clusters
    
    @staticmethod
    def _extract_action(tool_name: str, description: str) -> str:
        """Extract primary action verb from tool name or description"""
        
        action_verbs = [
            "compare", "calculate", "compute", "explain", "justify",
            "evaluate", "analyze", "retrieve", "fetch", "get",
            "list", "filter", "search", "find", "rank",
            "project", "predict", "forecast", "estimate", "build",
            "create", "generate", "construct"
        ]
        
        tool_name_lower = tool_name.lower()
        for verb in action_verbs:
            if tool_name_lower.startswith(verb):
                return verb
        
        description_lower = description.lower()
        for verb in action_verbs:
            if verb in description_lower:
                return verb
        
        return "misc"


class MergeAnalyzer:
    """Analyzes clusters to determine merge decisions"""
    
    def __init__(self, client: AzureOpenAI, model: str = DEFAULT_MODEL):
        self.client = client
        self.model = model
    
    def analyze_cluster(self, cluster: ToolCluster) -> Optional[MergeDecision]:
        """
        Analyze a cluster to determine if tools should merge.
        
        Args:
            cluster: ToolCluster to analyze
            
        Returns:
            MergeDecision if merge recommended, None otherwise
        """
        if len(cluster) < 2:
            return None
        
        logger.info(f"Analyzing cluster: {cluster.action}/{cluster.return_semantic} ({len(cluster)} tools)")
        
        prompt = self._build_merge_prompt(cluster)
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Analyze whether these tools should be merged and provide your recommendation."}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=DEFAULT_TEMPERATURE,
                response_format={"type": "json_object"}
            )
        except Exception as e:
            logger.error(f"Merge analysis failed for cluster {cluster.action}: {e}")
            raise
        
        response_text = response.choices[0].message.content.strip()
        
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse merge analysis: {response_text}")
            raise
        
        should_merge = parsed.get("should_merge", False)
        confidence = parsed.get("confidence", 0.0)
        
        if not should_merge or confidence < MERGE_CONFIDENCE_THRESHOLD:
            logger.info(f"No merge: confidence={confidence:.2f} (threshold={MERGE_CONFIDENCE_THRESHOLD})")
            return None
        
        decision = MergeDecision(
            should_merge=True,
            confidence=confidence,
            merged_tool_name=parsed["merged_tool_name"],
            merged_description=parsed["merged_description"],
            merged_parameters=parsed["merged_parameters"],
            merged_return_type=parsed["merged_return_type"],
            rationale=parsed["rationale"],
            tool_names_to_merge=[t.tool_name for t in cluster.tools]
        )
        
        logger.info(f"Merge recommended: {decision.merged_tool_name} (confidence={confidence:.2f})")
        return decision
    
    def _build_merge_prompt(self, cluster: ToolCluster) -> str:
        """Build prompt for merge analysis"""
        
        tools_text = ""
        for i, tool in enumerate(cluster.tools, 1):
            params = ", ".join([f"{p.name}: {p.type}" for p in tool.parameters])
            tools_text += f"\n{i}. {tool.tool_name}({params}) -> {tool.return_type}\n"
            tools_text += f"   Description: {tool.description}\n"
            tools_text += f"   Frequency: {tool.frequency}\n"
            tools_text += f"   Contexts: {len(tool.conversation_contexts)}\n"
        
        return f"""You are analyzing whether multiple API functions should be merged into one.

CLUSTER INFORMATION:
Action Type: {cluster.action}
Return Semantic: {cluster.return_semantic}
Number of Tools: {len(cluster.tools)}

TOOLS IN CLUSTER:
{tools_text}

DOMAIN-AGNOSTIC MERGE CRITERIA:

Tools should be MERGED if ALL of the following are true:
1. Same PRIMARY ACTION (same verb: compare, explain, calculate, etc.)
2. Same SEMANTIC RETURN TYPE (both return metrics, both return explanations, etc.)
3. Subject entities are similar enough to be parameterized (e.g., all measurable entities, all comparable items)
4. Differences are only in parameter values, not fundamental functionality

IGNORE these differences when merging:
- Entity subtypes (fund vs portfolio vs bond → all "investments" or "entities")
- Time periods (10yr vs 5yr → parameterize with dates)
- Specific metrics (returns vs yields → parameterize with metrics list)
- Specific amounts or names in descriptions

AVOID merging if:
- Tools perform fundamentally different computations
- Return types have incompatible structures or meanings
- Tools would need complex conditional logic to handle both cases
- Merging would create an overly broad "do-everything" function

DECISION:
Analyze these tools and decide if they should merge. Be aggressive about merging similar functionality.

Return JSON in this format:
{{
  "should_merge": true|false,
  "confidence": 0.0-1.0,
  "merged_tool_name": "generalized_name_in_snake_case",
  "merged_description": "description covering all use cases",
  "merged_parameters": [
    {{
      "name": "param_name",
      "type": "python_type",
      "description": "what this parameter does",
      "required": true|false
    }}
  ],
  "merged_return_type": "return type with semantic meaning",
  "rationale": "why these should or should not be merged"
}}

IMPORTANT:
- Confidence should reflect how certain you are about the merge decision
- Merged tool name should be generic and action-focused
- Include all necessary parameters to handle all use cases
- Mark parameters as required=false if they're for filtering/optional features
- Be specific in your rationale
- Return valid JSON only"""


class SemanticMappingRewriter:
    """Rewrites semantic mappings for merged tools"""
    
    def __init__(self, client: AzureOpenAI, model: str = DEFAULT_MODEL):
        self.client = client
        self.model = model
    
    def rewrite_context(
        self,
        context: Dict,
        old_tool: HypotheticalTool,
        new_tool: HypotheticalTool
    ) -> RewrittenContext:
        """
        Rewrite a single conversation context for a merged tool.
        
        Args:
            context: Original conversation context dict
            old_tool: Original tool specification
            new_tool: Merged tool specification
            
        Returns:
            RewrittenContext with updated semantic_mapping and example_usage
        """
        prompt = self._build_rewrite_prompt(context, old_tool, new_tool)
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Rewrite the semantic mapping and example usage for the new merged tool."}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=DEFAULT_TEMPERATURE,
                response_format={"type": "json_object"}
            )
        except Exception as e:
            logger.error(f"Context rewriting failed: {e}")
            raise
        
        response_text = response.choices[0].message.content.strip()
        
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse rewrite response: {response_text}")
            raise
        
        return RewrittenContext(
            user_utterance=context["user_utterance"],
            timestamp=context["timestamp"],
            hallucinated_response=context["hallucinated_response"],
            turn_number=context["turn_number"],
            allow_hallucination=context["allow_hallucination"],
            investor_profile=context.get("investor_profile", ""),
            user_risk_profile=context.get("user_risk_profile", ""),
            semantic_mapping=parsed["semantic_mapping"],
            example_usage=parsed["example_usage"],
            is_new_discovery=context.get("is_new_discovery"),
            is_update=context.get("is_update")
        )
    
    def _build_rewrite_prompt(
        self,
        context: Dict,
        old_tool: HypotheticalTool,
        new_tool: HypotheticalTool
    ) -> str:
        """Build prompt for semantic mapping rewrite"""
        
        old_params = ", ".join([f"{p.name}: {p.type}" for p in old_tool.parameters])
        new_params = ", ".join([f"{p.name}: {p.type}" for p in new_tool.parameters])
        
        old_mapping = context.get("semantic_mapping", "")
        old_usage = context.get("example_usage", "")
        
        return f"""You are rewriting semantic mappings and example usage after tools have been merged.

ORIGINAL TOOL:
{old_tool.tool_name}({old_params}) -> {old_tool.return_type}
Description: {old_tool.description}

NEW MERGED TOOL:
{new_tool.tool_name}({new_params}) -> {new_tool.return_type}
Description: {new_tool.description}

USER UTTERANCE:
"{context['user_utterance']}"

ORIGINAL SEMANTIC MAPPING:
{old_mapping}

ORIGINAL EXAMPLE USAGE:
{old_usage}

TASK:
Rewrite the semantic_mapping and example_usage to reference the NEW merged tool.

GUIDELINES:
- semantic_mapping should explain how THIS SPECIFIC user query maps to the new tool's parameters
- Extract parameter values from the user utterance
- Explain what the tool should return for this specific query
- example_usage should show the actual function call with values from this query
- Use the new tool name and parameter names
- If new tool has additional parameters, set them appropriately or to None/default

Return JSON in this format:
{{
  "semantic_mapping": "Detailed explanation of how THIS query maps to new tool parameters and expected return",
  "example_usage": "new_tool_name(param1=value1, param2=value2, ...)"
}}

IMPORTANT:
- Be specific to this user utterance
- Use actual values from the query, not placeholders
- Ensure example_usage is syntactically valid
- Return valid JSON only"""


class ParameterStandardizer:
    """Standardizes parameter names and types across merged tools"""
    
    @staticmethod
    def standardize_parameters(parameters: List[Dict]) -> List[ToolParameter]:
        """
        Standardize parameter names and types.
        
        Args:
            parameters: List of parameter dicts
            
        Returns:
            List of standardized ToolParameter objects
        """
        standardized = []
        
        # Common standardization rules
        name_mappings = {
            "time_period": "time_period",
            "years": "time_period",
            "duration": "time_period",
            "entities": "entities",
            "items": "entities",
            "targets": "entities",
            "start_date": "start_date",
            "begin_date": "start_date",
            "end_date": "end_date",
            "finish_date": "end_date",
        }
        
        seen_names = set()
        
        for param in parameters:
            name = param["name"]
            param_type = param["type"]
            description = param["description"]
            required = param.get("required", True)
            
            # Apply name standardization
            standardized_name = name_mappings.get(name, name)
            
            # Avoid duplicates
            if standardized_name in seen_names:
                continue
            
            seen_names.add(standardized_name)
            
            standardized.append(ToolParameter(
                name=standardized_name,
                type=param_type,
                description=description,
                required=required
            ))
        
        return standardized


class BroadToolDetector:
    """Detects overly broad tools that try to do too many things"""
    
    @staticmethod
    def detect_broad_tools(tools: List[HypotheticalTool]) -> List[Tuple[str, str]]:
        """
        Detect tools that are overly broad.
        
        Args:
            tools: List of tools to check
            
        Returns:
            List of (tool_name, reason) tuples for broad tools
        """
        broad_tools = []
        
        action_verbs = [
            "compare", "calculate", "compute", "explain", "justify",
            "evaluate", "analyze", "retrieve", "fetch", "get",
            "list", "filter", "search", "find", "rank",
            "project", "predict", "forecast", "estimate", "build"
        ]
        
        for tool in tools:
            description_lower = tool.description.lower()
            
            # Count how many different actions are mentioned
            actions_found = [verb for verb in action_verbs if verb in description_lower]
            
            if len(actions_found) >= 3:
                reason = f"Mentions multiple actions: {', '.join(actions_found[:3])}"
                broad_tools.append((tool.tool_name, reason))
                continue
            
            # Check for "action" parameter that switches behavior
            has_action_param = any(
                p.name.lower() in ["action", "operation", "mode", "type"]
                for p in tool.parameters
            )
            
            if has_action_param:
                reason = "Has parameter that switches between different behaviors"
                broad_tools.append((tool.tool_name, reason))
                continue
        
        return broad_tools


class RegistryRefiner:
    """Main orchestrator for tool registry refinement"""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.client = AzureOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            api_version=os.getenv("LLM_API_VERSION"),
            azure_endpoint=os.getenv("LLM_API_ENDPOINT"),
        )
        
        self.return_type_analyzer = ReturnTypeAnalyzer(self.client, model)
        self.tool_clusterer = ToolClusterer(self.return_type_analyzer)
        self.merge_analyzer = MergeAnalyzer(self.client, model)
        self.semantic_rewriter = SemanticMappingRewriter(self.client, model)
    
    def refine_registry(self, registry: ToolRegistry) -> ToolRegistry:
        """
        Refine a tool registry by merging similar tools.
        
        Args:
            registry: Input ToolRegistry
            
        Returns:
            Refined ToolRegistry
        """
        logger.info("Starting tool registry refinement")
        
        # Step 1: Cluster tools
        clusters = self.tool_clusterer.cluster_tools(registry.missing_tools)
        
        # Step 2: Analyze clusters for merges
        merge_decisions = []
        for cluster in clusters:
            decision = self.merge_analyzer.analyze_cluster(cluster)
            if decision:
                merge_decisions.append(decision)
        
        logger.info(f"Found {len(merge_decisions)} clusters to merge")
        
        # Step 3: Apply merges
        refined_tools = self._apply_merges(registry.missing_tools, merge_decisions)
        
        # Step 4: Detect broad tools
        broad_tools = BroadToolDetector.detect_broad_tools(list(refined_tools.values()))
        if broad_tools:
            logger.warning(f"Detected {len(broad_tools)} overly broad tools:")
            for tool_name, reason in broad_tools:
                logger.warning(f"  - {tool_name}: {reason}")
        
        # Step 5: Create refined registry
        refined_registry = ToolRegistry(
            existing_apis=registry.existing_apis,
            existing_kb_tables=registry.existing_kb_tables,
            missing_tools=refined_tools,
            data_constraints=registry.data_constraints,
            out_of_scope_queries=registry.out_of_scope_queries,
            ambiguous_requests=registry.ambiguous_requests,
            registry_version=registry.registry_version,
            last_updated=datetime.now().isoformat()
        )
        
        logger.info(f"Refinement complete: {len(registry.missing_tools)} -> {len(refined_tools)} tools")
        
        return refined_registry
    
    def _apply_merges(
        self,
        original_tools: Dict[str, HypotheticalTool],
        merge_decisions: List[MergeDecision]
    ) -> Dict[str, HypotheticalTool]:
        """Apply merge decisions to create refined tool set"""
        
        # Track which tools have been merged
        merged_tool_names = set()
        for decision in merge_decisions:
            merged_tool_names.update(decision.tool_names_to_merge)
        
        refined_tools = {}
        
        # Keep tools that weren't merged
        for name, tool in original_tools.items():
            if name not in merged_tool_names:
                refined_tools[name] = tool
        
        # Create merged tools
        for decision in merge_decisions:
            merged_tool = self._create_merged_tool(decision, original_tools)
            refined_tools[merged_tool.tool_name] = merged_tool
        
        return refined_tools
    
    def _create_merged_tool(
        self,
        decision: MergeDecision,
        original_tools: Dict[str, HypotheticalTool]
    ) -> HypotheticalTool:
        """Create a merged tool from a merge decision"""
        
        logger.info(f"Creating merged tool: {decision.merged_tool_name}")
        
        # Standardize parameters
        standardized_params = ParameterStandardizer.standardize_parameters(
            decision.merged_parameters
        )
        
        # Collect all conversation contexts and rewrite them
        all_contexts = []
        total_frequency = 0
        all_version_history = []
        first_seen = None
        last_seen = None
        
        for tool_name in decision.tool_names_to_merge:
            if tool_name not in original_tools:
                continue
            
            old_tool = original_tools[tool_name]
            total_frequency += old_tool.frequency
            
            # Track first/last seen
            if not first_seen or old_tool.first_seen < first_seen:
                first_seen = old_tool.first_seen
            if not last_seen or old_tool.last_seen > last_seen:
                last_seen = old_tool.last_seen
            
            # Preserve version history
            all_version_history.extend(old_tool.version_history)
            
            # Create new merged tool for rewriting (temporary)
            new_tool = HypotheticalTool(
                tool_name=decision.merged_tool_name,
                description=decision.merged_description,
                parameters=standardized_params,
                return_type=decision.merged_return_type,
                category=old_tool.category
            )
            
            # Rewrite each context
            for context in old_tool.conversation_contexts:
                logger.debug(f"Rewriting context for {tool_name}")
                rewritten = self.semantic_rewriter.rewrite_context(
                    context, old_tool, new_tool
                )
                
                # Convert back to dict
                context_dict = {
                    "user_utterance": rewritten.user_utterance,
                    "timestamp": rewritten.timestamp,
                    "hallucinated_response": rewritten.hallucinated_response,
                    "turn_number": rewritten.turn_number,
                    "allow_hallucination": rewritten.allow_hallucination,
                    "investor_profile": rewritten.investor_profile,
                    "user_risk_profile": rewritten.user_risk_profile,
                    "semantic_mapping": rewritten.semantic_mapping,
                    "example_usage": rewritten.example_usage
                }
                
                if rewritten.is_new_discovery is not None:
                    context_dict["is_new_discovery"] = rewritten.is_new_discovery
                if rewritten.is_update is not None:
                    context_dict["is_update"] = rewritten.is_update
                
                all_contexts.append(context_dict)
        
        # Add merge entry to version history
        merge_entry = {
            "version": 1,
            "timestamp": datetime.now().isoformat(),
            "change_type": "merged_from_refinement",
            "change_description": f"Merged from: {', '.join(decision.tool_names_to_merge)}",
            "merged_tool_names": decision.tool_names_to_merge,
            "merge_rationale": decision.rationale,
            "confidence": decision.confidence,
            "parameters_snapshot": [asdict(p) for p in standardized_params],
            "description_snapshot": decision.merged_description
        }
        all_version_history.append(merge_entry)
        
        # Determine category (use most common from merged tools)
        categories = [original_tools[name].category for name in decision.tool_names_to_merge if name in original_tools]
        category = max(set(categories), key=categories.count) if categories else "api_function"
        
        # Create merged tool
        merged_tool = HypotheticalTool(
            tool_name=decision.merged_tool_name,
            description=decision.merged_description,
            parameters=standardized_params,
            return_type=decision.merged_return_type,
            category=category,
            frequency=total_frequency,
            first_seen=first_seen or datetime.now().isoformat(),
            last_seen=last_seen or datetime.now().isoformat(),
            conversation_contexts=all_contexts,
            version=1,
            version_history=all_version_history
        )
        
        logger.info(
            f"Merged {len(decision.tool_names_to_merge)} tools into "
            f"{merged_tool.tool_name} (freq={total_frequency}, contexts={len(all_contexts)})"
        )
        
        return merged_tool


def main():
    """Main entry point for tool registry refinement"""
    
    logger.info("=" * 80)
    logger.info("Tool Registry Refiner - Starting")
    logger.info("=" * 80)
    
    # Paths
    script_dir = Path(__file__).parent
    input_path = script_dir / REGISTRY_INPUT
    output_path = script_dir / REGISTRY_OUTPUT
    
    # Load registry
    if not input_path.exists():
        logger.error(f"Input registry not found: {input_path}")
        return
    
    logger.info(f"Loading registry from: {input_path}")
    registry = ToolRegistry.load_or_create(input_path)
    
    original_count = len(registry.missing_tools)
    logger.info(f"Loaded {original_count} tools to refine")
    
    # Run refinement
    refiner = RegistryRefiner()
    refined_registry = refiner.refine_registry(registry)
    
    refined_count = len(refined_registry.missing_tools)
    logger.info(f"Refined: {original_count} -> {refined_count} tools ({original_count - refined_count} merged)")
    
    # Save refined registry
    logger.info(f"Saving refined registry to: {output_path}")
    refined_registry.save_to_disk(output_path)
    
    logger.info("=" * 80)
    logger.info("Tool Registry Refiner - Complete")
    logger.info("=" * 80)
    logger.info(f"Output: {output_path}")
    
    # Summary statistics
    logger.info("\nSummary:")
    logger.info(f"  Original tools: {original_count}")
    logger.info(f"  Refined tools: {refined_count}")
    logger.info(f"  Tools merged: {original_count - refined_count}")
    logger.info(f"  Merge rate: {(original_count - refined_count) / original_count * 100:.1f}%")
    
    # Show refined tools by action
    action_counts = defaultdict(int)
    for tool in refined_registry.missing_tools.values():
        action = ToolClusterer._extract_action(tool.tool_name, tool.description)
        action_counts[action] += 1
    
    logger.info("\nTools by action:")
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {action}: {count}")


if __name__ == "__main__":
    main()

