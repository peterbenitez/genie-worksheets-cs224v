"""
Run tool discovery experiments across all domains.
This script auto-discovers domains and generates realistic user queries
to identify missing capabilities. No hardcoded domain configurations.
"""

import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import get_openai_client, discover_domains

client = get_openai_client()


def load_domain_config(domain_path: Path) -> Dict[str, Any]:
    """Load domain configuration from config.yaml and infer from directory structure."""
    config = {
        "path": str(domain_path / "hallucinations"),
        "description": "",
        "existing_capabilities": [],
        "schema_tables": [],
        "persona_types": []
    }

    # Try to load config.yaml
    config_path = domain_path / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                config.update(yaml_config)

    # Parse schema tables from table_schema.txt if exists
    schema_path = domain_path / "table_schema.txt"
    if schema_path.exists():
        config["schema_tables"] = parse_schema_tables(schema_path)

    # Use LLM to infer domain description and capabilities if not set
    if not config.get("description") or not config.get("existing_capabilities"):
        inferred = infer_domain_config(domain_path)
        if not config.get("description"):
            config["description"] = inferred.get("description", f"{domain_path.name} assistant")
        if not config.get("existing_capabilities"):
            config["existing_capabilities"] = inferred.get("capabilities", [])
        if not config.get("persona_types"):
            config["persona_types"] = inferred.get("personas", [])

    return config


def parse_schema_tables(schema_path: Path) -> List[str]:
    """Extract table names from table_schema.txt."""
    import re
    tables = []
    content = schema_path.read_text()
    for match in re.finditer(r'CREATE TABLE (\w+)', content, re.IGNORECASE):
        tables.append(match.group(1))
    return tables


def infer_domain_config(domain_path: Path) -> Dict[str, Any]:
    """Use LLM to infer domain configuration from directory name and files."""
    domain_name = domain_path.name.replace("_agent", "").replace("_", " ")

    prompt = f"""Given a domain called "{domain_name}", infer:
1. A brief description of what this agent does
2. 3-5 typical capabilities it might have
3. 3-5 persona types that would use it

Respond in JSON:
{{
    "description": "brief description",
    "capabilities": ["cap1", "cap2", "cap3"],
    "personas": ["persona1", "persona2", "persona3"]
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except:
        return {
            "description": f"{domain_name} assistant",
            "capabilities": ["Search and retrieve information", "Answer questions", "Provide recommendations"],
            "personas": ["new user", "experienced user", "curious user"]
        }


def generate_queries_for_domain(domain: str, config: Dict, num_queries: int = 30) -> List[str]:
    """Generate realistic user queries that would trigger hallucinations."""

    prompt = f"""You are simulating users interacting with a {config.get('description', domain + ' assistant')}.

The system currently has these capabilities:
{chr(10).join(f"- {cap}" for cap in config.get('existing_capabilities', ['Basic search', 'Information retrieval']))}

Database tables available: {', '.join(config.get('schema_tables', ['data']))}

Generate {num_queries} realistic user queries that would likely EXCEED these capabilities
and cause the agent to hallucinate. Focus on queries asking for:
1. Explanations of concepts (why, what, how questions)
2. Personalized recommendations/suggestions
3. Comparisons between options
4. Calculations or predictions
5. Information not in the database

Vary the queries across these persona types:
{chr(10).join(f"- {persona}" for persona in config.get('persona_types', ['curious user', 'confused user', 'demanding user']))}

Output ONLY a JSON array of query strings, nothing else. Example format:
["Query 1", "Query 2", "Query 3"]
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=2000,
    )

    content = response.choices[0].message.content.strip()

    try:
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        queries = json.loads(content)
        return queries
    except json.JSONDecodeError:
        print(f"Failed to parse queries for {domain}")
        raise


def analyze_query(query: str, domain: str, config: Dict, registry_path: Path) -> Dict:
    """Analyze a single query to discover missing tools."""

    with open(registry_path) as f:
        registry = json.load(f)

    prompt = f"""Analyze this user query for a {config.get('description', domain + ' assistant')}:

Query: "{query}"

Current capabilities:
{chr(10).join(f"- {cap}" for cap in config.get('existing_capabilities', []))}

Database tables: {', '.join(config.get('schema_tables', []))}

This query likely CANNOT be answered with existing capabilities. Determine:
1. What type of tool/capability is missing?
2. What action type is this? (explain, suggest, compare, list, calculate, find)
3. What would the tool be called?
4. What parameters would it need?

Respond in JSON format:
{{
    "needs_new_tool": true/false,
    "action_type": "explain|suggest|compare|list|calculate|find",
    "tool_name": "suggested_tool_name",
    "description": "what this tool would do",
    "parameters": [{{"name": "param_name", "type": "str|int|float", "required": true/false}}],
    "example_usage": "tool_name(param='value')"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500,
    )

    content = response.choices[0].message.content.strip()

    try:
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        analysis = json.loads(content)
        analysis["query"] = query
        analysis["timestamp"] = datetime.now().isoformat()
        return analysis
    except:
        return {
            "query": query,
            "needs_new_tool": True,
            "action_type": "unknown",
            "tool_name": "unknown_tool",
            "error": "Failed to parse analysis"
        }


def update_registry(registry_path: Path, analyses: List[Dict], domain: str):
    """Update the tool registry with discovered tools."""

    with open(registry_path) as f:
        registry = json.load(f)

    tool_counts = {}
    tool_specs = {}

    for analysis in analyses:
        if not analysis.get("needs_new_tool", False):
            continue

        tool_name = analysis.get("tool_name", "unknown")
        action_type = analysis.get("action_type", "misc")

        if tool_name not in tool_counts:
            tool_counts[tool_name] = 0
            tool_specs[tool_name] = {
                "tool_name": tool_name,
                "description": analysis.get("description", ""),
                "parameters": analysis.get("parameters", []),
                "return_type": "str",
                "category": "api_function",
                "frequency": 0,
                "first_seen": analysis.get("timestamp", datetime.now().isoformat()),
                "last_seen": analysis.get("timestamp", datetime.now().isoformat()),
                "conversation_contexts": [],
                "action_type": action_type
            }

        tool_counts[tool_name] += 1
        tool_specs[tool_name]["frequency"] = tool_counts[tool_name]
        tool_specs[tool_name]["last_seen"] = analysis.get("timestamp", datetime.now().isoformat())
        tool_specs[tool_name]["conversation_contexts"].append({
            "query": analysis.get("query", ""),
            "example_usage": analysis.get("example_usage", ""),
            "timestamp": analysis.get("timestamp", "")
        })

    for tool_name, spec in tool_specs.items():
        action_type = spec.pop("action_type", "misc")

        if action_type not in registry["missing_tools_by_action"]:
            registry["missing_tools_by_action"][action_type] = {}

        registry["missing_tools_by_action"][action_type][tool_name] = spec

    registry["last_updated"] = datetime.now().isoformat()

    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)

    return tool_counts


def run_domain_experiment(domain: str, domain_path: Path, num_queries: int = 30):
    """Run full experiment for a single domain."""

    print(f"\n{'='*60}")
    print(f"Running experiment for: {domain.upper()}")
    print(f"{'='*60}")

    config = load_domain_config(domain_path)
    registry_path = domain_path / "hallucinations" / "tool_registry.json"

    print(f"\n📝 Generating {num_queries} test queries...")
    queries = generate_queries_for_domain(domain, config, num_queries)
    print(f"   Generated {len(queries)} queries")

    print(f"\n🔍 Analyzing queries for missing tools...")
    analyses = []
    for i, query in enumerate(queries):
        print(f"   [{i+1}/{len(queries)}] {query[:50]}...")
        analysis = analyze_query(query, domain, config, registry_path)
        analyses.append(analysis)

    print(f"\n💾 Updating tool registry...")
    tool_counts = update_registry(registry_path, analyses, domain)

    print(f"\n📊 Results for {domain}:")
    print(f"   Total queries: {len(queries)}")
    print(f"   Unique tools discovered: {len(tool_counts)}")
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"   - {tool}: {count} instances")

    return {
        "domain": domain,
        "queries": len(queries),
        "tools": len(tool_counts),
        "tool_counts": tool_counts
    }


def run_refiner_for_domain(domain: str, domain_path: Path) -> Dict[str, Any]:
    """Run the tool registry refiner for a domain."""
    hallucinations_path = domain_path / "hallucinations"

    # Add to path and import modules
    sys.path.insert(0, str(hallucinations_path))

    # Clear cached modules to ensure fresh import
    for mod in ['tool_registry_refiner', 'tool_discovery']:
        if mod in sys.modules:
            del sys.modules[mod]

    import tool_registry_refiner
    import tool_discovery

    registry_path = hallucinations_path / "tool_registry.json"
    output_path = hallucinations_path / "tool_registry_refined.json"

    if not registry_path.exists():
        return {"error": "No tool_registry.json found"}

    # Load the registry
    registry = tool_discovery.ToolRegistry.load_or_create(registry_path)
    original_count = len(registry.missing_tools)

    # Run the refiner
    refiner = tool_registry_refiner.ToolRegistryRefiner()
    refined_registry = refiner.refine_registry(registry)

    # Save refined registry
    refined_registry.save_to_disk(output_path)

    return {
        "original_tools": original_count,
        "refined_tools": len(refined_registry.missing_tools),
        "output_path": str(output_path)
    }


def main():
    """Run experiments across all auto-discovered domains."""

    print("=" * 60)
    print("GenieWorksheet Wizard - Multi-Domain Experiment Runner")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    base_path = Path(__file__).parent

    # Auto-discover domains
    domains = discover_domains(base_path)
    print(f"\nDiscovered {len(domains)} domains: {', '.join(domains.keys())}")

    results = {}

    for domain, domain_path in domains.items():
        try:
            result = run_domain_experiment(domain, domain_path, num_queries=30)
            results[domain] = result
        except Exception as e:
            print(f"Error in {domain}: {e}")
            results[domain] = {"error": str(e)}

    # Run refiner for each domain
    print("\n" + "=" * 60)
    print("RUNNING TOOL REGISTRY REFINER")
    print("=" * 60)

    for domain, domain_path in domains.items():
        print(f"\nRefining {domain}...")
        try:
            refine_result = run_refiner_for_domain(domain, domain_path)
            if "error" not in refine_result:
                print(f"  Refined: {refine_result.get('original_tools', '?')} -> {refine_result['refined_tools']} tools")
                results[domain]["refiner"] = refine_result
            else:
                print(f"  Skipped: {refine_result['error']}")
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    total_tools = 0
    total_queries = 0

    for domain, result in results.items():
        if "error" not in result:
            print(f"\n{domain.upper()}:")
            print(f"  Tools: {result['tools']}")
            print(f"  Gap instances: {sum(result['tool_counts'].values())}")
            if "refiner" in result:
                print(f"  Refined to: {result['refiner']['refined_tools']} tools")
            total_tools += result['tools']
            total_queries += sum(result['tool_counts'].values())

    print(f"\nTOTAL: {total_tools} tools, {total_queries} gap instances")


if __name__ == "__main__":
    main()
