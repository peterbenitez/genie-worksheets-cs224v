# GenieWorksheet Wizard - Hallucination Analysis Framework

Discovers missing tools by forcing agent hallucinations on out-of-scope queries, then analyzes traces to infer missing APIs or database columns/schemas.

## Pipeline

```
User Personas -> Simulated Queries (30/domain) -> Force Hallucination -> Schema Check -> LLM Analysis -> Tool Discovery (unlimited) -> Refinement (MAX_TOOLS=5)
```

## Setup

```bash
cd genie-worksheets-cs224v
source .venv/bin/activate
pip install -e .
pip install psycopg2-binary
```

Requires `.env` file:
```
LLM_API_ENDPOINT=your_azure_endpoint
LLM_API_KEY=your_api_key
LLM_API_VERSION=2025-01-01-preview
```

## Run All Domains

```bash
python experiments/domain_agents/run_all_experiments.py
```
Generates 30 generated queries per domain (90 total), analyzes each query, discovers missing tools (unlimited), then refiner consolidates to MAX_TOOLS=5.

## Run Real Agent Conversations (Investment Agent only for now)

```bash
PYTHONPATH=src python experiments/domain_agents/investment_agent/hallucinations/run_automated_conversations.py
```
Runs multi-turn conversations with 5 investor profiles * 2 hallucination modes = 10 scenarios, up to 8 turns each.

## Run Refiner Per Domain
- Requires agent conversation/hallucination tool_registry.json file to be created using **all domains OR investment agent only command** as described above

```bash
# Investment Agent
cd experiments/domain_agents/investment_agent/hallucinations && python tool_registry_refiner.py

# Course Enrollment
cd experiments/domain_agents/course_enroll/hallucinations && python tool_registry_refiner.py

# Yelp Restaurant
cd experiments/domain_agents/yelpbot/hallucinations && python tool_registry_refiner.py
```

## Output

Each domain produces:
- `tool_registry.json` - Raw discovered tools (unlimited)
- `tool_registry_refined.json` - Consolidated to MAX_TOOLS=5

## Configuration

Each domain has a `config.yaml` file that controls agent behavior:

```yaml
# experiments/domain_agents/investment_agent/config.yaml
validate_response: false
allow_hallucination: true   # Toggle hallucination mode

prompt_log_path: "logs/investment_agent_prompts.log"

semantic_parser:
    model_name: "azure/gpt-4.1"
# ... other model configs
```

### Config Options

| Option | Values | Description |
|--------|--------|-------------|
| `allow_hallucination` | `true` / `false` | **true**: Agent generates responses even when lacking data (may hallucinate). **false**: Agent refuses to answer when it doesn't have enough information. |
| `validate_response` | `true` / `false` | Enable/disable response validation |
| `prompt_log_path` | path string | Where to log prompts |

### Toggling Hallucination Mode

```bash
# In config.yaml, set:
allow_hallucination: true   # Agent may hallucinate when lacking data
allow_hallucination: false  # Agent refuses instead of hallucinating
```

This is useful for:
- **Testing**: Compare agent behavior with hallucinations on vs off
- **Production**: Set to `false` for safer, more reliable responses
- **Research**: Set to `true` to discover what tools/data the agent needs

## Structure

```
experiments/domain_agents/
├── run_all_experiments.py          # Run all domains
├── shared/                         # Shared code (symlinked)
│   ├── tool_discovery.py           # Discovers tools (unlimited)
│   ├── tool_registry_refiner.py    # Consolidates to MAX_TOOLS=5
│   ├── data_availability.py        # Schema checking
│   └── schema_parser.py            # Parses table_schema.txt
├── investment_agent/
│   ├── hallucinations/
│   │   ├── run_automated_conversations.py
│   │   ├── investor_profiles.py
│   │   └── investor_simulator.py
│   └── table_schema.txt
├── course_enroll/
│   ├── hallucinations/
│   │   ├── student_profiles.py
│   │   └── student_simulator.py
│   └── table_schema.txt
└── yelpbot/
    ├── hallucinations/
    │   ├── yelp_profiles.py
    │   └── yelp_simulator.py
    └── table_schema.txt
```
