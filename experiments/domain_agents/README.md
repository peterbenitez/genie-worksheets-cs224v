# GenieWorksheet Wizard - Hallucination Analysis Framework

Discovers missing tools by forcing agent hallucinations on out-of-scope queries, then analyzes traces to infer missing APIs, database columns, or policy logic.

## Pipeline

```
User Personas -> Simulated Queries (30/domain) -> Force Hallucination -> Schema Check -> LLM Analysis -> Tool Discovery ->  Refinement (MAX_TOOLS=5)
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
Generates 30 queries/domain (90 total), analyzes each query, discovers missing tools, and consolidates to MAX_TOOLS=5.

## Run Real Agent Conversations (Investment Agent)

```bash
PYTHONPATH=src python experiments/domain_agents/investment_agent/hallucinations/run_automated_conversations.py
```
Runs multi-turn conversations with 5 investor profiles × 2 hallucination modes = 10 scenarios, up to 8 turns each.

## Run Refiner Per Domain

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
- `tool_registry.json` - Raw discovered tools
- `tool_registry_refined.json` - Refined to MAX_TOOLS=5

## Structure

```
experiments/domain_agents/
├── run_all_experiments.py          # Run all domains
├── shared/                         # Shared code (symlinked)
│   ├── tool_discovery.py           # MAX_TOOLS=5 enforcement
│   ├── tool_registry_refiner.py    # Merges similar tools
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
