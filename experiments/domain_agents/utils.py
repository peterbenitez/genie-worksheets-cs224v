"""
Shared utilities for domain agent experiments.
"""

import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
ENV_PATH = Path(__file__).parent.parent.parent / ".env"
load_dotenv(ENV_PATH)


def get_openai_client() -> AzureOpenAI:
    """Get configured Azure OpenAI client."""
    return AzureOpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        api_version=os.getenv("LLM_API_VERSION"),
        azure_endpoint=os.getenv("LLM_API_ENDPOINT"),
    )


def discover_domains(base_path: Path) -> Dict[str, Path]:
    """
    Auto-discover domains by finding directories with hallucinations/tool_registry.json.

    Returns:
        Dict mapping domain name to domain directory path
    """
    domains = {}

    for item in base_path.iterdir():
        if not item.is_dir() or item.name.startswith('.'):
            continue

        registry_path = item / "hallucinations" / "tool_registry.json"
        if registry_path.exists():
            domain_name = item.name.replace("_agent", "")
            domains[domain_name] = item

    return domains
