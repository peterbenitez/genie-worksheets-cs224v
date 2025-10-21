from worksheets.agent.builder import AgentBuilder
from worksheets.agent.config import AzureModelConfig, Config, OpenAIModelConfig
from worksheets.knowledge.parser import (
    DatatalkParser,
    SUQLKnowledgeBase,
    SUQLReActParser,
)
from worksheets.utils.interface import conversation_loop
from worksheets.llm.prompts import init_llm

import os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

# Initialize with multiple prompt directories
prompt_dirs = [
    os.path.join(CURRENT_DIR, "prompts"),
    os.path.join(CURRENT_DIR, "kraken", "prompts"),
]

init_llm(
    prompt_dir=prompt_dirs,
    dotenv_path=os.path.join(CURRENT_DIR, "..", ".env"),
)


__all__ = [
    "SUQLKnowledgeBase",
    "SUQLReActParser",
    "DatatalkParser",
    "conversation_loop",
    "AgentBuilder",
    "Config",
    "AzureModelConfig",
    "OpenAIModelConfig",
]
