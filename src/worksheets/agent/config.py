import os
from functools import wraps

import yaml
from pydantic import BaseModel, Field

# Registry to store all agent APIs
_AGENT_API_REGISTRY = []


class OpenAIModelConfig(BaseModel):
    model_name: str = Field(default="azure/gpt-4.1-mini")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=1024)
    top_p: float = Field(default=1.0)
    frequency_penalty: float = Field(default=0.0)
    presence_penalty: float = Field(default=0.0)
    config_name: str = Field(default="openai", frozen=True)


class AzureModelConfig(BaseModel):
    model_name: str = Field(default="azure/gpt-4.1-mini")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=1024)
    top_p: float = Field(default=1.0)
    frequency_penalty: float = Field(default=0.0)
    presence_penalty: float = Field(default=0.0)
    config_name: str = Field(default="azure", frozen=True)


class Config(BaseModel):
    semantic_parser: OpenAIModelConfig | AzureModelConfig
    response_generator: OpenAIModelConfig | AzureModelConfig
    knowledge_parser: OpenAIModelConfig | AzureModelConfig
    knowledge_base: OpenAIModelConfig | AzureModelConfig

    # number of turns to use for response generator prompt
    # each turn means (user, agent)
    rg_num_turns: int = Field(default=2)

    prompt_log_path: str | None = Field(default=None)
    conversation_log_path: str | None = Field(default=None)
    append_to_conversation_log: bool = Field(default=False)
    validate_response: bool = Field(default=False)
    allow_hallucination: bool = Field(default=False)

    @classmethod
    def load_from_yaml(cls, path: str):
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        for model_config in [
            "semantic_parser",
            "response_generator",
            "knowledge_parser",
            "knowledge_base",
        ]:
            if "azure/" in config[model_config]["model_name"]:
                config[model_config] = AzureModelConfig(
                    **config[model_config],
                )
            else:
                config[model_config] = OpenAIModelConfig(**config[model_config])

        return cls(**config)


def agent_api(name: str = None, description: str = None):
    """Decorator to mark functions as agent APIs

    @agent_api("get_course", "Gets course details")
    def get_course_details(course_id: str):
        ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._is_agent_api = True
        wrapper._api_name = name or func.__name__
        wrapper._api_description = description or func.__doc__

        # Add to registry
        _AGENT_API_REGISTRY.append(wrapper)

        return wrapper

    return decorator


def get_all_agent_apis():
    """Returns a list of all functions decorated with @agent_api"""
    api_list = []
    for api in _AGENT_API_REGISTRY:
        api_list.append((api, api._api_description))
    return api_list


if __name__ == "__main__":
    config = Config.load_from_yaml(
        "/home/harshit/genie-worksheets/experiments/domain_agents/course_enroll/config.yaml"
    )
    print(config)
