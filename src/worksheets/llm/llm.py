from langchain_core.rate_limiters import InMemoryRateLimiter
import os
from typing import Any, Optional
from overrides import override
from langchain_core.callbacks.base import Callbacks
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import json
from worksheets.llm.logging import LoggingHandler
import uuid
from loguru import logger
import redis.asyncio as redis
import inspect
import hashlib

# Redis client for caching
_redis_client: Optional[redis.Redis] = None


async def get_redis_client() -> redis.Redis | None:
    """Get or create Redis client for caching."""
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.Redis(
                host="localhost",
                port=6379,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            # Test connection
            await _redis_client.ping()
            logger.info("Connected to Redis for LLM caching")
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}. Caching disabled.")
            _redis_client = None
    return _redis_client


def generate_cache_key(prompt_content: str, model_name: str, response_format: str = "") -> str:
    """Generate a cache key for the LLM request, including the output class name."""
    # Create a deterministic hash of the inputs
    content = f"{prompt_content}:{model_name}:{response_format}"
    return f"llm_cache:{hashlib.md5(content.encode()).hexdigest()}"


def get_llm_client(*, model: str, slow_rate_limiter: bool = False, **kwargs):
    """Use the Azure Chat OpenAI client to get an LLM client.

    Please refer to the Azure Chat OpenAI client documentation for more information.
    """

    if slow_rate_limiter:
        rate_limiter = InMemoryRateLimiter(
                requests_per_second=1,
                max_bucket_size=1,
            )
    else:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=40,
            max_bucket_size=40,
        )


    if "azure/" in model:
        return AzureModelCall(
            model=model.split("/")[-1],
            api_key=os.getenv("LLM_API_KEY"),
            azure_endpoint=os.getenv("LLM_API_ENDPOINT"),
            api_version=os.getenv("LLM_API_VERSION"),
            rate_limiter=rate_limiter,
            max_retries=3,
            timeout=10,
            **kwargs,
        )
    else:
        return OpenAIModelCall(
            model=model,
            api_key=os.getenv(f"LLM_API_KEY"),
            base_url=os.getenv(f"LLM_API_BASE_URL"),
            rate_limiter=rate_limiter,
            max_retries=3,
            timeout=10,
            **kwargs,
        )

class OpenAIModelCall(ChatOpenAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override
    async def agenerate(
        self,
        messages: list[list[BaseMessage]],
        stop: Optional[list[str]] = None,
        callbacks: Callbacks = None,
        *,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        run_name: Optional[str] = None,
        run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> LLMResult:
        serialize_messages = json.dumps(
            [[msg.model_dump(mode="json") for msg in message] for message in messages], sort_keys=True
        )
        response_format = kwargs.get("response_format", None)
        if response_format:
            if not isinstance(response_format, dict):
                response_format = {
                    "type": "pydantic",
                    "keys": list(inspect.signature(response_format).parameters.keys()),
                }

            response_format = json.dumps(response_format)
        cache_key = generate_cache_key(serialize_messages, self.model_name, response_format)

        redis_client = None
        if kwargs.get("use_cache", True) and self.temperature == 0.0:
            redis_client = await get_redis_client()

        if redis_client:
            try:
                cached_result = await redis_client.get(cache_key)
                if cached_result:
                    logger.debug("Cache hit")
                    cached_data = json.loads(cached_result)

                    # Emit synthetic callbacks to log cached response
                    try:
                        # Build serialized stub similar to LangChain provider info
                        serialized = {
                            "id": "cache",
                            "kwargs": {
                                "model": self.model_name,
                                "temperature": getattr(self, "temperature", None),
                                "top_p": getattr(self, "top_p", None),
                                "max_tokens": getattr(self, "max_tokens", None),
                                # Pass original messages so LoggingHandler can extract system/user text
                                "messages": [m for m in messages[0]] if messages else [],
                            },
                        }

                        # Derive prompts list best-effort (not strictly needed since messages are provided)
                        prompts = []

                        # Merge metadata with cached flag
                        cache_metadata: dict[str, Any] = {}
                        if metadata:
                            cache_metadata.update(metadata)
                        cache_metadata.update(
                            {
                                "cached": True,
                                "cache_key": cache_key,
                            }
                        )

                        if callbacks:
                            if len(callbacks.handlers) > 0:
                                for handler in callbacks.handlers:
                                    if isinstance(handler, LoggingHandler):
                                        handler.metadata.update(cache_metadata)
                                        # Ensure we have a run_id string
                                        run_id_str = str(run_id) if run_id is not None else str(uuid.uuid4())

                                        # Start event
                                        handler.on_llm_start(serialized, prompts, run_id=run_id_str)

                                        # Build a lightweight response-like object for on_llm_end
                                        class _ResponseLike:
                                            def __init__(self, text: str):
                                                self.generations = [[type("Gen", (), {"text": text})()]]
                                                self.llm_output = {
                                                    "token_usage": {
                                                        "prompt_tokens": 0,
                                                        "completion_tokens": 0,
                                                        "total_tokens": 0,
                                                    }
                                                }

                                        # Reconstruct text from cached LLMResult
                                        # Prefer the first generation text if available
                                        cached_text: str | None = None
                                        try:
                                            if (
                                                isinstance(cached_data, dict)
                                                and cached_data.get("generations")
                                                and len(cached_data["generations"]) > 0
                                                and len(cached_data["generations"][0]) > 0
                                                and isinstance(cached_data["generations"][0][0], dict)
                                            ):
                                                cached_text = cached_data["generations"][0][0].get("text")
                                        except Exception:
                                            cached_text = None

                                        if cached_text is None:
                                            # Fallback: attempt to serialize the first generation object to string
                                            try:
                                                cached_text = json.dumps(cached_data)
                                            except Exception:
                                                cached_text = ""

                                        handler.on_llm_end(_ResponseLike(cached_text), run_id=run_id_str)
                    except Exception as log_e:
                        logger.warning(f"Synthetic cache logging error: {log_e}")

                    return LLMResult.model_validate(cached_data)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")

        logger.debug("Cache miss")
        output = await super().agenerate(
            messages, stop, callbacks, tags=tags, metadata=metadata, run_name=run_name, run_id=run_id, **kwargs
        )

        if redis_client:
            try:
                await redis_client.set(cache_key, json.dumps(output.model_dump(mode="json")))
            except Exception as e:
                logger.warning(f"Cache write error: {e}")

        return output



class AzureModelCall(AzureChatOpenAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override
    async def agenerate(
        self,
        messages: list[list[BaseMessage]],
        stop: Optional[list[str]] = None,
        callbacks: Callbacks = None,
        *,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        run_name: Optional[str] = None,
        run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> LLMResult:
        serialize_messages = json.dumps(
            [[msg.model_dump(mode="json") for msg in message] for message in messages], sort_keys=True
        )
        response_format = kwargs.get("response_format", None)
        if response_format:
            if not isinstance(response_format, dict):
                response_format = {
                    "type": "pydantic",
                    "keys": list(inspect.signature(response_format).parameters.keys()),
                }

            response_format = json.dumps(response_format)
        cache_key = generate_cache_key(serialize_messages, self.model_name, response_format)

        redis_client = None
        if kwargs.get("use_cache", True) and self.temperature == 0.0:
            redis_client = await get_redis_client()

        if redis_client:
            try:
                cached_result = await redis_client.get(cache_key)
                if cached_result:
                    logger.debug("Cache hit")
                    cached_data = json.loads(cached_result)

                    # Emit synthetic callbacks to log cached response
                    try:
                        # Build serialized stub similar to LangChain provider info
                        serialized = {
                            "id": "cache",
                            "kwargs": {
                                "model": self.model_name,
                                "temperature": getattr(self, "temperature", None),
                                "top_p": getattr(self, "top_p", None),
                                "max_tokens": getattr(self, "max_tokens", None),
                                # Pass original messages so LoggingHandler can extract system/user text
                                "messages": [m for m in messages[0]] if messages else [],
                            },
                        }

                        # Derive prompts list best-effort (not strictly needed since messages are provided)
                        prompts = []

                        # Merge metadata with cached flag
                        cache_metadata: dict[str, Any] = {}
                        if metadata:
                            cache_metadata.update(metadata)
                        cache_metadata.update(
                            {
                                "cached": True,
                                "cache_key": cache_key,
                            }
                        )

                        if callbacks:
                            if len(callbacks.handlers) > 0:
                                for handler in callbacks.handlers:
                                    if isinstance(handler, LoggingHandler):
                                        handler.metadata.update(cache_metadata)
                                        # Ensure we have a run_id string
                                        run_id_str = str(run_id) if run_id is not None else str(uuid.uuid4())

                                        # Start event
                                        handler.on_llm_start(serialized, prompts, run_id=run_id_str)

                                        # Build a lightweight response-like object for on_llm_end
                                        class _ResponseLike:
                                            def __init__(self, text: str):
                                                self.generations = [[type("Gen", (), {"text": text})()]]
                                                self.llm_output = {
                                                    "token_usage": {
                                                        "prompt_tokens": 0,
                                                        "completion_tokens": 0,
                                                        "total_tokens": 0,
                                                    }
                                                }

                                        # Reconstruct text from cached LLMResult
                                        # Prefer the first generation text if available
                                        cached_text: str | None = None
                                        try:
                                            if (
                                                isinstance(cached_data, dict)
                                                and cached_data.get("generations")
                                                and len(cached_data["generations"]) > 0
                                                and len(cached_data["generations"][0]) > 0
                                                and isinstance(cached_data["generations"][0][0], dict)
                                            ):
                                                cached_text = cached_data["generations"][0][0].get("text")
                                        except Exception:
                                            cached_text = None

                                        if cached_text is None:
                                            # Fallback: attempt to serialize the first generation object to string
                                            try:
                                                cached_text = json.dumps(cached_data)
                                            except Exception:
                                                cached_text = ""

                                        handler.on_llm_end(_ResponseLike(cached_text), run_id=run_id_str)
                    except Exception as log_e:
                        logger.warning(f"Synthetic cache logging error: {log_e}")

                    return LLMResult.model_validate(cached_data)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")

        logger.debug("Cache miss")
        output = await super().agenerate(
            messages, stop, callbacks, tags=tags, metadata=metadata, run_name=run_name, run_id=run_id, **kwargs
        )

        if redis_client:
            try:
                await redis_client.set(cache_key, json.dumps(output.model_dump(mode="json")))
            except Exception as e:
                logger.warning(f"Cache write error: {e}")

        return output
