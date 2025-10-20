import json
import os
import random
import sys
from typing import Any, Dict, List

import chainlit as cl
from loguru import logger

from worksheets import Config
from worksheets.agent.chainlit import ChainlitAgent
from worksheets.core.dialogue import CurrentDialogueTurn
from worksheets.utils.annotation import get_agent_action_schemas, get_context_schema
from worksheets.utils.chainlite_config import load_chainlite_config

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, ".."))

# Initialize LLM configuration
load_chainlite_config(os.path.join(current_dir, "llm_config.yaml"))

from investment_agent import agent_builder

# Constants
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data", "user_conversation")
LOGS_DIR = os.path.join(CURRENT_DIR, "..", "user_logs_investment_agent")
LOGS_FILE = os.path.join(LOGS_DIR, "user_logs_250628.log")


# Extend python path to include parent directories
sys.path.append(os.path.join(CURRENT_DIR, "..", ".."))

# Configure logger
logger.remove()
logger.add(LOGS_FILE, rotation="1 day")

# Load configurations from YAML

config = Config.load_from_yaml(os.path.join(CURRENT_DIR, "..", "config.yaml"))
config.conversation_log_path = os.path.join(
    CURRENT_DIR, "..", "logs", "investment_agent_conversation.json"
)
config.prompt_log_path = os.path.join(
    CURRENT_DIR, "..", "logs", "investment_agent_prompts.jsonl"
)


def convert_to_json(dialogue: List[CurrentDialogueTurn]) -> List[Dict[str, Any]]:
    """
    Convert dialogue history to JSON format for storage.

    Args:
        dialogue: List of dialogue turns

    Returns:
        List of dialogue turns in JSON format
    """
    json_dialogue = []
    for turn in dialogue:
        json_turn = {
            "user": turn.user_utterance,
            "bot": turn.system_response,
            "turn_context": get_context_schema(turn.context),
            "global_context": get_context_schema(turn.global_context),
            "system_action": get_agent_action_schemas(turn.system_action),
            "user_target_sp": turn.user_target_sp,
            "user_target": turn.user_target,
            "user_target_suql": turn.user_target_suql,
        }
        json_dialogue.append(json_turn)
    return json_dialogue


def get_user_conversation_dir(user_id: str) -> str:
    """
    Get the user conversation directory path.

    Args:
        user_id: User ID

    Returns:
        Path to user conversation directory
    """
    return os.path.join(DATA_DIR, user_id)


def ensure_user_dir_exists(user_id: str) -> None:
    """
    Ensure that the user conversation directory exists.

    Args:
        user_id: User ID
    """
    user_dir = get_user_conversation_dir(user_id)
    os.makedirs(user_dir, exist_ok=True)


def save_conversation(
    user_id: str, dialogue: List[CurrentDialogueTurn], filename: str
) -> str:
    """
    Save conversation to a file.

    Args:
        user_id: User ID
        dialogue: Dialogue history
        filename: Filename to save to

    Returns:
        Path to saved file
    """
    user_dir = get_user_conversation_dir(user_id)
    file_path = os.path.join(user_dir, filename)

    with open(file_path, "w") as f:
        json.dump(convert_to_json(dialogue), f, indent=4)

    return file_path


def create_agent() -> ChainlitAgent:
    """
    Create and configure the Investment Agent.

    Returns:
        Configured ChainlitAgent
    """
    return agent_builder.build(config, ChainlitAgent)


@cl.on_chat_start
async def initialize():
    """Initialize the chat session when a user starts a conversation."""
    agent = create_agent()
    agent.enter()
    user_id = random.randint(1000, 9999)
    user_risk_profile = random.choice(
        ["conservative", "moderate", "balanced", "bold", "aggressive"]
    )
    agent.runtime.context.update(
        {
            "user_profile": agent.runtime.context.context["UserProfile"](
                user_id=user_id, risk_profile=user_risk_profile
            ),
        }
    )
    cl.user_session.set("bot", agent)

    user_id = cl.user_session.get("id")
    logger.info(f"Chat started for user {user_id}")

    ensure_user_dir_exists(user_id)

    welcome_message = (
        f"{agent.starting_prompt}\n\nConversation ID: **{user_id}**\n"
        # "Please be a user who asks several questions.\n\n"
        # "It is mandatory for users (you) to enroll in at least two courses.\n"
        # "**Note:** The database only contains Computer Science courses and contains course data till the academic 2023."
    )

    await cl.Message(welcome_message).send()


@cl.on_message
async def get_user_message(message):
    """Handle user messages and generate responses."""
    agent = cl.user_session.get("bot")
    await agent.generate_next_turn(message.content)

    cl.user_session.set("bot", agent)
    user_id = cl.user_session.get("id")
    response = agent.dlg_history[-1].system_response

    ensure_user_dir_exists(user_id)

    # Save intermediate conversation
    file_path = save_conversation(
        user_id, agent.dlg_history, "intermediate_conversation.json"
    )

    # Send response with conversation log attachment
    await cl.Message(
        response,
        author="Investment Agent",
        elements=[cl.File(name="conv_log.json", path=file_path)],
    ).send()


@cl.on_chat_end
def on_chat_end():
    """Handle the end of a chat session."""
    user_id = cl.user_session.get("id")
    agent = cl.user_session.get("bot")
    agent.close()
    ensure_user_dir_exists(user_id)

    if agent.dlg_history:
        user_dir = get_user_conversation_dir(user_id)
        file_name = "conversation.json"
        file_path = os.path.join(user_dir, file_name)

        # If file already exists, create a new one with random suffix
        if os.path.exists(file_path):
            file_name = f"conversation_{random.randint(0, 1000)}.json"
            file_path = os.path.join(user_dir, file_name)

        save_conversation(user_id, agent.dlg_history, file_name)
    else:
        # If no dialogue history, remove the user directory
        try:
            user_dir = get_user_conversation_dir(user_id)
            os.rmdir(user_dir)
        except OSError:
            # Directory might not be empty or might not exist
            logger.warning(f"Could not remove directory for user {user_id}")

    logger.info(f"Chat ended for user {user_id}")
