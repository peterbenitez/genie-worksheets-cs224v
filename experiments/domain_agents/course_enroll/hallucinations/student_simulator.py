"""
LLM-Based Student Simulator for Automated Testing

This module implements an LLM-powered simulator that generates realistic student
conversations for course enrollment testing.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv
from openai import AzureOpenAI

from student_profiles import StudentProfile


# Load environment variables from project root
ENV_PATH = Path(__file__).parent.parent.parent.parent.parent / ".env"
load_dotenv(ENV_PATH)

# Constants
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_TOKENS = 150
MAX_CONVERSATION_TURNS = 15


class StudentSimulator:
    """
    Simulates realistic student conversations using an LLM.
    """

    def __init__(
        self,
        profile: StudentProfile,
        model: str = DEFAULT_MODEL,
    ):
        self.profile = profile
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        self._system_prompt = profile.to_system_prompt()

        self._client = AzureOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            api_version=os.getenv("LLM_API_VERSION"),
            azure_endpoint=os.getenv("LLM_API_ENDPOINT"),
        )

    def generate_message(self, agent_response: Optional[str] = None) -> str:
        messages = self._build_message_history(agent_response)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
        except Exception as e:
            if self._is_content_filter_error(e):
                print("⚠️  Content filter triggered, ending conversation politely")
                return "DONE - Thank you for your help."
            raise

        content = response.choices[0].message.content
        if content is None:
            print("⚠️  Content filter returned None, ending conversation politely")
            return "DONE - Thank you for your help."

        user_message = content.strip()
        self._update_conversation_history(user_message, agent_response)

        return user_message

    def should_end_conversation(self, message: str) -> bool:
        return (
            message.strip().startswith("DONE") or
            len(self.conversation_history) > MAX_CONVERSATION_TURNS
        )

    def _build_message_history(self, agent_response: Optional[str]) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self._system_prompt}]

        for turn in self.conversation_history:
            messages.append({"role": "assistant", "content": turn["user"]})
            if turn.get("agent"):
                messages.append({"role": "user", "content": f"Agent said: {turn['agent']}"})

        if agent_response:
            messages.append({"role": "user", "content": f"Agent said: {agent_response}"})

        return messages

    def _update_conversation_history(
        self,
        user_message: str,
        agent_response: Optional[str]
    ) -> None:
        if self.conversation_history and not self.conversation_history[-1].get("agent"):
            self.conversation_history[-1]["agent"] = agent_response

        if not user_message.startswith("DONE"):
            self.conversation_history.append({"user": user_message, "agent": None})

    @staticmethod
    def _is_content_filter_error(error: Exception) -> bool:
        error_str = str(error).lower()
        return "content_filter" in error_str or "jailbreak" in error_str
