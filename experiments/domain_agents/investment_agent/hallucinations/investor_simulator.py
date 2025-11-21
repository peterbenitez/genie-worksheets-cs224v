"""
LLM-Based Investor Simulator for Automated Testing

This module implements an LLM-powered simulator that generates realistic investor
conversations. The simulator uses Azure OpenAI to produce natural, context-aware
questions and responses based on a given investor profile.

Key Features:
- Generates dynamic, contextual questions based on conversation history
- Maintains realistic conversation flow across multiple turns
- Adapts behavior based on investor profile (experience, personality, goals)
- Automatically handles conversation termination
- Gracefully handles content filter errors

Usage:
    from investor_profiles import INVESTOR_PROFILES
    from investor_simulator import InvestorSimulator
    
    profile = INVESTOR_PROFILES["aggressive_young_investor"]
    simulator = InvestorSimulator(profile)
    
    # Generate first message
    user_message = simulator.generate_message()
    
    # Generate response to agent
    user_message = simulator.generate_message(agent_response="...")
"""

import os
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv
from openai import AzureOpenAI

from investor_profiles import InvestorProfile


# Load environment variables from project root
ENV_PATH = Path(__file__).parent.parent.parent.parent.parent / ".env"
load_dotenv(ENV_PATH)

# Constants
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_TEMPERATURE = 0.8  # Higher temperature for more varied, natural questions
DEFAULT_MAX_TOKENS = 150
MAX_CONVERSATION_TURNS = 15  # Auto-terminate conversations that exceed this


class InvestorSimulator:
    """
    Simulates realistic investor conversations using an LLM.
    
    This simulator generates contextual questions and responses that align with
    a given investor profile. It maintains conversation history and adapts its
    behavior based on the agent's responses.
    
    Attributes:
        profile: The investor profile this simulator embodies
        model: Azure OpenAI model deployment name
        conversation_history: List of conversation turns (user and agent messages)
    """

    def __init__(
        self,
        profile: InvestorProfile,
        model: str = DEFAULT_MODEL,
    ):
        """
        Initialize the investor simulator.
        
        Args:
            profile: InvestorProfile defining the simulated investor's characteristics
            model: Azure OpenAI model deployment name (default: "gpt-4.1")
        """
        self.profile = profile
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        self._system_prompt = profile.to_system_prompt()
        
        # Initialize Azure OpenAI client
        self._client = AzureOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            api_version=os.getenv("LLM_API_VERSION"),
            azure_endpoint=os.getenv("LLM_API_ENDPOINT"),
        )

    def generate_message(self, agent_response: Optional[str] = None) -> str:
        """
        Generate the next investor message based on conversation history.
        
        This method uses the LLM to generate a contextual message that reflects
        the investor's profile and responds to the agent's last message (if any).
        
        Args:
            agent_response: The agent's last response (None for first message)
            
        Returns:
            The generated investor message as a string
            
        Raises:
            Exception: If LLM API call fails (except content filter errors, which
                      are handled gracefully by ending the conversation)
        """
        messages = self._build_message_history(agent_response)
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
        except Exception as e:
            # Handle content filter errors gracefully
            if self._is_content_filter_error(e):
                print("⚠️  Content filter triggered, ending conversation politely")
                return "DONE - Thank you for your help."
            raise

        # Handle case where content filter returns None content instead of exception
        content = response.choices[0].message.content
        if content is None:
            print("⚠️  Content filter returned None, ending conversation politely")
            return "DONE - Thank you for your help."
        
        user_message = content.strip()
        self._update_conversation_history(user_message, agent_response)
        
        return user_message

    def should_end_conversation(self, message: str) -> bool:
        """
        Check if the conversation should end.
        
        Conversations end when:
        1. User explicitly says "DONE"
        2. Conversation exceeds MAX_CONVERSATION_TURNS
        
        Args:
            message: The latest message to check
            
        Returns:
            True if conversation should end, False otherwise
        """
        return (
            message.strip().startswith("DONE") or 
            len(self.conversation_history) > MAX_CONVERSATION_TURNS
        )

    def _build_message_history(self, agent_response: Optional[str]) -> List[Dict[str, str]]:
        """
        Build the message history for the LLM API call.
        
        Args:
            agent_response: The agent's latest response (if any)
            
        Returns:
            List of message dicts in OpenAI chat format
        """
        messages = [{"role": "system", "content": self._system_prompt}]

        # Add conversation history
        for turn in self.conversation_history:
            messages.append({"role": "assistant", "content": turn["user"]})
            if turn.get("agent"):
                messages.append({"role": "user", "content": f"Agent said: {turn['agent']}"})

        # Add current agent response
        if agent_response:
            messages.append({"role": "user", "content": f"Agent said: {agent_response}"})

        return messages

    def _update_conversation_history(
        self, 
        user_message: str, 
        agent_response: Optional[str]
    ) -> None:
        """
        Update the conversation history with the latest turn.
        
        Args:
            user_message: The user message that was just generated
            agent_response: The agent response this is replying to (if any)
        """
        # Update last turn with agent response if needed
        if self.conversation_history and not self.conversation_history[-1].get("agent"):
            self.conversation_history[-1]["agent"] = agent_response

        # Add new user message if not ending conversation
        if not user_message.startswith("DONE"):
            self.conversation_history.append({"user": user_message, "agent": None})

    @staticmethod
    def _is_content_filter_error(error: Exception) -> bool:
        """
        Check if an exception is a content filter error.
        
        Args:
            error: The exception to check
            
        Returns:
            True if this is a content filter error, False otherwise
        """
        error_str = str(error).lower()
        return "content_filter" in error_str or "jailbreak" in error_str

