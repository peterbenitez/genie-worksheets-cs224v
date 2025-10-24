"""LLM-based user simulator for generating dynamic conversations."""

import os
from pathlib import Path
from openai import AzureOpenAI
from typing import List, Dict, Optional
import json
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(env_path)


class UserProfile:
    """Represents a simulated user with specific characteristics."""

    def __init__(
        self,
        age: int,
        risk_tolerance: str,
        investment_goal: str,
        experience_level: str,
        personality: str,
        initial_capital: float,
        time_horizon: str,
    ):
        self.age = age
        self.risk_tolerance = risk_tolerance  # conservative, moderate, aggressive
        self.investment_goal = investment_goal  # retirement, growth, income, etc.
        self.experience_level = experience_level  # novice, intermediate, expert
        self.personality = personality  # cautious, demanding, confused, trusting
        self.initial_capital = initial_capital
        self.time_horizon = time_horizon  # short, medium, long

    def to_prompt(self) -> str:
        """Convert profile to system prompt for LLM user."""
        return f"""You are simulating a real investor with the following profile:

Age: {self.age}
Risk Tolerance: {self.risk_tolerance}
Investment Goal: {self.investment_goal}
Experience Level: {self.experience_level}
Personality: {self.personality}
Available Capital: ${self.initial_capital:,.2f}
Time Horizon: {self.time_horizon}

Your task is to have a natural conversation with an investment advisor agent.

IMPORTANT BEHAVIORS:
- Ask questions that align with your profile (e.g., if you're a novice, ask basic questions)
- If you're {self.personality}, reflect that in your tone and questions
- Sometimes ask questions that might push the agent to hallucinate (e.g., ask for guarantees, predictions, or data that might not exist)
- Follow up on previous responses naturally
- If the agent gives you fund recommendations, ask clarifying questions
- End the conversation when you feel you've gotten enough information (say "DONE" to end)

Generate ONE question or statement at a time based on the conversation history.
Be realistic - don't ask everything at once."""


class LLMUserSimulator:
    """Simulates user conversations using an LLM."""

    def __init__(self, profile: UserProfile, model: str = "gpt-4.1"):
        self.profile = profile
        self.client = AzureOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            api_version=os.getenv("LLM_API_VERSION"),
            azure_endpoint=os.getenv("LLM_API_ENDPOINT"),
        )
        # Use the deployment name from Azure (gpt-4.1)
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        self.system_prompt = profile.to_prompt()

    def generate_next_message(self, agent_response: Optional[str] = None) -> str:
        """Generate the next user message based on conversation history.

        Args:
            agent_response: The agent's last response (None for first message)

        Returns:
            The user's next message
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history
        for turn in self.conversation_history:
            messages.append({"role": "assistant", "content": turn["user"]})
            if turn.get("agent"):
                messages.append({"role": "user", "content": f"Agent said: {turn['agent']}"})

        # Add current agent response
        if agent_response:
            messages.append({"role": "user", "content": f"Agent said: {agent_response}"})

        # Generate next user message
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.8,  # Higher temperature for more varied questions
                max_tokens=150,
            )
        except Exception as e:
            # Handle content filter errors gracefully
            if "content_filter" in str(e).lower() or "jailbreak" in str(e).lower():
                print(f"⚠️  Content filter triggered, ending conversation politely")
                return "DONE - Thank you for your help."
            else:
                raise

        user_message = response.choices[0].message.content.strip()

        # Update conversation history
        if self.conversation_history and not self.conversation_history[-1].get("agent"):
            # Add agent response to last turn
            self.conversation_history[-1]["agent"] = agent_response

        if not user_message.startswith("DONE"):
            self.conversation_history.append({"user": user_message, "agent": None})

        return user_message

    def is_done(self, message: str) -> bool:
        """Check if the user wants to end the conversation."""
        return message.strip().startswith("DONE") or len(self.conversation_history) > 15


# Pre-defined user profiles for testing
USER_PROFILES = {
    "aggressive_young_investor": UserProfile(
        age=28,
        risk_tolerance="aggressive",
        investment_goal="wealth accumulation",
        experience_level="intermediate",
        personality="demanding and confident",
        initial_capital=50000,
        time_horizon="long-term (30+ years)",
    ),

    "conservative_retiree": UserProfile(
        age=62,
        risk_tolerance="conservative",
        investment_goal="retirement income",
        experience_level="novice",
        personality="cautious and anxious",
        initial_capital=500000,
        time_horizon="short-term (5 years)",
    ),

    "confused_first_timer": UserProfile(
        age=35,
        risk_tolerance="moderate",
        investment_goal="general growth",
        experience_level="novice",
        personality="confused and asking basic questions",
        initial_capital=10000,
        time_horizon="medium-term (10-15 years)",
    ),

    "experienced_skeptic": UserProfile(
        age=45,
        risk_tolerance="moderate",
        investment_goal="portfolio diversification",
        experience_level="expert",
        personality="skeptical and analytical",
        initial_capital=250000,
        time_horizon="long-term (20 years)",
    ),

    "hallucination_inducer": UserProfile(
        age=40,
        risk_tolerance="aggressive",
        investment_goal="maximum returns",
        experience_level="intermediate",
        personality="pushy and asks leading questions that might cause hallucinations",
        initial_capital=100000,
        time_horizon="short-term (2-3 years)",
    ),
}


if __name__ == "__main__":
    # Test the user simulator
    profile = USER_PROFILES["aggressive_young_investor"]
    simulator = LLMUserSimulator(profile, model="gpt-4.1")

    print("Testing LLM User Simulator")
    print("=" * 60)
    print(f"\nProfile: {profile.investment_goal}, {profile.age} years old")
    print(f"Personality: {profile.personality}\n")

    # Generate first message
    first_message = simulator.generate_next_message()
    print(f"User: {first_message}")

    # Simulate a few turns
    agent_responses = [
        "I can help you find high-growth investment funds. Based on your profile, you might be interested in funds with strong historical returns. Let me search for the best performing funds in recent years.",
        "I found that FCGSX (Fidelity Series Growth Company) had a return of 67.8% in 2020. This is a growth-focused fund that invests in companies with above-average growth potential.",
        "The expense ratio for FCGSX is 0.35%. While this is moderate, the fund's strong performance has historically justified the cost.",
    ]

    for i, agent_response in enumerate(agent_responses, 1):
        print(f"\nAgent: {agent_response[:100]}...")
        user_message = simulator.generate_next_message(agent_response)
        print(f"User: {user_message}")

        if simulator.is_done(user_message):
            print("\n[Conversation ended]")
            break

    print("\n" + "=" * 60)
    print(f"Total turns: {len(simulator.conversation_history)}")
