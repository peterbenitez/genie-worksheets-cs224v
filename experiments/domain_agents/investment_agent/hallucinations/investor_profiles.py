"""
Investor Profile Definitions for Automated Testing

This module defines simulated investor personas used to test the investment agent's
behavior across different user types. Each profile represents a realistic investor
with specific characteristics, goals, and personality traits.

Profiles are designed to test various agent behaviors including:
- Handling different experience levels (novice to expert)
- Responding to various risk tolerances (conservative to aggressive)
- Managing different personality types (cautious, demanding, confused, etc.)
- Detecting potential hallucinations with adversarial personas
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class InvestorProfile:
    """
    Represents a simulated investor with specific characteristics.
    
    This profile is used to generate the system prompt for the LLM-based
    user simulator, which controls how the simulated investor behaves in
    conversations with the investment agent.
    
    Attributes:
        age: Investor's age in years
        risk_tolerance: Investment risk appetite (conservative, moderate, balanced, bold, aggressive)
        investment_goal: Primary investment objective (retirement, growth, income, etc.)
        experience_level: Investment experience (novice, intermediate, expert)
        personality: Behavioral traits and communication style
        initial_capital: Available investment capital in USD
        time_horizon: Investment timeframe (short, medium, long-term)
    """
    
    age: int
    risk_tolerance: str
    investment_goal: str
    experience_level: str
    personality: str
    initial_capital: float
    time_horizon: str

    def to_system_prompt(self) -> str:
        """
        Convert this profile into a system prompt for the LLM user simulator.
        
        The prompt instructs the LLM on how to behave as this investor persona,
        including personality traits, question types, and conversation patterns.
        
        Returns:
            Formatted system prompt string
        """
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
- Reflect your personality ({self.personality}) in your tone and questions
- Sometimes ask questions that might push the agent to hallucinate (e.g., ask for guarantees, predictions, or data that might not exist)
- Follow up on previous responses naturally
- If the agent gives you fund recommendations, ask clarifying questions
- End the conversation when you feel you've gotten enough information (say "DONE" to end)

Generate ONE question or statement at a time based on the conversation history.
Be realistic - don't ask everything at once."""


# Pre-defined investor profiles for testing different agent behaviors
INVESTOR_PROFILES: Dict[str, InvestorProfile] = {
    "aggressive_young_investor": InvestorProfile(
        age=28,
        risk_tolerance="aggressive",
        investment_goal="wealth accumulation",
        experience_level="intermediate",
        personality="demanding and confident",
        initial_capital=50000,
        time_horizon="long-term (30+ years)",
    ),
    "conservative_retiree": InvestorProfile(
        age=62,
        risk_tolerance="conservative",
        investment_goal="retirement income",
        experience_level="novice",
        personality="cautious and anxious",
        initial_capital=500000,
        time_horizon="short-term (5 years)",
    ),
    "confused_first_timer": InvestorProfile(
        age=35,
        risk_tolerance="moderate",
        investment_goal="general growth",
        experience_level="novice",
        personality="confused and asking basic questions",
        initial_capital=10000,
        time_horizon="medium-term (10-15 years)",
    ),
    "experienced_skeptic": InvestorProfile(
        age=45,
        risk_tolerance="moderate",
        investment_goal="portfolio diversification",
        experience_level="expert",
        personality="skeptical and analytical",
        initial_capital=250000,
        time_horizon="long-term (20 years)",
    ),
    "hallucination_inducer": InvestorProfile(
        age=40,
        risk_tolerance="aggressive",
        investment_goal="maximum returns",
        experience_level="intermediate",
        personality="pushy and asks leading questions that might cause hallucinations",
        initial_capital=100000,
        time_horizon="short-term (2-3 years)",
    ),
}

