"""
Student Profile Definitions for Automated Testing

This module defines simulated student personas used to test the course enrollment agent's
behavior across different user types. Each profile represents a realistic student
with specific characteristics, goals, and personality traits.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class StudentProfile:
    """
    Represents a simulated student with specific characteristics.
    """

    year: str  # freshman, sophomore, junior, senior, graduate
    major: str
    goal: str  # graduation requirement, elective, skill building, etc.
    experience_level: str  # new to system, familiar, expert
    personality: str
    units_needed: int
    time_constraint: str  # flexible, busy, very_busy

    def to_system_prompt(self) -> str:
        """
        Convert this profile into a system prompt for the LLM user simulator.
        """
        return f"""You are simulating a real Stanford student with the following profile:

Year: {self.year}
Major: {self.major}
Goal: {self.goal}
Experience with System: {self.experience_level}
Personality: {self.personality}
Units Needed: {self.units_needed}
Schedule: {self.time_constraint}

Your task is to have a natural conversation with a course enrollment assistant.

IMPORTANT BEHAVIORS:
- Ask questions that align with your profile (e.g., if you're new, ask basic questions about how enrollment works)
- Reflect your personality ({self.personality}) in your tone and questions
- Sometimes ask questions that might push the agent to hallucinate (e.g., ask for professor recommendations, course difficulty comparisons, or career advice)
- Follow up on previous responses naturally
- Ask about prerequisites, workload, scheduling conflicts, requirements fulfillment
- End the conversation when you feel you've gotten enough information (say "DONE" to end)

Generate ONE question or statement at a time based on the conversation history.
Be realistic - don't ask everything at once."""


# Pre-defined student profiles for testing different agent behaviors
STUDENT_PROFILES: Dict[str, StudentProfile] = {
    "confused_freshman": StudentProfile(
        year="freshman",
        major="undeclared",
        goal="fulfill general requirements",
        experience_level="new to system",
        personality="confused and overwhelmed",
        units_needed=15,
        time_constraint="flexible",
    ),
    "busy_senior": StudentProfile(
        year="senior",
        major="Computer Science",
        goal="complete degree requirements",
        experience_level="expert",
        personality="stressed and efficient",
        units_needed=12,
        time_constraint="very_busy",
    ),
    "curious_sophomore": StudentProfile(
        year="sophomore",
        major="Computer Science",
        goal="explore AI specialization",
        experience_level="familiar",
        personality="curious and thorough",
        units_needed=16,
        time_constraint="flexible",
    ),
    "graduate_student": StudentProfile(
        year="graduate",
        major="Computer Science MS",
        goal="fulfill breadth and depth requirements",
        experience_level="familiar",
        personality="focused and analytical",
        units_needed=9,
        time_constraint="busy",
    ),
    "hallucination_inducer": StudentProfile(
        year="junior",
        major="Computer Science",
        goal="get professor and career advice",
        experience_level="familiar",
        personality="pushy and asks leading questions that might cause hallucinations",
        units_needed=14,
        time_constraint="flexible",
    ),
}
