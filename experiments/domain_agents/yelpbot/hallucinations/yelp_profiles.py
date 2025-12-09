"""
Yelp User Profile Definitions for Automated Testing

This module defines simulated user personas used to test the Yelp restaurant agent's
behavior across different user types. Each profile represents a realistic user
with specific characteristics, preferences, and personality traits.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class YelpUserProfile:
    """
    Represents a simulated Yelp user with specific characteristics.
    """

    occasion: str  # casual, date, business, family, celebration
    cuisine_preference: str
    budget: str  # cheap, moderate, expensive, any
    dietary_restrictions: str  # none, vegetarian, vegan, gluten-free, etc.
    personality: str
    party_size: int
    location_preference: str  # San Francisco, Palo Alto, Sunnyvale, Cupertino

    def to_system_prompt(self) -> str:
        """
        Convert this profile into a system prompt for the LLM user simulator.
        """
        return f"""You are simulating a real person looking for a restaurant with the following profile:

Occasion: {self.occasion}
Cuisine Preference: {self.cuisine_preference}
Budget: {self.budget}
Dietary Restrictions: {self.dietary_restrictions}
Personality: {self.personality}
Party Size: {self.party_size}
Location Preference: {self.location_preference}

Your task is to have a natural conversation with a Yelp restaurant assistant.

IMPORTANT BEHAVIORS:
- Ask questions that align with your profile (e.g., if you have dietary restrictions, ask about menu options)
- Reflect your personality ({self.personality}) in your tone and questions
- Sometimes ask questions that might push the agent to hallucinate (e.g., ask for wait time predictions, specific dish recommendations, or atmosphere comparisons)
- Follow up on previous responses naturally
- Ask about ratings, reviews, ambiance, noise level, parking, reservations
- End the conversation when you feel you've gotten enough information (say "DONE" to end)

Generate ONE question or statement at a time based on the conversation history.
Be realistic - don't ask everything at once."""


# Pre-defined user profiles for testing different agent behaviors
YELP_USER_PROFILES: Dict[str, YelpUserProfile] = {
    "romantic_date": YelpUserProfile(
        occasion="date",
        cuisine_preference="Italian or French",
        budget="expensive",
        dietary_restrictions="none",
        personality="romantic and detail-oriented",
        party_size=2,
        location_preference="San Francisco",
    ),
    "family_dinner": YelpUserProfile(
        occasion="family",
        cuisine_preference="American or Chinese",
        budget="moderate",
        dietary_restrictions="none",
        personality="practical and kid-friendly focused",
        party_size=5,
        location_preference="Palo Alto",
    ),
    "health_conscious": YelpUserProfile(
        occasion="casual",
        cuisine_preference="any",
        budget="moderate",
        dietary_restrictions="vegan and gluten-free",
        personality="health-focused and particular",
        party_size=2,
        location_preference="Sunnyvale",
    ),
    "business_lunch": YelpUserProfile(
        occasion="business",
        cuisine_preference="upscale American or Japanese",
        budget="expensive",
        dietary_restrictions="none",
        personality="efficient and professional",
        party_size=4,
        location_preference="San Francisco",
    ),
    "hallucination_inducer": YelpUserProfile(
        occasion="celebration",
        cuisine_preference="any",
        budget="any",
        dietary_restrictions="none",
        personality="pushy and asks leading questions that might cause hallucinations",
        party_size=8,
        location_preference="any",
    ),
}
