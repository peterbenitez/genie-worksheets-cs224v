"""
Automated Conversation Test Runner

This script orchestrates automated conversations between simulated investors and
the investment agent. It's designed to test agent behavior across different scenarios,
particularly focusing on hallucination detection and refusal behavior.

Test Matrix:
- Multiple investor profiles (conservative, aggressive, skeptical, etc.)
- Multiple agent personalities (friendly, conservative, etc.)
- Hallucination modes (on = agent may hallucinate, off = agent refuses to answer instead of hallucinating )

Results include:
- Full conversation logs
- Hallucination/refusal metrics
- Comparative analysis across configurations
- Timestamped JSON output files

Usage:
    python run_automated_conversations.py

Output:
    - Console: Real-time conversation display and summary statistics
    - File: JSON results in hallucination_results/ directory
"""

import asyncio
import json
import logging
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv
from loguru import logger

# Suppress debug logging from GenieWorksheets framework (Comment out to see all genie agent logs)
logger.remove()  # Remove default loguru handler

# Filter to exclude Redis connection warnings (not critical - caching is disabled but everything works)
def filter_redis_warnings(record):
    return "Could not connect to Redis" not in record["message"]

logger.add(sys.stderr, level="WARNING", filter=filter_redis_warnings)  # Only show warnings and errors

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from investor_profiles import INVESTOR_PROFILES, InvestorProfile
from investor_simulator import InvestorSimulator
from investment_agent import agent_builder, config


# Load environment variables
ENV_PATH = Path(__file__).parent.parent.parent.parent.parent / ".env"
load_dotenv(ENV_PATH)

# Constants
DEFAULT_MAX_TURNS = 8
RESULTS_DIR = Path(__file__).parent / "hallucination_results"
PERSONALITIES_DIR = Path(__file__).parent.parent / "personalities"


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    turn_number: int
    user_message: str
    agent_response: str
    called_no_response: bool
    has_error: Optional[str] = None


@dataclass
class ConversationResult:
    """
    Complete result of a single automated conversation.
    
    Attributes:
        metadata: Configuration and timing information
        turns: List of conversation turns
        metrics: Performance metrics (total turns, refusal rate, etc.)
    """
    metadata: Dict[str, any]
    turns: List[Dict[str, any]]
    metrics: Dict[str, any]


class AgentFactory:
    """Factory for creating and configuring investment agents."""
    
    @staticmethod
    def create_agent(
        agent_personality: str,
        investor_profile: InvestorProfile,
        allow_hallucination: bool
    ):
        """
        Create and configure an investment agent.
        
        Args:
            agent_personality: Personality type (e.g., "friendly", "conservative", "default")
            investor_profile: The investor profile for risk tolerance context
            allow_hallucination: If True, agent may hallucinate; if False, agent refuses when lacking data
            
        Returns:
            Configured Agent instance
        """
        # Update config for hallucination mode
        config.allow_hallucination = allow_hallucination
        
        # Build agent
        agent = agent_builder.build(config)
        
        # Load agent personality from file (if not default)
        if agent_personality != "default":
            personality_file = PERSONALITIES_DIR / f"{agent_personality}.txt"
            if personality_file.exists():
                with open(personality_file) as f:
                    agent.starting_prompt = f.read()
        
        # Set user profile context for agent
        user_id = random.randint(1000, 9999)
        agent.runtime.context.update({
            "user_profile": agent.runtime.context.context["UserProfile"](
                user_id=user_id,
                risk_profile=investor_profile.risk_tolerance
            ),
        })
        
        return agent


class AgentAnalyzer:
    """Analyzes agent behavior and responses."""
    
    @staticmethod
    def called_no_response(agent) -> bool:
        """
        Check if the agent called no_response() in the last turn.
        
        The no_response() action is tracked in the dialogue history's
        system_action field.
        
        Args:
            agent: The investment agent to analyze
            
        Returns:
            True if agent called no_response() in last turn, False otherwise
        """
        if not agent.dlg_history:
            return False
        
        last_turn = agent.dlg_history[-1]
        
        if not (hasattr(last_turn, 'system_action') and last_turn.system_action):
            return False
        
        actions = getattr(last_turn.system_action, 'actions', [])
        return any(getattr(action, 'is_no_response', False) for action in actions)


class ConversationRunner:
    """Orchestrates automated conversations between investor simulator and agent."""
    
    async def run_conversation(
        self,
        investor_profile_name: str,
        agent_personality: str,
        allow_hallucination: bool,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> ConversationResult:
        """
        Run a single automated conversation and collect results.
        
        Args:
            investor_profile_name: Name of investor profile from INVESTOR_PROFILES
            agent_personality: Agent personality type (e.g., "friendly", "conservative")
            allow_hallucination: If True, agent may hallucinate; if False, agent refuses to answer instead of hallucinating when lacking data
            max_turns: Maximum number of conversation turns
            
        Returns:
            ConversationResult with full conversation log and metrics
        """
        self._print_conversation_header(investor_profile_name, agent_personality, allow_hallucination)
        
        # Initialize investor simulator
        investor_profile = INVESTOR_PROFILES[investor_profile_name]
        simulator = InvestorSimulator(investor_profile)
        
        # Initialize investment agent
        agent = AgentFactory.create_agent(agent_personality, investor_profile, allow_hallucination)
        
        # Run conversation turns
        turns: List[ConversationTurn] = []
        refusal_count = 0
        
        for turn_num in range(1, max_turns + 1):
            # Generate investor message
            if turn_num == 1:
                user_message = simulator.generate_message()
            else:
                last_agent_response = turns[-1].agent_response
                user_message = simulator.generate_message(last_agent_response)
            
            # Check if conversation should end
            if simulator.should_end_conversation(user_message):
                print(f"\n[Investor ended conversation after {turn_num - 1} turns]")
                break
            
            print(f"\nTurn {turn_num}")
            print(f"Investor: {user_message}")
            
            # Get agent response
            try:
                await agent.generate_next_turn(user_message)
                
                if agent.dlg_history:
                    agent_response = agent.dlg_history[-1].system_response
                    self._print_agent_response(agent_response)
                    
                    # Check if agent called no_response()
                    called_no_response = AgentAnalyzer.called_no_response(agent)
                    if called_no_response:
                        refusal_count += 1
                        self._print_refusal_marker(allow_hallucination)
                    
                    # Record turn
                    turns.append(ConversationTurn(
                        turn_number=turn_num,
                        user_message=user_message,
                        agent_response=agent_response,
                        called_no_response=called_no_response,
                    ))
                    
            except Exception as e:
                print(f"ERROR: {e}")
                turns.append(ConversationTurn(
                    turn_number=turn_num,
                    user_message=user_message,
                    agent_response="",
                    called_no_response=False,
                    has_error=str(e),
                ))
                break
        
        # Build result
        return self._build_result(
            investor_profile_name,
            agent_personality,
            allow_hallucination,
            turns,
            refusal_count
        )
    
    def _print_conversation_header(
        self, 
        investor_profile: str, 
        agent_personality: str, 
        allow_hallucination: bool
    ) -> None:
        """Print formatted conversation header."""
        print(f"\n{'='*70}")
        print(f"Investor Profile: {investor_profile}")
        print(f"Agent Personality: {agent_personality}")
        print(f"Hallucination Mode: {'ON (may hallucinate)' if allow_hallucination else 'OFF (refuses to answer instead of hallucinating)'}")
        print(f"{'='*70}\n")
    
    def _print_agent_response(self, response: str) -> None:
        """Print agent response."""
        print(f"\nAgent: {response}")
    
    def _print_refusal_marker(self, allow_hallucination: bool) -> None:
        """Print visual marker for no_response() detection."""
        if allow_hallucination:
            print("🟢 HALLUCINATIONS ALLOWED- calling no_response()")
        else:
            print("🔴 REFUSE TO ANSWER TO PREVENT HALLUCINATION - calling no_response()")
    
    def _build_result(
        self,
        investor_profile_name: str,
        agent_personality: str,
        allow_hallucination: bool,
        turns: List[ConversationTurn],
        refusal_count: int,
    ) -> ConversationResult:
        """Build ConversationResult from conversation data."""
        total_turns = len([t for t in turns if not t.has_error])
        
        return ConversationResult(
            metadata={
                "investor_profile": investor_profile_name,
                "agent_personality": agent_personality,
                "allow_hallucination": allow_hallucination,
                "timestamp": datetime.now().isoformat(),
            },
            turns=[asdict(turn) for turn in turns],
            metrics={
                "total_turns": total_turns,
                "refusal_count": refusal_count,
                "refusal_rate": refusal_count / total_turns if total_turns > 0 else 0,
            },
        )


class ResultsReporter:
    """Generates formatted reports from conversation results."""
    
    @staticmethod
    def print_summary(results: List[ConversationResult]) -> None:
        """
        Print comprehensive summary of all conversation results.
        
        Args:
            results: List of ConversationResult objects to analyze
        """
        print(f"\n{'='*70}")
        print("RESULTS SUMMARY")
        print(f"{'='*70}\n")
        
        # Overall statistics
        total_conversations = len(results)
        total_turns = sum(r.metrics["total_turns"] for r in results)
        total_refusals = sum(r.metrics["refusal_count"] for r in results)
        overall_refusal_rate = total_refusals / total_turns if total_turns > 0 else 0
        
        print(f"Total conversations: {total_conversations}")
        print(f"Total turns: {total_turns}")
        print(f"Total refusals: {total_refusals}")
        print(f"Overall refusal rate: {overall_refusal_rate:.1%}")
        
        # Refusal mode comparison
        ResultsReporter._print_refusal_mode_comparison(results)
        
        # By investor profile
        ResultsReporter._print_by_investor_profile(results)
        
        # By agent personality
        ResultsReporter._print_by_agent_personality(results)
    
    @staticmethod
    def _print_refusal_mode_comparison(results: List[ConversationResult]) -> None:
        """Print comparison between hallucination on/off modes."""
        print(f"\n{'='*70}")
        print("HALLUCINATION MODE COMPARISON")
        print(f"{'='*70}\n")
        
        hallucination_on = [r for r in results if r.metadata["allow_hallucination"]]
        hallucination_off = [r for r in results if not r.metadata["allow_hallucination"]]
        
        for mode_name, mode_results, hallucination_enabled in [
            ("ON (may hallucinate)", hallucination_on, True),
            ("OFF (refuses)", hallucination_off, False),
        ]:
            turns = sum(r.metrics["total_turns"] for r in mode_results)
            refusals = sum(r.metrics["refusal_count"] for r in mode_results)
            rate = refusals / turns if turns > 0 else 0
            
            print(f"Hallucination Mode {mode_name}:")
            print(f"  Conversations: {len(mode_results)}")
            print(f"  Total turns: {turns}")
            print(f"  no_response() calls: {refusals}")
            print(f"  Rate: {rate:.1%}")
            if hallucination_enabled:
                print(f"  Agent may generate plausible but unverified answers")
            else:
                print(f"  Agent refuses to answer instead of hallucinating when lacking information")
            print()
    
    @staticmethod
    def _print_by_investor_profile(results: List[ConversationResult]) -> None:
        """Print breakdown by investor profile."""
        print(f"\n{'='*70}")
        print("BY INVESTOR PROFILE")
        print(f"{'='*70}\n")
        
        profiles = set(r.metadata["investor_profile"] for r in results)
        for profile in sorted(profiles):
            profile_results = [r for r in results if r.metadata["investor_profile"] == profile]
            turns = sum(r.metrics["total_turns"] for r in profile_results)
            refusals = sum(r.metrics["refusal_count"] for r in profile_results)
            rate = refusals / turns if turns > 0 else 0
            
            print(f"{profile}:")
            print(f"  Refusal rate: {rate:.1%} ({refusals}/{turns} turns)")
    
    @staticmethod
    def _print_by_agent_personality(results: List[ConversationResult]) -> None:
        """Print breakdown by agent personality."""
        print(f"\n{'='*70}")
        print("BY AGENT PERSONALITY")
        print(f"{'='*70}\n")
        
        personalities = set(r.metadata["agent_personality"] for r in results)
        for personality in sorted(personalities):
            personality_results = [r for r in results if r.metadata["agent_personality"] == personality]
            turns = sum(r.metrics["total_turns"] for r in personality_results)
            refusals = sum(r.metrics["refusal_count"] for r in personality_results)
            rate = refusals / turns if turns > 0 else 0
            
            print(f"{personality}:")
            print(f"  Refusal rate: {rate:.1%} ({refusals}/{turns} turns)")
    
    @staticmethod
    def save_results(results: List[ConversationResult], output_path: Path) -> None:
        """
        Save results to JSON file.
        
        Args:
            results: List of ConversationResult objects
            output_path: Path to output JSON file
        """
        output_data = [asdict(r) for r in results]
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*70}\n")


async def main():
    """
    Main entry point for automated conversation testing.
    
    Runs conversations across a test matrix of:
    - Investor profiles
    - Agent personalities  
    - Hallucination modes
    
    Results are saved to timestamped JSON files and summary statistics
    are printed to console.
    """
    # Configure test matrix
    investor_profiles_to_test = [
        "hallucination_inducer",  # Most likely to trigger edge cases
    ]
    
    agent_personalities_to_test = [
        "friendly",
    ]
    

    hallucination_modes = [False, True]  # False = refuses, True = may hallucinate
    
    # Number of scenarios to test
    total_scenarios = (
        len(investor_profiles_to_test) * 
        len(agent_personalities_to_test) * 
        len(hallucination_modes)
    )
    
    print(f"\n{'='*70}")
    print("AUTOMATED CONVERSATION TESTING")
    print(f"{'='*70}")
    print(f"Investor Profiles: {len(investor_profiles_to_test)}")
    print(f"Agent Personalities: {len(agent_personalities_to_test)}")
    print(f"Hallucination Modes: {len(hallucination_modes)}")
    print(f"Total Scenarios: {total_scenarios}")
    print(f"{'='*70}")
    
    # Run all test combinations
    runner = ConversationRunner()
    all_results: List[ConversationResult] = []
    scenario_num = 0
    
    for investor_profile in investor_profiles_to_test:
        for agent_personality in agent_personalities_to_test:
            for allow_hallucination in hallucination_modes:
                scenario_num += 1
                print(f"\n[Scenario {scenario_num}/{total_scenarios}]")
                
                result = await runner.run_conversation(
                    investor_profile_name=investor_profile,
                    agent_personality=agent_personality,
                    allow_hallucination=allow_hallucination,
                    max_turns=DEFAULT_MAX_TURNS,
                )
                
                all_results.append(result)
    
    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"conversation_test_{timestamp}.json"
    
    ResultsReporter.save_results(all_results, output_file)
    
    # Print summary
    ResultsReporter.print_summary(all_results)


if __name__ == "__main__":
    asyncio.run(main())

