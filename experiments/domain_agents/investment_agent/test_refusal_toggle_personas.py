"""Test the refusal toggle with different user and agent personas.

This script runs conversations with:
- 1 user profile (hallucination_inducer)
- 1 agent personality (friendly)
- 2 refusal modes (on/off)

"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(env_path)

from llm_user_simulator import LLMUserSimulator, USER_PROFILES
from investment_agent import agent_builder, config
import random


async def run_single_conversation(
    user_profile_name: str,
    agent_personality: str,
    refusal_mode: bool,
    max_turns: int = 8,
) -> dict:
    """Run a single conversation and track hallucination behavior.

    Args:
        user_profile_name: Name of user profile to use
        agent_personality: Agent personality (aggressive/friendly/conservative)
        refusal_mode: True = agent refuses, False = agent hallucinates
        max_turns: Maximum conversation turns

    Returns:
        Dictionary with conversation data and metrics
    """
    print(f"\n{'='*70}")
    print(f"User: {user_profile_name}")
    print(f"Agent: {agent_personality}")
    print(f"Refusal: {'ON (agent refuses)' if refusal_mode else 'OFF (agent hallucinates)'}")
    print(f"{'='*70}\n")

    # Update config for this run
    config.refusal = refusal_mode

    # Initialize user simulator
    user_profile = USER_PROFILES[user_profile_name]
    llm_user = LLMUserSimulator(user_profile, model="gpt-4.1")

    # Initialize agent
    agent = agent_builder.build(config)

    # Load agent personality
    if agent_personality != "default":
        personality_file = Path(__file__).parent / "personalities" / f"{agent_personality}.txt"
        if personality_file.exists():
            with open(personality_file) as f:
                agent.starting_prompt = f.read()

    # Set user profile for agent
    user_id = random.randint(1000, 9999)
    agent.runtime.context.update({
        "user_profile": agent.runtime.context.context["UserProfile"](
            user_id=user_id, risk_profile=user_profile.risk_tolerance
        ),
    })

    # Run conversation
    conversation_log = []
    no_response_count = 0

    for turn in range(1, max_turns + 1):
        # Get user message
        if turn == 1:
            user_message = llm_user.generate_next_message()
        else:
            last_agent_response = conversation_log[-1]["agent_response"]
            user_message = llm_user.generate_next_message(last_agent_response)

        # Check if done
        if llm_user.is_done(user_message):
            print(f"\n[User ended conversation after {turn - 1} turns]")
            break

        print(f"\nTurn {turn}")
        print(f"User: {user_message}")

        # Get agent response
        try:
            await agent.generate_next_turn(user_message)

            if agent.dlg_history:
                agent_response = agent.dlg_history[-1].system_response
                print(f"Agent: {agent_response[:150]}{'...' if len(agent_response) > 150 else ''}")

                # Check if no_response was called
                no_response_called = check_no_response(agent)

                if no_response_called:
                    no_response_count += 1
                    marker = "🔴 NO RESPONSE" if refusal_mode else "🟡 HALLUCINATION"
                    print(f"{marker} - no_response() called!")

                conversation_log.append({
                    "turn": turn,
                    "user_message": user_message,
                    "agent_response": agent_response,
                    "no_response_called": no_response_called,
                })

        except Exception as e:
            print(f"ERROR: {e}")
            conversation_log.append({
                "turn": turn,
                "user_message": user_message,
                "error": str(e),
            })
            break

    # Generate summary
    total_turns = len([t for t in conversation_log if "error" not in t])

    return {
        "metadata": {
            "user_profile": user_profile_name,
            "agent_personality": agent_personality,
            "refusal_mode": refusal_mode,
            "timestamp": datetime.now().isoformat(),
        },
        "conversation": conversation_log,
        "metrics": {
            "total_turns": total_turns,
            "no_response_count": no_response_count,
            "no_response_rate": no_response_count / total_turns if total_turns > 0 else 0,
        },
    }


def check_no_response(agent) -> bool:
    """Check if no_response() was called in the last turn."""
    if not agent.dlg_history:
        return False

    last_turn = agent.dlg_history[-1]

    if hasattr(last_turn, 'system_action') and last_turn.system_action is not None:
        actions_list = getattr(last_turn.system_action, 'actions', [])
        for action in actions_list:
            if getattr(action, 'is_no_response', False):
                return True

    return False


async def main():
    """Run comprehensive refusal toggle testing."""

    # Test configurations
    user_profiles_to_test = [
        "hallucination_inducer",  # This one should trigger many no_response() calls
    ]

    agent_personalities_to_test = [
        "friendly",
    ]

    refusal_modes = [False, True]  # False = hallucinate, True = refuse

    # Run all combinations
    all_results = []
    total_scenarios = len(user_profiles_to_test) * len(agent_personalities_to_test) * len(refusal_modes)
    current = 0

    print(f"\n{'='*70}")
    print(f"REFUSAL TOGGLE TESTING")
    print(f"{'='*70}")
    print(f"User Profiles: {len(user_profiles_to_test)}")
    print(f"Agent Personalities: {len(agent_personalities_to_test)}")
    print(f"Refusal Modes: {len(refusal_modes)}")
    print(f"Total Scenarios: {total_scenarios}")
    print(f"{'='*70}")

    for user_profile in user_profiles_to_test:
        for agent_personality in agent_personalities_to_test:
            for refusal_mode in refusal_modes:
                current += 1
                print(f"\n[{current}/{total_scenarios}] Running scenario...")

                result = await run_single_conversation(
                    user_profile_name=user_profile,
                    agent_personality=agent_personality,
                    refusal_mode=refusal_mode,
                    max_turns=8,
                )

                all_results.append(result)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "refusal_toggle_results"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"refusal_toggle_test_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate summary report
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}\n")

    total_turns = sum(r["metrics"]["total_turns"] for r in all_results)
    total_no_response = sum(r["metrics"]["no_response_count"] for r in all_results)

    print(f"Total conversations: {len(all_results)}")
    print(f"Total turns: {total_turns}")
    print(f"Total no_response() calls: {total_no_response}")
    print(f"Overall no_response rate: {total_no_response/total_turns:.1%}")

    # Compare refusal modes
    print(f"\n{'='*70}")
    print(f"REFUSAL MODE COMPARISON")
    print(f"{'='*70}\n")

    hallucination_results = [r for r in all_results if not r["metadata"]["refusal_mode"]]
    refusal_results = [r for r in all_results if r["metadata"]["refusal_mode"]]

    h_turns = sum(r["metrics"]["total_turns"] for r in hallucination_results)
    h_no_response = sum(r["metrics"]["no_response_count"] for r in hallucination_results)

    r_turns = sum(r["metrics"]["total_turns"] for r in refusal_results)
    r_no_response = sum(r["metrics"]["no_response_count"] for r in refusal_results)

    print(f"HALLUCINATION MODE (refusal=False):")
    print(f"  Conversations: {len(hallucination_results)}")
    print(f"  Total turns: {h_turns}")
    print(f"  no_response() calls: {h_no_response}")
    print(f"  Rate: {h_no_response/h_turns:.1%}")
    print(f"  → Agent generates plausible answers when lacking tools")

    print(f"\nREFUSAL MODE (refusal=True):")
    print(f"  Conversations: {len(refusal_results)}")
    print(f"  Total turns: {r_turns}")
    print(f"  no_response() calls: {r_no_response}")
    print(f"  Rate: {r_no_response/r_turns:.1%}")
    print(f"  → Agent says 'I cannot answer'")

    # By user profile
    print(f"\n{'='*70}")
    print(f"BY USER PROFILE")
    print(f"{'='*70}\n")

    for user_profile in user_profiles_to_test:
        profile_results = [r for r in all_results if r["metadata"]["user_profile"] == user_profile]
        p_turns = sum(r["metrics"]["total_turns"] for r in profile_results)
        p_no_response = sum(r["metrics"]["no_response_count"] for r in profile_results)
        p_rate = p_no_response / p_turns if p_turns > 0 else 0

        print(f"{user_profile}:")
        print(f"  no_response rate: {p_rate:.1%} ({p_no_response}/{p_turns} turns)")

    # By agent personality
    print(f"\n{'='*70}")
    print(f"BY AGENT PERSONALITY")
    print(f"{'='*70}\n")

    for agent_personality in agent_personalities_to_test:
        personality_results = [r for r in all_results if r["metadata"]["agent_personality"] == agent_personality]
        a_turns = sum(r["metrics"]["total_turns"] for r in personality_results)
        a_no_response = sum(r["metrics"]["no_response_count"] for r in personality_results)
        a_rate = a_no_response / a_turns if a_turns > 0 else 0

        print(f"{agent_personality}:")
        print(f"  no_response rate: {a_rate:.1%} ({a_no_response}/{a_turns} turns)")

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    asyncio.run(main())
