"""
Test script to verify semantic definitions are properly formatted in LLM prompts.

This script tests that the enhanced semantic metadata is:
1. Properly loaded into the ToolRegistry
2. Correctly formatted by _format_existing_apis() and _format_db_schema()
3. Included in the LLM prompt

Run: python test_semantic_definitions.py
"""

from tool_discovery import ToolRegistry, ToolAnalyzer


def test_api_semantic_loading():
    """Test that API semantic metadata is loaded correctly"""
    print("=" * 80)
    print("TEST 1: API Semantic Metadata Loading")
    print("=" * 80)

    registry = ToolRegistry()
    registry._initialize_existing_tools()

    # Check GetRecommendation has semantic metadata
    rec_api = registry.existing_apis.get("GetRecommendation")
    assert rec_api is not None, "GetRecommendation API not found"

    # Check for semantic fields
    assert "capabilities" in rec_api, "Missing 'capabilities' field"
    assert "can_do" in rec_api["capabilities"], "Missing 'can_do' list"
    assert "cannot_do" in rec_api["capabilities"], "Missing 'cannot_do' list"
    assert "when_to_use" in rec_api, "Missing 'when_to_use' field"
    assert "example_queries" in rec_api, "Missing 'example_queries' field"

    print("✓ GetRecommendation has all semantic fields")
    print(f"  - CAN DO items: {len(rec_api['capabilities']['can_do'])}")
    print(f"  - CANNOT DO items: {len(rec_api['capabilities']['cannot_do'])}")
    print(f"  - Example queries: {len(rec_api['example_queries'])}")

    # Sample one CANNOT DO item
    cannot_do = rec_api["capabilities"]["cannot_do"]
    assert any("contribution" in item.lower() for item in cannot_do), \
        "Should mention contributions in CANNOT DO"
    print(f"  - Example CANNOT DO: {cannot_do[0]}")

    print("\n✅ TEST 1 PASSED\n")


def test_db_semantic_loading():
    """Test that DB table semantic metadata is loaded correctly"""
    print("=" * 80)
    print("TEST 2: DB Table Semantic Metadata Loading")
    print("=" * 80)

    registry = ToolRegistry()
    registry._initialize_existing_tools()

    # Check fidelity_funds has semantic metadata
    funds_table = registry.existing_kb_tables.get("fidelity_funds")
    assert funds_table is not None, "fidelity_funds table not found"

    # Check for semantic fields
    assert "columns" in funds_table, "Missing 'columns' field"
    assert "temporal_coverage" in funds_table, "Missing 'temporal_coverage' field"
    assert "semantics" in funds_table, "Missing 'semantics' field"
    assert "common_use_cases" in funds_table, "Missing 'common_use_cases' field"

    print("✓ fidelity_funds has all semantic fields")

    # Check temporal coverage
    tc = funds_table["temporal_coverage"]
    assert tc["earliest_year"] == 2015, "Wrong earliest year"
    assert tc["latest_year"] == 2024, "Wrong latest year"
    print(f"  - Temporal coverage: {tc['earliest_year']}-{tc['latest_year']}")

    # Check semantics
    sem = funds_table["semantics"]
    assert "can_answer" in sem, "Missing 'can_answer' list"
    assert "cannot_answer" in sem, "Missing 'cannot_answer' list"
    print(f"  - CAN ANSWER items: {len(sem['can_answer'])}")
    print(f"  - CANNOT ANSWER items: {len(sem['cannot_answer'])}")

    # Sample one CANNOT ANSWER item
    cannot_answer = sem["cannot_answer"]
    assert any("2015" in item or "before" in item.lower() for item in cannot_answer), \
        "Should mention historical data limitation"
    print(f"  - Example CANNOT ANSWER: {cannot_answer[0]}")

    print("\n✅ TEST 2 PASSED\n")


def test_api_formatting():
    """Test that _format_existing_apis() shows semantic info"""
    print("=" * 80)
    print("TEST 3: API Formatting for LLM Prompt")
    print("=" * 80)

    registry = ToolRegistry()
    registry._initialize_existing_tools()

    # Format APIs
    formatted = ToolAnalyzer._format_existing_apis(registry.existing_apis)

    # Check that semantic info is included
    assert "✓ CAN DO:" in formatted, "Missing CAN DO section"
    assert "✗ CANNOT DO:" in formatted, "Missing CANNOT DO section"
    assert "When to use:" in formatted, "Missing 'When to use' field"
    assert "Example queries" in formatted, "Missing example queries"

    print("✓ Formatted APIs contain semantic information")
    print(f"  - Total formatted length: {len(formatted)} chars")

    # Check specific content for GetRecommendation
    assert "GetRecommendation" in formatted, "Missing GetRecommendation"
    assert "allocation percentages" in formatted.lower(), \
        "Missing capability description"
    assert "contributions" in formatted.lower(), \
        "Missing CANNOT DO item about contributions"

    print(f"  - Contains 'GetRecommendation' section: ✓")
    print(f"  - Contains CAN DO/CANNOT DO lists: ✓")

    # Show sample of formatted output
    print("\n  Sample formatted output:")
    lines = formatted.split("\n")
    rec_start = next(i for i, line in enumerate(lines) if "GetRecommendation" in line)
    for line in lines[rec_start:rec_start+10]:
        print(f"    {line}")

    print("\n✅ TEST 3 PASSED\n")


def test_db_formatting():
    """Test that _format_db_schema() shows semantic info"""
    print("=" * 80)
    print("TEST 4: DB Schema Formatting for LLM Prompt")
    print("=" * 80)

    registry = ToolRegistry()
    registry._initialize_existing_tools()

    # Format DB schema
    formatted = ToolAnalyzer._format_db_schema(registry.existing_kb_tables)

    # Check that semantic info is included
    assert "Data available:" in formatted, "Missing temporal coverage"
    assert "CAN ANSWER:" in formatted, "Missing CAN ANSWER section"
    assert "CANNOT ANSWER:" in formatted, "Missing CANNOT ANSWER section"
    assert "Common queries:" in formatted, "Missing common queries"

    print("✓ Formatted DB schema contains semantic information")
    print(f"  - Total formatted length: {len(formatted)} chars")

    # Check specific content for fidelity_funds
    assert "fidelity_funds" in formatted, "Missing fidelity_funds"
    assert "2015-2024" in formatted, "Missing temporal range"
    assert "WHERE" in formatted, "Missing SQL example queries"

    print(f"  - Contains temporal coverage: ✓")
    print(f"  - Contains CAN/CANNOT ANSWER lists: ✓")
    print(f"  - Contains SQL query examples: ✓")

    # Show sample of formatted output
    print("\n  Sample formatted output:")
    lines = formatted.split("\n")
    funds_start = next(i for i, line in enumerate(lines) if "fidelity_funds" in line)
    for line in lines[funds_start:funds_start+15]:
        print(f"    {line}")

    print("\n✅ TEST 4 PASSED\n")


def test_prompt_contains_classification_rules():
    """Test that LLM prompt contains classification rules"""
    print("=" * 80)
    print("TEST 5: LLM Prompt Contains Classification Rules")
    print("=" * 80)

    registry = ToolRegistry()
    registry._initialize_existing_tools()

    analyzer = ToolAnalyzer()
    prompt = analyzer._build_prompt("Test query", registry, None)

    # Check for classification rules
    assert "CRITICAL CLASSIFICATION RULES" in prompt, "Missing classification rules section"
    assert "✗ DON'T create:" in prompt, "Missing negative examples"
    assert "✓ DO create:" in prompt, "Missing positive examples"
    assert "Composite Workflows" in prompt, "Missing composite workflow guidance"
    assert "ATOMIC Gaps Only" in prompt, "Missing atomic gap guidance"
    assert "CLASSIFICATION DECISION TREE" in prompt, "Missing decision tree"

    print("✓ LLM prompt contains classification rules")
    print(f"  - Total prompt length: {len(prompt)} chars")
    print(f"  - Contains decision tree: ✓")
    print(f"  - Contains examples: ✓")
    print(f"  - Contains atomic gap guidance: ✓")

    # Check for specific examples
    assert "get_aggressive_funds" in prompt, "Missing example about DB queries"
    assert "investment_goal_planning" in prompt, "Missing example about composite tools"

    print(f"  - Contains specific examples from documentation: ✓")

    print("\n✅ TEST 5 PASSED\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("SEMANTIC DEFINITIONS TEST SUITE")
    print("=" * 80 + "\n")

    try:
        test_api_semantic_loading()
        test_db_semantic_loading()
        test_api_formatting()
        test_db_formatting()
        test_prompt_contains_classification_rules()

        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nSemantic definitions are properly integrated.")
        print("The LLM will now have full context to make better classification decisions.")
        print("\nNext steps:")
        print("1. Run the investment agent with test queries")
        print("2. Verify that monolithic tools are no longer created")
        print("3. Check that atomic tools are created correctly")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
