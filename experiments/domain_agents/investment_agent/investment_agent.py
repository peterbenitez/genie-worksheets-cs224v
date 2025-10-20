import asyncio
import os
import random
import uuid

from worksheets import (
    AgentBuilder,
    Config,
    SUQLReActParser,
    SUQLKnowledgeBase,
    conversation_loop,
)
from worksheets.agent.builder import TemplateLoader
from worksheets.agent.config import agent_api

risk_profile_to_rating = {
    "conservative": 5,
    "moderate": 4,
    "balanced": 3,
    "bold": 2,
    "aggressive": 1,
}


@agent_api("get_recommendation_api", "Get a recommendation for an investment")
def get_recommendation_api(risk_profile: str, value_to_invest: float):
    """
    Provides investment recommendations based on the given risk profile.

    Args:
        risk_profile (str): The investor's risk appetite. Must be one of the following:
                            'conservative', 'moderate', 'balanced', 'bold', or 'aggressive'.
        value_to_invest (float): The amount of money the investor wants to allocate.

    Returns:
        list of tuples: Each tuple contains:
            - The recommended fund (str)
            - Allocation percentage (int)
            - Allocated amount (float)

    Raises:
        ValueError: If the risk profile is not recognized.
    """
    print(value_to_invest, risk_profile)
    if risk_profile not in risk_profile_to_rating:
        raise ValueError(
            f"Invalid risk profile. Choose from: {', '.join(risk_profile_to_rating.keys())}. You provided: {risk_profile}"
        )

    selected_products = knowledge_base.run(
        f'SELECT * FROM fidelity_funds WHERE "ratings_morningstarRisk" = {risk_profile_to_rating[risk_profile]} ORDER BY "ratings_morningstarOverall" DESC LIMIT 3'
    )
    selected_products = [product["name"] for product in selected_products]

    # Generate random allocation percentages ensuring they sum to 100%
    allocations = sorted([random.randint(20, 60) for _ in range(2)])
    allocations.append(100 - sum(allocations))

    # Compute allocated values
    allocation_data = {
        "products": selected_products,
        "allocations": allocations,
        "allocated_values": [
            round(value_to_invest * (allocations[i] / 100), 2) for i in range(3)
        ],
    }
    print(allocation_data)

    return allocation_data


@agent_api("get_account_balance_api", "Get the account balance")
def get_account_balance_api():
    """
    Simulates the amount of money currently available in the client's account balance.

    Returns:
        float: A randomly sampled account balance between 0 and 250,000 BRL.
    """
    return round(random.uniform(0, 250000), 2)


@agent_api("investment_portfolio_api", "Investment portfolio for a client")
def investment_portfolio_api():
    """
    Simulates an investment portfolio for a client.

    Returns:
        dict: A dictionary containing:
            - 'total_investment': Randomly sampled total investment value between 200 and 500,000 BRL.
            - 'portfolio': List of up to 7 randomly selected investment products with allocation percentages and values.
    """
    total_investment = round(random.uniform(200, 500000), 2)

    # Randomly select up to 7 investment products
    num_products = random.randint(1, 7)
    selected_products = knowledge_base.run(
        f"SELECT name FROM fidelity_funds ORDER BY random() LIMIT {num_products}"
    )
    selected_products = [product["name"] for product in selected_products]

    # Generate random allocation percentages ensuring they sum to 100% and each is between 5 and 40
    allocation_percentages = []
    remaining = 100
    for i in range(num_products - 1):
        min_val = max(5, remaining - 40 * (num_products - i - 1))
        max_val = min(40, remaining - 5 * (num_products - i - 1))
        val = random.randint(min_val, max_val)
        allocation_percentages.append(val)
        remaining -= val
    allocation_percentages.append(remaining)  # The last allocation

    # Compute allocated values
    portfolio = [
        {
            "product": selected_products[i],
            "percentage": allocation_percentages[i],
            "allocated_value": round(
                total_investment * (allocation_percentages[i] / 100), 2
            ),
        }
        for i in range(num_products)
    ]

    return {"total_investment": total_investment, "portfolio": portfolio}


@agent_api("cd_investment_api", "Process a CD investment")
def cd_investment_api(bond_allocations: list):
    """
    Simulates the hiring process of a Bank Deposit Certificate (CD).

    Args:
        bond_allocations (list): A list of dictionaries where each dictionary contains the bond name and the exact amount to invest in each bond.

    Returns:
        dict: A dictionary containing the transaction status, total invested, and allocation details.
    """
    if not bond_allocations:
        raise ValueError("At least one bond must be provided for investment.")

    total_invested = [bond["investment_amount"] for bond in bond_allocations]

    for bond in bond_allocations:
        if bond["investment_amount"] <= 0:
            raise ValueError(
                f"Investment amount for {bond['bond_name']} must be greater than zero."
            )

    return {
        "transaction_status": "Success",
        "total_invested": round(total_invested, 2),
        "allocations": bond_allocations,
    }


@agent_api("fund_investment_api", "Process a fund investment")
def fund_investment_api(fund_allocations):
    """
    Simulates the investment process in specified funds with exact allocation amounts.

    Args:
        fund_allocations (FundAllocation): GenieWorksheet type (FundAllocation) contains the fund name and the exact amount to invest in each fund.

    Returns:
        dict: A dictionary containing the transaction status, total invested, and allocation details.
    """
    if not fund_allocations:
        raise ValueError("At least one fund must be provided for investment.")

    return {
        "transaction_status": "Success",
        "transaction_id": uuid.uuid4(),
        "investment_amount": fund_allocations.investment_amount.value,
        "allocations": {
            "symbol": fund_allocations.fund_to_invest_in[0]["symbol"],
            "name": fund_allocations.fund_to_invest_in[0]["name"],
        },
    }


current_dir = os.path.dirname(os.path.realpath(__file__))


config = Config.load_from_yaml(os.path.join(current_dir, "config.yaml"))

# Initialize a standalone knowledge base instance so API functions can
# execute queries without depending on the global `agent` variable. This
# allows the APIs to work in both the terminal and Chainlit front-ends.
knowledge_base = SUQLKnowledgeBase(
    config.knowledge_base,
    tables_with_primary_keys={
        "fidelity_funds": "id",
    },
    database_name="banco_itau",
    embedding_server_address="http://127.0.0.1:8509",
    db_username="select_user",
    db_password="select_user",
    db_host=os.getenv("DB_HOST", "127.0.0.1"),
    db_port=os.getenv("DB_PORT", "5432"),
)

starting_prompt = TemplateLoader.load(
    os.path.join(current_dir, "starting_prompt.md"), format="jinja2"
)

agent_builder = (
    AgentBuilder(
        name="Fidelity Funds Agent",
        description="You are a helpful assistant that can help with the following tasks: 1. Investment portfolio management 2. CD investment 3. Fund investment 4. Get account balance 5. Get recommendation for investment",
        starting_prompt=starting_prompt.render(),
    )
    .with_knowledge_base(
        SUQLKnowledgeBase,
        tables_with_primary_keys={
            "fidelity_funds": "id",
        },
        database_name="banco_itau",
        embedding_server_address="http://127.0.0.1:8509",
        db_username="select_user",
        db_password="select_user",
        db_host=os.getenv("DB_HOST", "127.0.0.1"),
        db_port=os.getenv("DB_PORT", "5432"),
    )
    .with_parser(
        # DatatalkParser,
        # domain="banco_itau",
        # api_key=os.getenv("DATATALK_API"),
        SUQLReActParser,
        example_path=os.path.join(current_dir, "examples.txt"),
        instruction_path=os.path.join(current_dir, "instructions.txt"),
        table_schema_path=os.path.join(current_dir, "table_schema.txt"),
    )
    .with_gsheet_specification("18dEfdpdHQxuT6nvvBBCMtmj7cy4sF5Jlnfpt66YxSIA")
)

if __name__ == "__main__":
    user_id = random.randint(1000, 9999)
    user_risk_profile = random.choice(
        ["conservative", "moderate", "balanced", "bold", "aggressive"]
    )
    print(f"User ID: {user_id}, Risk Profile: {user_risk_profile}")
    agent = agent_builder.build(config)

    # Use agent as context manager for automatic logging and conversation saving
    with agent:
        agent.runtime.context.update(
            {
                "user_profile": agent.runtime.context.context["UserProfile"](
                    user_id=user_id, risk_profile=user_risk_profile
                ),
            }
        )
        asyncio.run(conversation_loop(agent, debug=True))
