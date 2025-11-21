"""Test LLM column identification"""
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load env
ENV_PATH = Path(__file__).parent.parent.parent.parent.parent / ".env"
load_dotenv(ENV_PATH)

client = AzureOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    api_version=os.getenv("LLM_API_VERSION"),
    azure_endpoint=os.getenv("LLM_API_ENDPOINT"),
)

# Test the exact prompt used in _identify_requested_column
table = "fidelity_bonds"
available_columns = [
    "Description", "Coupon", "Coupon Frequency", "Maturity Date",
    "Moody's Rating", "S&P Rating", "Expected Price", "Expected Yield",
    "Call Protected", "Offering Period", "Settlement Date", "Attributes"
]
utterance = "What is the expense ratio for bond CUSIP 12345?"

prompt = f"""Given this user query about {table} data:
"{utterance}"

And these available columns:
{", ".join(available_columns)}

What specific column or field is the user asking about? Respond with ONLY the column name, or "NONE" if the query doesn't ask for a specific column.

Examples:
Query: "What's the expense ratio for fund XYZ?" → expenseRatio
Query: "Show me the beta" → ratings_beta3Year
Query: "What's the Moody's rating?" → Moody's Rating
Query: "Tell me about this fund" → NONE (general query)

Column name:"""

print("Sending prompt to LLM...")
print("="*70)
print(prompt)
print("="*70)

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0,
    max_tokens=50
)

result = response.choices[0].message.content.strip()
print(f"\nLLM Response: '{result}'")

cleaned = result.strip('"\'')
first_word = cleaned.split()[0] if cleaned else ""

print(f"Stripped quotes: '{cleaned}'")
print(f"First word: '{first_word}'")
print(f"\nExpected: 'expense_ratio' or 'expenseRatio' or similar")
print(f"Got: '{result}'")
