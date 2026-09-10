import os
import time
from strands import Agent, tool
from strands.models.gemini import GeminiModel

# 1. Define custom tools for travel utilities
@tool
def estimate_budget_breakdown(budget_inr: float) -> str:
    """Calculates a recommended budget split in INR for Stay, Food, Transport, and Activities."""
    stay = budget_inr * 0.40
    food = budget_inr * 0.25
    transport = budget_inr * 0.20
    activities = budget_inr * 0.15
    return (
        f"Suggested Budget Breakdown (Total: ₹{budget_inr:,.2f}):\n"
        f"- Accommodation (40%): ₹{stay:,.2f}\n"
        f"- Food & Dining (25%): ₹{food:,.2f}\n"
        f"- Transportation (20%): ₹{transport:,.2f}\n"
        f"- Activities & Entry Fees (15%): ₹{activities:,.2f}"
    )

# 2. Configure Gemini Model with supported model_id
gemini_model = GeminiModel(
    model_id="gemini-2.5-flash",
    client_args={"api_key": os.environ.get("GOOGLE_API_KEY")}
)

# 3. Initialize the Travel Planner Agent
SYSTEM_PROMPT = """
You are an expert AI Travel Planner.
When a user provides a destination, duration, and budget, help them plan a comprehensive trip.

Always include the following structured sections:
1. Budget Breakdown (Use the `estimate_budget_breakdown` tool to calculate this)
2. Places to Visit
3. Accommodation Suggestions
4. Transportation Options
5. Food & Local Cuisine Recommendations
6. Day-wise Detailed Itinerary

Provide realistic, well-paced recommendations suited to the given budget.
"""

agent = Agent(
    model=gemini_model,
    system_prompt=SYSTEM_PROMPT,
    tools=[estimate_budget_breakdown]
)

# 4. Interactive Agent Loop with Retry Logic
def start_chat_loop():
    print("=" * 60)
    print("✈️  Welcome to the AI Travel Planner!")
    print("Type 'exit' or 'quit' to end the session.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("\nYou 👤: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("\nSafe travels! Goodbye! 👋")
                break

            print("\nTravel Planner 🧳:\n")
            
            # Simple retry wrapper for transient server busy errors
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    agent(user_input)
                    break
                except Exception as e:
                    if "503" in str(e) or "UNAVAILABLE" in str(e):
                        if attempt < max_retries - 1:
                            print(f"\n⚠️ Gemini server busy. Retrying in 3 seconds... (Attempt {attempt + 1}/{max_retries})")
                            time.sleep(3)
                        else:
                            print("\n❌ Gemini API is currently experiencing extreme traffic. Please try again shortly.")
                    else:
                        raise e
            print()

        except KeyboardInterrupt:
            print("\nSession ended. Safe travels!")
            break

if __name__ == "__main__":
    start_chat_loop()