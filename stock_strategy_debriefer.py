import os
import logging
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.callbacks import get_openai_callback


#Test Prompt
#Pullback to the 50-day EMA at $220. Lower volume on the pullback. Stochastic crossed bullishly out of oversold (<20). However, tech sector ETF (XLK) is showing negative momentum divergence on the 4-hour chart.

# 1. Logging and Environment Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY missing! Add it to your .env file.")

# 2. Core Model Declaration
# Low temperature enforces deterministic, grounded analysis without hallucinations
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# 3. Prompts for Worker Tools
bullish_prompt = PromptTemplate(
    input_variables=["market_context"],
    template="""You are a senior quantitative momentum analyst.
Analyze the following asset context and technical setup:
{market_context}

Provide a structured Bullish Thesis:
- Core Trend & Key Support Levels
- Momentum & Volume Confirmations (e.g., Stochastic divergence, EMA bounces)
- Upside Target Zones
Keep it objective and focused purely on upside confirmation."""
)

risk_prompt = PromptTemplate(
    input_variables=["market_context", "bullish_thesis"],
    template="""You are a strict quantitative risk auditor.
Original Market Context: {market_context}
Bullish Thesis to Audit: {bullish_thesis}

Perform a rigorous Risk Stress-Test:
- Hidden Vulnerabilities (Bearish divergence, overhead supply zones, market breadth weakness)
- Hard Invalidation Level (Exact price/condition that kills the bullish thesis)
- Recommended Stop-Loss Placement and downside danger zones
Ruthlessly challenge every optimistic assumption."""
)

# 4. Tool Definitions with Descriptive Docstrings
@tool
def draft_bullish_thesis(market_context: str) -> str:
    """Evaluates asset indicators, price structure, and technical setups to construct a bullish thesis.
    Input must be the raw market notes/technical context."""
    logging.info("--> Executing Tool: draft_bullish_thesis")
    chain = bullish_prompt | llm
    return chain.invoke({"market_context": market_context}).content

@tool
def stress_test_risk_factors(market_context: str, bullish_thesis: str) -> str:
    """Audits and stress-tests a bullish setup against downside traps, overhead supply, and invalidation triggers.
    Requires both the market context and the drafted bullish thesis."""
    logging.info("--> Executing Tool: stress_test_risk_factors")
    chain = risk_prompt | llm
    return chain.invoke({
        "market_context": market_context,
        "bullish_thesis": bullish_thesis
    }).content

tools = [draft_bullish_thesis, stress_test_risk_factors]

# 5. System Persona & Agent Initialization
system_prompt = """You are an elite Market Strategy & Risk Debriefer.
Follow the ReAct (Reason, Act, Observe) framework:
1. Reason: Parse the provided ticker, technical indicators, and price action.
2. Act: Call 'draft_bullish_thesis' to isolate upside drivers and key support.
3. Act: Pass both the context and the bullish result into 'stress_test_risk_factors' to identify invalidation levels and downside traps.
4. Synthesize: Produce an executive debrief with strict non-redundant sections:
   - Setup Snapshot: 2 sentences summarizing technical posture and sector backdrop.
   - Core Tension: The single biggest conflict between the upside drivers and downside risks.
   - Actionable Triggers: Confirmation entry trigger vs. Hard Invalidation line (exact price).
   - Final Stance: [Aggressive Long / Cautious Long / Neutral-Wait / Avoid] with 1-sentence rationale."""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)

# 6. Runner Function & Interactive Loop
def run_strategy_debriefer(user_notes: str) -> str:
    with get_openai_callback() as cb:
        response = agent.invoke({
            "messages": [("user", f"Run a full multi-perspective debrief on this setup:\n{user_notes}")]
        })
        print(f"\n[Usage: Total Tokens: {cb.total_tokens} | Prompt: {cb.prompt_tokens} | Completion: {cb.completion_tokens} | Cost: ${cb.total_cost:.5f}]")
    return response["messages"][-1].content

def main():
    print("=" * 60)
    print("📈 Multi-Perspective Market & Strategy Debriefer Agent")
    print("Type 'q', 'quit', or 'exit' to stop.")
    print("=" * 60 + "\n")

    while True:
        user_input = input("Enter Asset & Setup Notes: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["q", "quit", "exit"]:
            print("Session ended. Happy trading!")
            break

        print("\n[Agent active: Reasoning and running tools...]\n")
        try:
            report = run_strategy_debriefer(user_input)
            print("\n" + "#" * 60)
            print(report)
            print("#" * 60 + "\n")
        except Exception as err:
            logging.error(f"Error during execution: {err}")

if __name__ == "__main__":
    main()