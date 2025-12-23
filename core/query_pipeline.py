from datetime import datetime
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from .tech_query import process_tech_query
import json

# ===================== LOAD ENV =====================
ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

if not API_KEY or not AZURE_ENDPOINT or not CHAT_DEPLOYMENT:
    raise RuntimeError("Azure OpenAI config missing. Check .env location.")

# ===================== LLM =====================
llm = AzureChatOpenAI(
    azure_deployment=CHAT_DEPLOYMENT,
    api_key=API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    openai_api_version=API_VERSION,
    temperature=0
)

router_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are an intelligent routing assistant. Your task is to classify the user's query as one of the following intents:

- GENERAL: Greetings, chitchat, small talk, out-of-domain, irrelevant, or casual questions.
- TECHNICAL: Queries that are related to any labour law and likely need technical documentation or expert knowledge.

Respond strictly in this JSON format:
{{"intent": "<intent_type>"}}

Query: {query}
"""
)

import json

def route_query(llm, query: str) -> str:
    chain = router_prompt | llm
    result = chain.invoke({"query": query})

    try:
        data = json.loads(result.content.strip())
        return data.get("intent", "GENERAL")
    except json.JSONDecodeError:
        # Fail-safe fallback
        return "GENERAL"

def process_query(query, state, law_type, chat_history, session_id):
    print(chat_history)
    # Route the query
    intent = route_query(llm, f"{query} for {state}")

    print("🔀 Routed intent:", intent)

    if intent == "GENERAL":
        response = llm.invoke(
            f"""
If the user greets, reply with a polite greeting.
If the query is casual, irrelevant, or outside labour laws,
politely decline.

User query:
{query}
"""
        )
        return response.content

    elif intent == "TECHNICAL":
        return process_tech_query(query, state, law_type, chat_history)

    return "Unable to determine query intent."
