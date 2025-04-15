# app/core/llm.py
import os
import json
from openai import OpenAI
from app.logger import logger
from app.helpers.helpers import log_llm_input
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))

def call_openai_should_save(message: str, recent_messages: list, history: str = ""):
    system_prompt = """
You are an assistant that decides whether a message from the PROPERTY OWNER should be saved for future reference by an AI assistant.

You will be given:
- current_message: the owner's most recent message.
- recent_messages: a short conversation history between the tenant and the owner.

Your task is to:
1. Determine whether the owner has confirmed or denied or provided specific, factual, and actionable information about the property in their current message, based on the context in recent_messages.
2. If yes, reconstruct the confirmed information as a complete, standalone factual sentence. You must infer it from the context, but do not hallucinate or assume beyond what is clearly implied.
3. If no property-related fact is confirmed, choose "ignore".

Respond strictly in this JSON format:
{
  "action": "save" | "ignore",
  "reason": "...",
  "content_to_save": "..."  // A clear, factual statement if saving, otherwise empty
}
"""
    user_input = {
        "current_message": message,
        "recent_messages": recent_messages,
    }
    log_llm_input("should_save()", user_input)
    response = client.chat.completions.create(
        model="o3-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_input)}
        ]
    )
    logger.info(f"[LLM] Raw should_save() response: {response.choices[0].message.content}")
    return json.loads(response.choices[0].message.content)

def call_openai_rag(query: str, context: str) -> str:
    system_prompt = """
You are a helpful assistant for a property rental platform.
Use the provided conversation history to answer the tenant's question.
Be concise, friendly, and only use relevant information from the context.
Use only the most recent statements. If older statements contradict newer ones, trust the newest.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Tenant's question: {query}\n\nContext:\n{context}"}
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    logger.info(f"[LLM] Raw RAG response: {response.choices[0].message.content}")
    return response.choices[0].message.content

def call_openai_should_answer(message: str, history: str = ""):
    system_prompt = """
You are an assistant that decides whether a tenant's message should be processed using RAG (retrieval-augmented generation).
Only return 'true' if the message contains a clear question or request that may require stored information.
resonate with the conversation history {history}
Respond in JSON like this:
{
  "answer": true | false,
  "reason": "..."
}
"""
    user_input = {
        "message": message,
        "conversation_history": history
    }
    log_llm_input("should_answer()", user_input)
    response = client.chat.completions.create(
        model="o3-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_input)}
        ]
    )
    logger.info(f"[LLM] Raw should_answer() response: {response.choices[0].message.content}")
    return json.loads(response.choices[0].message.content)

def call_openai_check_relevance(query: str, context: str) -> dict:
    system_prompt = """
You are an assistant that determines if retrieved documents are SPECIFICALLY relevant enough to answer a tenant's question about a rental property.

You will be given:
1. A tenant's question or request
2. Retrieved documents from a knowledge base

Your task is to determine if the retrieved documents EXPLICITLY address the specific topic of the tenant's question.
Be very strict in your evaluation - the documents must contain information that DIRECTLY relates to the specific topic being asked about.

Respond in JSON like this:
{
  "is_relevant": true | false,
  "reason": "Brief explanation of your decision",
  "topic_mentioned": true | false
}
"""
    user_input = f"""
Tenant's question: {query}

Retrieved documents:
{context}

Important: Verify that the documents SPECIFICALLY and EXPLICITLY address the question about {query.strip('?').strip()}. 
Generic responses without clear context are not considered relevant.
"""
    log_llm_input("check_relevance()", {"query": query, "context": context})
    response = client.chat.completions.create(
        model="o3-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        response_format={"type": "json_object"},
    )
    logger.info(f"[LLM] Raw relevance check response: {response.choices[0].message.content}")
    return json.loads(response.choices[0].message.content)
