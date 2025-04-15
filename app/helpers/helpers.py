# helpers.py
import openai
from app.logger import logger
import json
from openai import OpenAI

# Use the existing client from your LLM module or initialize it here
client = OpenAI()

def log_llm_input(label: str, payload: dict):
    try:
        formatted = json.dumps(payload, indent=2, ensure_ascii=False)
    except Exception:
        formatted = str(payload)
    logger.info(f"[LLM INPUT] {label} Payload:\n{formatted}")

def infer_attribute_from_text(text: str) -> str:
    system_prompt = """
You are an assistant that categorizes property-related text.
Analyze the given sentence and extract the primary feature axis it mentions.
The allowed outputs are: rooms, amenities, appliances, location, price, neighbors ,general.
Use 'general' only if none of the other specific features are mentioned.
Respond with exactly one word.
"""
    user_prompt = f"Sentence: \"{text}\".\nAnswer with one word from the allowed list."
    
    try:
        log_llm_input("Attribute Inference", {"system": system_prompt, "user": user_prompt})
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using the desired model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        
        raw_attribute = response.choices[0].message.content.strip().lower()
        logger.info(f"Raw attribute response: '{raw_attribute}'")
        
        allowed = ["rooms", "amenities", "appliances", "location", "price","neighbors"]
        # Return one of the allowed values if it is present; otherwise, default to 'general'
        if raw_attribute in allowed:
            attribute = raw_attribute
        else:
            attribute = "general"
            
    except Exception as e:
        logger.error(f"LLM call for attribute inference failed: {e}")
        attribute = "general"
    
    return attribute

def determine_attribute_from_query(query: str) -> str:
    """
    Infer the attribute from a tenant query using the LLM.
    """
    attribute = infer_attribute_from_text(query)
    logger.info(f"Inferred attribute from query '{query}': {attribute}")
    return attribute

def determine_attribute(content: str) -> str:
    """
    Infer the attribute from an owner's content update using the LLM.
    """
    attribute = infer_attribute_from_text(content)
    logger.info(f"Inferred attribute from content: {attribute}")
    return attribute


def call_openai_merge_info(existing_doc: str, new_update: str) -> str:
    """
    Use an LLM to merge the existing document with the new update.
    The merged version should preserve details from the existing document 
    while incorporating new information or changes from the new_update.
    """
    system_prompt = """
You are an assistant that consolidates property update information.
Given an existing summary and a new update, produce a single, coherent summary 
that includes all relevant details. If the new update contradicts or adds details, 
incorporate these changes into the summary.
Respond with the merged summary.
"""
    user_prompt = (
        f"Existing document: \"{existing_doc}\"\n"
        f"New update: \"{new_update}\"\n"
        "Provide a merged summary that retains previous details and includes the new update."
    )
    
    try:
        logger.info("Calling LLM to merge info")
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using the desired model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        merged = response.choices[0].message.content.strip()
        logger.info(f"Merged result: {merged}")
        return merged
    except Exception as e:
        logger.error(f"Merge LLM call failed: {e}")
        # Fallback: If LLM call fails, just concatenate the documents.
        return existing_doc + "\n" + new_update
