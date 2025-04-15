# app/services/knowledge.py
from datetime import datetime
import logging
from app.core.llm import call_openai_should_save
from app.core.db import conversation_collection as chroma_collection  
from app.helpers.helpers import determine_attribute, call_openai_merge_info

logger = logging.getLogger(__name__)

def should_update(existing_doc, new_timestamp):
    existing_ts = existing_doc.get("timestamp")
    if not existing_ts:
        return True
    return new_timestamp > existing_ts

def process_owner_message(message, history="", user_id=None, conversation_id=None):
    decision = call_openai_should_save(message, history)
    if decision["action"] == "ignore":
        logger.info("Owner update ignored: " + decision["reason"])
        return {"status": "ignored", "reason": decision["reason"]}
    
    new_content = decision["content_to_save"]
    new_ts = datetime.utcnow().isoformat()
    
    attribute = determine_attribute(new_content)
    logger.info(f"Processing update for attribute '{attribute}' with content: '{new_content}' at {new_ts}")
    
    # Query candidate documents for this attribute.
    results = chroma_collection.query(
        query_texts=[attribute],
        n_results=10,
        include=["documents", "metadatas"]
    )
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    
    # Check if there is an existing document with this attribute.
    for doc, meta in zip(docs, metadatas):
        if meta.get("attribute") == attribute:
            doc_id = meta.get("doc_id")
            logger.info(f"Found existing document for attribute '{attribute}' with doc_id {doc_id}.")
            if doc_id is None:
                logger.warning("Document id is missing; skipping candidate.")
                continue
            
            # If the new update is not exactly the same as the stored content:
            if doc.strip().lower() != new_content.strip().lower():
                # Call the LLM to merge the existing document with the new update.
                merged_content = call_openai_merge_info(existing_doc=doc, new_update=new_content)
                logger.info(f"Merged content: {merged_content}")
                chroma_collection.update(
                    ids=[doc_id],
                    documents=[merged_content],
                    metadatas=[{
                        "timestamp": new_ts,
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "attribute": attribute,
                        "doc_id": doc_id
                    }]
                )
                logger.info(f"Updated (merged) document with doc_id {doc_id} for attribute '{attribute}'.")
                return {"status": "updated", "doc_id": doc_id}
            else:
                # If the new update is identical, you might just update the timestamp.
                chroma_collection.update(
                    ids=[doc_id],
                    documents=[doc],
                    metadatas=[{
                        "timestamp": new_ts,
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "attribute": attribute,
                        "doc_id": doc_id
                    }]
                )
                logger.info(f"Document for attribute '{attribute}' (doc_id {doc_id}) is already up-to-date.")
                return {"status": "already_up_to_date", "doc_id": doc_id}
    
    # No candidate exists, so add a new document.
    new_id = f"doc-{datetime.now().timestamp()}"
    chroma_collection.add(
        documents=[new_content],
        metadatas=[{
            "timestamp": new_ts,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "attribute": attribute,
            "doc_id": new_id
        }],
        ids=[new_id]
    )
    logger.info(f"Added new document with doc_id {new_id} for attribute '{attribute}'.")
    return {"status": "saved", "doc_id": new_id}
