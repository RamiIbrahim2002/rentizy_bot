from chromadb import logger
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from datetime import datetime
from app.helpers.helpers import determine_attribute_from_query
from app.core.db import conversation_collection  # Shared Chroma collection instance.
from app.storage import load_conversation, save_conversation
from app.core.llm import (
    call_openai_rag,
    call_openai_should_save,
    call_openai_should_answer,
    call_openai_check_relevance
)

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import chromadb
from dotenv import load_dotenv
import os
load_dotenv()
# Setup Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma")
collection = chroma_client.get_or_create_collection(
    name="support_conversations",
    embedding_function=OpenAIEmbeddingFunction(api_key=os.environ.get("OPENAI_API_KEY"))
)

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def get_chat_ui():
    return FileResponse("static/chat_interface.html")

def get_last_n_messages(history_str, n=5):
    lines = [line.strip() for line in history_str.strip().split('\n') if line.strip()]
    return lines[-n:]

@app.post("/api/send")
async def handle_message(request: Request):
    data = await request.json()
    message = data.get("message", "").strip()
    role = data.get("role", "").lower()
    user_id = data.get("user_id")

    if not message or not role or not user_id:
        return {"error": "Invalid input"}

    conversation_id = f"conv-{user_id}"
    chat_history_text = load_conversation(conversation_id)
    timestamp = datetime.utcnow().isoformat()
    entry = f"[{role.upper()}] {message}"
    updated_history = f"{chat_history_text}\n{entry}".strip()
    save_conversation(conversation_id, updated_history)
    
    response_data = {"status": "ok", "history": updated_history}

    if role == "tenant":
        decision = call_openai_should_answer(message, chat_history_text)
        if decision.get("answer", False):
# After retrieving documents from Chroma:
            results = conversation_collection.query(
                query_texts=[message],
                n_results=10,
                include=["documents", "metadatas", "distances"]
            )
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            # Determine the target attribute from the query:
            target_attribute = determine_attribute_from_query(message)
            logger.info(f"Filtering documents for attribute: {target_attribute}")

            # Filter candidates with metadata attribute equal to target:
            filtered_candidates = []
            for doc, meta, dist in zip(documents, metadatas, distances):
                doc_attr = meta.get("attribute", "general")
                logger.info(f"Candidate: attribute={doc_attr}, timestamp={meta.get('timestamp')}, distance={dist}")
                if doc_attr == target_attribute:
                    filtered_candidates.append((doc, meta, dist))

            # If no candidate matching the attribute, fallback to all candidates.
            if not filtered_candidates:
                logger.warning(f"No candidate documents match attribute '{target_attribute}'. Falling back to all candidates.")
                # Here the fallback is still "all candidates"
                filtered_candidates = list(zip(documents, metadatas, distances))
            
            # Assume filtered_candidates is a list of tuples: (doc, meta, distance)
            if not filtered_candidates:
                logger.warning(f"No candidate documents match attribute '{target_attribute}'.")
                # Handle the error or return a message indicating no stored information
                response_data["status"] = "ignored"
                response_data["reason"] = f"No documents available for attribute '{target_attribute}'."
                return JSONResponse(content=response_data)

            # Compute normalized scores for each candidate (similar to your previous code)
            current_epoch = datetime.utcnow().timestamp()
            candidate_distances = [cand[2] for cand in filtered_candidates]
            min_dist = min(candidate_distances) if candidate_distances else 0
            max_dist = max(candidate_distances) if candidate_distances else 0
            normalize_distance = lambda x: (x - min_dist) / (max_dist - min_dist) if (max_dist - min_dist) > 0 else 0

            candidate_times = []
            for _, meta, _ in filtered_candidates:
                try:
                    ts = datetime.fromisoformat(meta.get("timestamp")).timestamp()
                except Exception:
                    ts = 0
                candidate_times.append(ts)
            min_time = min(candidate_times) if candidate_times else current_epoch
            max_time = max(candidate_times) if candidate_times else current_epoch
            normalize_time = lambda t: (t - min_time) / (max_time - min_time) if (max_time - min_time) > 0 else 1

            alpha = 0.5  # weight for similarity (inverted distance)
            beta = 0.5   # weight for recency

            scored_candidates = []
            for cand in filtered_candidates:
                doc, meta, dist = cand
                norm_dist = normalize_distance(dist)
                try:
                    doc_time = datetime.fromisoformat(meta.get("timestamp")).timestamp()
                except Exception:
                    doc_time = 0
                norm_time = normalize_time(doc_time)
                final_score = alpha * (1 - norm_dist) + beta * norm_time
                scored_candidates.append((final_score, cand))
                logger.info(f"Candidate score: final_score={final_score:.3f}, norm_distance={norm_dist:.3f}, norm_time={norm_time:.3f}")

            # Sort the candidates by final score in descending order
            scored_candidates.sort(key=lambda x: x[0], reverse=True)

            # Choose the top 3 candidates (or fewer if less are available)
            top_n = 3
            top_candidates = [item[1] for item in scored_candidates[:top_n]]
            for idx, (doc, meta, dist) in enumerate(top_candidates):
                logger.info(f"Top candidate {idx+1}: {doc.strip()} (added on {meta.get('timestamp', 'unknown')})")

            # Combine these documents into a context string (with appropriate separators)
            combined_context = "\n\n".join(
                f"{doc.strip()} (added on {meta.get('timestamp', 'unknown')})"
                for doc, meta, _ in top_candidates
            )

            logger.info(f"Combined context for RAG:\n{combined_context}")

            # Pass the combined context to your LLM call for generating the answer.
            relevance_check = call_openai_check_relevance(query=message, context=combined_context)
            if relevance_check.get("is_relevant", False):
                assistant_reply = call_openai_rag(query=message, context=combined_context)
                response_data["assistant"] = assistant_reply
            else:
                response_data["status"] = "ignored"
                response_data["reason"] = relevance_check.get("reason", "Retrieved context is not relevant.")

            return JSONResponse(content=response_data)

    # In the owner branch we process messages via upsert (see below)
    elif role == "owner":
        from app.services.knowledge import process_owner_message
        result = process_owner_message(
            message=message,
            history=chat_history_text,
            user_id=user_id,
            conversation_id=conversation_id
        )
        response_data["saved"] = (result["status"] in ["saved", "updated"])

    return JSONResponse(content=response_data)