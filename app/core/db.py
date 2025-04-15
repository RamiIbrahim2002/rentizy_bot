from chromadb import PersistentClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os
from dotenv import load_dotenv
load_dotenv()
chroma_client = PersistentClient(path="data/chroma")
embedding_function = OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

conversation_collection = chroma_client.get_or_create_collection(
    name="conversation_chunks",
    embedding_function=embedding_function
)
