import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Cargar las variables de entorno
load_dotenv()

# Obtener la API Key
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Inicializar Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Nombre del índice
index_name = "langchain-test-index"

# Verificar si el índice existe, si no, crearlo
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,  # Dimensión del modelo de embeddings de OpenAI
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # Ajusta la región
    )

# Conectar al índice
index = pc.Index(index_name)

print(f"Índices disponibles: {pc.list_indexes()}")