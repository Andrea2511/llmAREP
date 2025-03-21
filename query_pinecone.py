import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Cargar variables de entorno
load_dotenv()

# Inicializar Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("langchain-test-index")

# Usar el mismo modelo de embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Crear la base de datos de vectores en Pinecone
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Consulta de ejemplo
query = "¿Qué dice sobre inteligencia artificial?"

# Realizar búsqueda de similitud
results = vector_store.similarity_search(query, k=3)

# Mostrar resultados
print("🔍 Resultados de la búsqueda:")
for i, res in enumerate(results):
    print(f"{i+1}. {res.page_content} (Fuente: {res.metadata['source']})")
