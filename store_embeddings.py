import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from uuid import uuid4

# Cargar variables de entorno
load_dotenv()

# Inicializar Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("langchain-test-index")

# 🔹 Usar modelo de embeddings con 1536 dimensiones
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Crear la base de datos de vectores en Pinecone
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Documentos de prueba para almacenar
documents = [
    Document(page_content="Hoy hace un día soleado y muy caluroso.", metadata={"source": "noticia"}),
    Document(page_content="Los perros son animales muy leales y cariñosos.", metadata={"source": "blog"}),
    Document(page_content="La inteligencia artificial está revolucionando la tecnología.", metadata={"source": "artículo"}),
]

# Generar IDs únicos y almacenar los documentos en Pinecone
uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)

print("✅ Embeddings almacenados con éxito en Pinecone.")
