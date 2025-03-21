import os
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener claves de API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# Verificar que las claves se cargaron correctamente
print("OpenAI API Key:", OPENAI_API_KEY[:5] + "..." if OPENAI_API_KEY else "No encontrada")
print("Pinecone API Key:", PINECONE_API_KEY[:5] + "..." if PINECONE_API_KEY else "No encontrada")
print("Pinecone Environment:", PINECONE_ENVIRONMENT if PINECONE_ENVIRONMENT else "No encontrado")
