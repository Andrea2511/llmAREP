import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from uuid import uuid4

# Cargar variables de entorno
load_dotenv()

# Inicializar Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("langchain-test-index")

# Inicializar modelo de embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Crear el vector store
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Función para cargar documentos de un directorio
def load_documents(directory):
    docs = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        print(f"📂 Procesando archivo: {file}")

        try:
            if file.endswith(".pdf"):
                print("🔍 Intentando cargar PDF...")
                loader = PyMuPDFLoader(file_path)
            elif file.endswith(".txt"):
                print("🔍 Intentando cargar TXT...")
                loader = TextLoader(file_path)
            else:
                print("⏭️ Archivo ignorado (no es PDF ni TXT).")
                continue
            
            documents = loader.load()
            if not documents:
                print(f"⚠️ {file} no tiene contenido legible.")
            else:
                print(f"✅ {file} cargado con {len(documents)} documentos.")
            docs.extend(documents)
        except Exception as e:
            print(f"❌ Error al cargar {file}: {e}")
    
    return docs


# Cargar documentos desde la carpeta 'data'
directory = "data"
documents = load_documents(directory)

# Dividir documentos en fragmentos
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

print(f"📊 Se generaron {len(chunks)} fragmentos para indexar.")

if not chunks:
    print("❌ No hay fragmentos para indexar. Revisa la carga de documentos.")
    exit()

# Guardar en Pinecone con IDs únicos
uuids = [str(uuid4()) for _ in range(len(chunks))]
vector_store.add_documents(documents=chunks, ids=uuids)

print(f"✅ {len(chunks)} fragmentos indexados en Pinecone.")
