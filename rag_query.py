import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Cargar variables de entorno
load_dotenv()

# Inicializar Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("langchain-test-index")

# Inicializar modelos
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
llm = ChatOpenAI(model_name="gpt-4o")

# Crear el vector store
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Definir un prompt personalizado para el LLM
prompt_template = PromptTemplate(
    template="Usa la siguiente información para responder la pregunta:\n{context}\n\nPregunta: {question}\nRespuesta:",
    input_variables=["context", "question"],
)

# Configurar la cadena RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    chain_type_kwargs={"prompt": prompt_template}
)

# Pregunta de prueba
query = input("🟢 Ingresa tu pregunta: ")
response = qa_chain.run(query)

print("\n🤖 Respuesta del Chatbot:")
print(response)
