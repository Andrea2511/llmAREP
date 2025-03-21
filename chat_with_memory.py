import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Cargar variables de entorno
load_dotenv()

# Inicializar Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("langchain-test-index")

# Modelos de embeddings y LLM
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
llm = ChatOpenAI(model_name="gpt-4o")

# Crear la base de datos de vectores en Pinecone
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Crear memoria para recordar conversaciones
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Crear la cadena de conversación con memoria
chatbot = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    memory=memory
)

# Interacción en tiempo real
while True:
    query = input("\n🟢 Tú: ")
    if query.lower() in ["salir", "exit", "quit"]:
        print("🔴 Chatbot finalizado.")
        break
    
    response = chatbot.run({"question": query})
    
    print("\n🤖 Chatbot:", response)
