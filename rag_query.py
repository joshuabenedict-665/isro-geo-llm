from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama  # Updated import

# Load vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# Load your local model (running in Ollama)
llm = Ollama(model="geo-llama")  # or "llama3" if that's what you're using

# 🔧 Custom Prompt Template
prompt_template = PromptTemplate.from_template("""
You are a helpful and accurate GIS assistant. Use the following documentation context to answer the user's geospatial question clearly and precisely. If the answer is not available, say "Sorry, I don't have enough information from the provided documents."

Context:
{context}

Question:
{question}

Answer:
""")

# Build RetrievalQA chain with custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

# --- Interactive loop ---
while True:
    try:
        query = input("\n🔎 Ask a geospatial question (Ctrl+C to exit):\n> ")
        result = qa_chain.invoke({"query": query})

        print("\n🧠 Answer:\n", result['result'])
        print("\n📚 Source Documents:")
        for doc in result['source_documents']:
            print("- " + doc.metadata["source"])

    except KeyboardInterrupt:
        print("\nExiting...")
        break
