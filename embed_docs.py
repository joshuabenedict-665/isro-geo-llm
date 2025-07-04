import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Folder containing your plain text documentation files
DOCS_PATH = "docs"
VECTORSTORE_DIR = "faiss_index"

# Embedding model (can be changed)
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Collect all text documents
documents = []
for root, _, files in os.walk(DOCS_PATH):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            try:
                loader = TextLoader(file_path, encoding='utf-8')  # 👈 Forces UTF-8 decoding
                documents.extend(loader.load())
                print(f"✅ Loaded: {file_path}")
            except Exception as e:
                print(f"❌ Failed to load {file_path}: {e}")

# Split the documents into chunks
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# Create FAISS vector store and save to disk
print("🔄 Creating vector index...")
vectorstore = FAISS.from_documents(chunks, embedding_model)
vectorstore.save_local(VECTORSTORE_DIR)
print(f"✅ Vector store saved to {VECTORSTORE_DIR}")
