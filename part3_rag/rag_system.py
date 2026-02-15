from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load docs
with open("part3_rag/docs.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split text into chunks
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_text(text)

# Create embeddings
emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Build vector database
db = FAISS.from_texts(chunks, emb)

print("\nWarehouse RAG System Ready")

# Query loop
while True:
    query = input("\nAsk robot (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    docs = db.similarity_search(query, k=2)

    print("\nRetrieved Instructions:\n")
    for d in docs:
        print("-", d.page_content)
