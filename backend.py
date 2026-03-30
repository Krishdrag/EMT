from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub  # or use OpenAI
import os

DB_DIR = "vectorstore"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


db = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_kwargs={"k": 3}

# Option 1: HuggingFace (Free, slower)
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0.5, "max_length": 512}
)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

def answer_query(query):
    result = qa_chain({"query": query})
    
    answer = result["result"]
    sources = result["source_documents"]

    return answer, sources



if __name__ == "__main__":
    print("🤖 RAG Chatbot Ready! Type 'exit' to quit\n")
    
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        
        answer, sources = answer_query(query)
        
        print("\nBot:", answer)
        print("\n📚 Sources:")
        for i, doc in enumerate(sources):
            print(f"{i+1}. {doc.metadata.get('source', 'Unknown')}")
        print("-" * 50)
