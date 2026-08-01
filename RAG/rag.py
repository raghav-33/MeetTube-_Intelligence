from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
load_dotenv()

class AdvancedHybridRAG:
    """
    Production-grade RAG utility implementing Hybrid Search (Dense + Sparse)
    and FlashRank Cross-Encoder Reranking.
    """
    def __init__(self):
        
        self.embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3"
)
        self.vector_store = None
        self.compression_retriever = None
        
    
    # Transcript Ingestion 
    def ingest_transcript(self, transcript: str):
        """ Processes text into chunks, builds dense/sparse indices, and sets up reranking."""
        print("Starting advanced ingestion pipeline...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=120,
            length_function=len
        )
        
        # Wrapping raw text in a Document so LangChain's pipeline doesn't crash
        raw_doc = Document(page_content=transcript, metadata={"source": "media_transcript"})
        chunks = text_splitter.split_documents([raw_doc])
        print(f"Generated {len(chunks)} highly contextual text segments.")

        # Dense Index (Chroma)
        self.vector_store = Chroma.from_documents(chunks, self.embeddings)
        dense_retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})

        # Sparse Index (BM25)
        sparse_retriever = BM25Retriever.from_documents(chunks)
        sparse_retriever.k = 10

        # Hybrid Ensemble (50/50 weighting)
        hybrid_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=[0.5, 0.5]
        )

        # Reranker (FlashRank)
        compressor = FlashrankRerank()
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=hybrid_retriever
        )
        print(" Advanced Hybrid RAG pipeline successfully compiled in-memory.")
        
        
    # Retriving and Reranking
    def retrieve_and_rerank(self, query: str, final_k: int = 4) -> str:
        """Retrieves, reranks, and formats the top context blocks as a clean string."""
        if not self.compression_retriever:
            return "No content ingested yet. Vector database is empty."

        print(f"Executing hybrid search + FlashRank rerank for query: '{query}'")
        retrieved_docs = self.compression_retriever.invoke(query)
        final_docs = retrieved_docs[:final_k]
        
        context_blocks = [
            f"[Chunk {i}]:\n{doc.page_content}\n" 
            for i, doc in enumerate(final_docs, 1)
        ]
        return "\n".join(context_blocks)


