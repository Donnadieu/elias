"""
RAG System Demo

This script demonstrates how to use the RAG system with the PersonalAssistantAgent.
It shows how to add documents to the knowledge base and query them.
"""
import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path

# Add the project root to the Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from agents.personal_assistant_agent import PersonalAssistantAgent

# Load environment variables
load_dotenv()

async def main():
    """Demonstrate the RAG system with example data."""
    print("Initializing RAG system...")
    
    # Initialize the assistant
    assistant = PersonalAssistantAgent(
        name="RAG Demo Assistant",
        model_name="gpt-4",
        rag_collection="demo_knowledge"
    )
    
    # Example documents to add to the knowledge base
    documents = [
        """
        Python is a high-level, interpreted programming language known for its simplicity and readability. 
        It was created by Guido van Rossum and first released in 1991.
        """,
        """
        Machine learning is a subset of artificial intelligence that focuses on building systems 
        that learn from data and improve their performance over time without being explicitly programmed.
        """,
        """
        The Python Standard Library includes modules for working with files, regular expressions, 
        dates and times, and many other common programming tasks.
        """,
        """
        Deep learning is a type of machine learning that uses neural networks with many layers 
        to model complex patterns in data.
        """,
        """
        Pandas is a popular Python library for data manipulation and analysis, 
        particularly for working with structured data like spreadsheets and SQL tables.
        """
    ]
    
    print("\nAdding documents to the knowledge base...")
    for i, doc in enumerate(documents, 1):
        doc_id = await assistant.add_document(
            doc.strip(),
            {
                "source": "demo",
                "doc_id": f"doc_{i}",
                "type": "example"
            }
        )
        print(f"- Added document {i} with ID: {doc_id}")
    
    # Example queries
    queries = [
        "What is Python?",
        "What is machine learning?",
        "What is the Python Standard Library?",
        "What is deep learning?",
        "What is Pandas used for?"
    ]
    
    print("\nQuerying the knowledge base...")
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 80)
        
        # Get search results
        results = await assistant.rag.search(query, top_k=2)
        
        # Print search results
        print("Top matching documents:")
        for i, result in enumerate(results, 1):
            print(f"{i}. Score: {result['score']:.4f}")
            print(f"   Text: {result['text'][:150]}...")
            print()
        
        # Generate a response using RAG
        response = await assistant.process_message(query)
        print("Generated Response:")
        print(f"{response}")
        print("=" * 80)
    
    print("\nDemo complete!")

if __name__ == "__main__":
    asyncio.run(main())
