#!/usr/bin/env python
"""
Personal Assistant MVP using AutoGen's AgentChat framework.
This script provides a simple CLI interface for interacting with the PersonalAssistantAgent.
"""
import asyncio
import os
import sys
import logging
from agents.personal_assistant_agent import PersonalAssistantAgent
from llm.deepseek_client import get_llm_config

# Try to load environment variables if dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using environment variables directly.")
    # Continue without dotenv


def print_welcome_message(assistant_name: str = "Assistant"):
    """Print welcome message and usage instructions."""
    print("=" * 60)
    print(f"Welcome to {assistant_name} - Your AI Assistant with RAG")
    print("=" * 60)
    print("This assistant can help you with various tasks using Retrieval-Augmented Generation (RAG).")
    print("It can remember information and use it to provide better responses.")
    print("\nUsage:")
    print("  - Ask any question and I'll try to answer using my knowledge")
    print("  - Type 'remember that [information]' to store information in memory")
    print("  - Type 'exit' to quit the program")
    print("\nExamples:")
    print("  - 'What is the capital of France?'")
    print("  - 'Explain how a neural network works'")
    print("  - 'Generate a Python function to sort a list of dictionaries by a key'")
    print("  - 'remember that the capital of France is Paris'")
    print("  - 'remember: John's favorite color is blue'")
    print("=" * 60)
    print()


async def main():
    """Main function to run the personal assistant with RAG capabilities."""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('personal_assistant')
    
    # Check if required environment variables are set
    required_vars = ["QDRANT_API_KEY", "QDRANT_HOST_URL"]
    
    # Check for either DeepSeek or OpenAI credentials
    has_deepseek = all([os.environ.get(var) for var in ["DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"]])
    has_openai = os.environ.get("OPENAI_API_KEY")
    
    if not (has_deepseek or has_openai):
        required_vars.extend(["DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "OPENAI_API_KEY"])
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print("\nError: Missing required environment variables!")
        for var in missing_vars:
            print(f"- {var} is not set")
        print("\nPlease set these variables in your .env file or environment:")
        print("Required for vector database:")
        print("- QDRANT_API_KEY: Your Qdrant Cloud API key")
        print("- QDRANT_HOST_URL: Your Qdrant Cloud host URL")
        print("\nRequired for LLM (set either DeepSeek OR OpenAI):")
        print("- DEEPSEEK_API_KEY: Your DeepSeek API key (recommended for development)")
        print("- DEEPSEEK_BASE_URL: Your DeepSeek API base URL")
        print("- DEEPSEEK_MODEL: Your DeepSeek model name (optional)")
        print("OR")
        print("- OPENAI_API_KEY: Your OpenAI API key")
        print("\nYou can copy the .env.example file to .env and add your credentials.")
        return
    
    # Check if Qdrant API key and host URL are set
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    qdrant_host_url = os.environ.get("QDRANT_HOST_URL")
    if not qdrant_api_key or not qdrant_host_url:
        print("\nError: Qdrant API key or host URL not found!")
        print("Please set your Qdrant credentials using one of the following methods:")
        print("1. Create a .env file in the project root with QDRANT_API_KEY and QDRANT_HOST_URL")
        print("2. Set the QDRANT_API_KEY and QDRANT_HOST_URL environment variables")
        print("\nYou can copy the .env.example file to .env and add your Qdrant credentials.")
        return
    
    # Default assistant name (can be overridden by knowledge base)
    default_assistant_name = "Elias"
    
    try:
        print("Initializing AI Assistant with RAG capabilities...")
        
        # Determine which LLM provider to use
        use_deepseek = bool(os.environ.get("DEEPSEEK_API_KEY") and os.environ.get("DEEPSEEK_BASE_URL"))
        llm_provider = "DeepSeek" if use_deepseek else "OpenAI"
        logger.info(f"Using {llm_provider} as the LLM provider")
        
        # Initialize the assistant with Qdrant for RAG
        assistant = PersonalAssistantAgent(
            name=default_assistant_name,
            system_message="""You are a helpful AI assistant. 
            Always introduce yourself with your name when appropriate.
            You can help with coding, answer questions, and remember information from the conversation.
            Be friendly, professional, and concise in your responses.""",
            qdrant_api_key=qdrant_api_key,
            qdrant_host_url=qdrant_host_url,
            rag_collection="assistant_knowledge",
            use_deepseek=use_deepseek
        )
        
        # Get the assistant's name from knowledge base if available
        assistant_name = await assistant.get_assistant_name()
        
        # Ensure the assistant's name is stored in the knowledge base
        if assistant_name == default_assistant_name:
            await assistant._store_assistant_info()
        
        # Print welcome message with the assistant's name
        print_welcome_message(assistant_name)
        
        # Pre-load some initial knowledge if the collection is empty
        try:
            # Check if we have any documents in the knowledge base
            results = await assistant.rag.search("test", top_k=1)
            if not results:
                print("Initializing knowledge base with some basic information...")
                initial_knowledge = [
                    "The capital of France is Paris.",
                    "Python is a high-level, interpreted programming language.",
                    "The Earth revolves around the Sun in approximately 365.25 days.",
                    "The human body is composed of approximately 60% water.",
                    "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data."
                ]
                
                for fact in initial_knowledge:
                    await assistant.add_document(fact, {"type": "general_knowledge", "source": "system"})
                
                print("Knowledge base initialized with basic information.\n")
                
        except Exception as e:
            print(f"Warning: Could not initialize knowledge base: {str(e)}")
        
        # Main interaction loop
        while True:
            try:
                # Get user input
                try:
                    user_input = input("You: ").strip()
                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    print(f"Error reading input: {str(e)}")
                    continue
                
                # Check for exit command
                if user_input.lower() in ('exit', 'quit', 'bye'):
                    print(f"\n{assistant_name}: Goodbye!")
                    break
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Process the message using RAG
                print(f"\n{assistant_name}: ", end="", flush=True)
                
                try:
                    response = await assistant.process_message(user_input)
                    print(response + "\n")
                except Exception as e:
                    print(f"I encountered an error: {str(e)}\n")
                
            except Exception as e:
                print(f"\nAn error occurred: {str(e)}\n")
                
    except KeyboardInterrupt:
        print("\nGoodbye! Have a great day!")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        # Clean up resources
        if hasattr(assistant, 'close'):
            await assistant.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Exiting...")
        sys.exit(0)
