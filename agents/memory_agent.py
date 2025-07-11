"""
Memory agent for the personal assistant.
"""
import os
from typing import Dict, Any, List, Optional
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Import the vector store
from memory.vector_store import CloudMemoryStore

# Try to load environment variables if dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using environment variables directly.")
    # Continue without dotenv


class MemoryAgent:
    """
    A wrapper class for the AutoGen AssistantAgent that can store and recall memories.
    """
    
    def __init__(
        self,
        name: str = "memory_agent",
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        system_message: Optional[str] = None,
        collection_name: str = "assistant_memories",
        qdrant_api_key: Optional[str] = None,
        qdrant_host_url: Optional[str] = None,
        auto_recall: bool = True,
        auto_recall_threshold: float = 0.3
    ):
        """
        Initialize the MemoryAgent.
        
        Args:
            name (str): Name of the agent
            model_name (str): Name of the model to use
            api_key (Optional[str]): OpenAI API key (if None, will use environment variable)
            system_message (Optional[str]): System message for the agent (defaults to memory-aware message)
            collection_name (str): Name of the Qdrant collection to use
            qdrant_api_key (Optional[str]): Qdrant API key (if None, will use environment variable)
            qdrant_host_url (Optional[str]): Qdrant host URL (if None, will use environment variable)
            auto_recall (bool): Whether to automatically recall relevant memories
            auto_recall_threshold (float): Threshold for including auto-recalled memories (0.0 to 1.0)
        """
        # Create model client
        model_client_kwargs = {}
        if api_key:
            model_client_kwargs["api_key"] = api_key
            
        self.model_client = OpenAIChatCompletionClient(
            model=model_name,
            **model_client_kwargs
        )
        
        # Initialize the memory store
        self.memory_store = CloudMemoryStore(
            collection_name=collection_name,
            api_key=qdrant_api_key,
            host_url=qdrant_host_url
        )
        
        # Default system message if none provided
        if system_message is None:
            system_message = """You are a helpful AI assistant with memory capabilities. 
            You can remember important information from your conversations and use it to provide better responses.
            
            When you receive a message, you'll be provided with relevant memories from previous conversations.
            Use these memories to provide more contextual and personalized responses.
            
            If the user shares personal information or preferences, try to remember them for future reference.
            You can also store important information using the store_memory function."""
        
        # Store configuration
        self.auto_recall = auto_recall
        self.auto_recall_threshold = auto_recall_threshold
        self.conversation_history = []
        self.user_name = None  # Will store the user's name
        self.assistant_name = name  # Store the assistant's name
        
        # Create the assistant agent with memory tools
        self.agent = AssistantAgent(
            name=name,
            model_client=self.model_client,
            system_message=system_message,
            tools=[self.store_memory, self.recall_memory],
            reflect_on_tool_use=True,
            model_client_stream=True,  # Enable streaming tokens
        )
    
    async def store_memory(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Store a memory in the vector database.
        
        Args:
            text: The text content to store
            metadata: Additional metadata to store with the text
            
        Returns:
            A confirmation message
        """
        if metadata is None:
            metadata = {}
        
        try:
            # Add timestamp to metadata
            import time
            metadata["timestamp"] = int(time.time())
            
            # Store the memory
            memory_id = self.memory_store.store(text, metadata)
            
            # Verify the memory was stored by trying to retrieve it
            results = self.memory_store.search(text, top_k=1)
            if not results or results[0]["text"] != text:
                raise ValueError("Failed to verify memory storage")
                
            return "Memory stored successfully."
            
        except Exception as e:
            print(f"Error in store_memory: {str(e)}")
            return f"Failed to store memory. Please try again."
    
    async def recall_memory(self, query: str, top_k: int = 3) -> str:
        """
        Recall memories related to the query.
        
        Args:
            query: The search query
            top_k: Number of results to return (1-10)
            
        Returns:
            A formatted string of the search results
        """
        try:
            # Validate top_k
            top_k = max(1, min(10, int(top_k)))  # Ensure top_k is between 1 and 10
            
            # Search for relevant memories
            results = self.memory_store.search(query, top_k)
            
            if not results:
                return "No relevant memories found."
            
            # Format the results
            formatted_results = []
            for i, result in enumerate(results):
                # Skip invalid results
                if not result or "text" not in result:
                    continue
                    
                # Format metadata
                metadata = result.get("metadata", {})
                metadata_items = []
                
                # Add timestamp if available
                if "timestamp" in metadata:
                    from datetime import datetime
                    try:
                        timestamp = int(metadata["timestamp"])
                        dt = datetime.fromtimestamp(timestamp)
                        metadata_items.append(f"remembered on {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                        # Remove timestamp from metadata to avoid duplication
                        metadata = metadata.copy()
                        del metadata["timestamp"]
                    except (ValueError, TypeError):
                        pass
                
                # Add other metadata
                if metadata:
                    metadata_items.extend([f"{k}: {v}" for k, v in metadata.items()])
                
                metadata_str = f" ({', '.join(metadata_items)})" if metadata_items else ""
                
                formatted_results.append(
                    f"{i+1}. (Relevance: {result.get('score', 0.0):.1%}) {result['text']}{metadata_str}"
                )
            
            if not formatted_results:
                return "No valid memories found."
                
            return "Here's what I found:\n\n" + "\n\n".join(formatted_results)
            
        except Exception as e:
            print(f"Error in recall_memory: {str(e)}")
            return "I'm having trouble accessing my memory right now. Please try again later."
    
    async def _get_relevant_memories(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Get relevant memories for a query.
        Always includes the user's and assistant's names if available.
        
        Args:
            query: The search query
            top_k: Maximum number of memories to return
            
        Returns:
            List of relevant memories with scores
        """
        try:
            # First, search for the user's name if we don't have it yet
            if self.user_name is None:
                name_memories = self.memory_store.search("user's name is", top_k=1)
                if name_memories:
                    # Try to extract the name from the memory text
                    memory_text = name_memories[0].get('text', '').lower()
                    if 'my name is' in memory_text:
                        self.user_name = memory_text.split('my name is', 1)[1].strip().split()[0].strip("'\"\',.?!")
                    elif "i'm" in memory_text:
                        self.user_name = memory_text.split("i'm", 1)[1].strip().split()[0].strip("'\"\',.?!")
                    elif 'name is' in memory_text:
                        self.user_name = memory_text.split('name is', 1)[1].strip().split()[0].strip("'\"\',.?!")
            
            # Get relevant memories
            results = self.memory_store.search(query, top_k=top_k) if query else []
            
            # Filter out low-relevance results
            relevant_results = [r for r in results if r.get('score', 0) >= self.auto_recall_threshold]
            
            # Add the assistant's name memory if not already in results
            assistant_name_memory = {
                'text': f"The assistant's name is {self.assistant_name}",
                'metadata': {
                    'type': 'assistant_info',
                    'importance': 'high',
                    'timestamp': datetime.now().isoformat()
                },
                'score': 1.0  # High score to ensure it's always included
            }
            relevant_results.insert(0, assistant_name_memory)
            
            # Add the user's name memory if we have one and it's not already in the results
            if self.user_name and not any('user' in str(r.get('metadata', {})).lower() for r in relevant_results):
                user_name_memory = {
                    'text': f"The user's name is {self.user_name}",
                    'metadata': {
                        'type': 'user_info',
                        'importance': 'high',
                        'timestamp': datetime.now().isoformat()
                    },
                    'score': 1.0  # High score to ensure it's always included
                }
                relevant_results.insert(0, user_name_memory)
            
            return relevant_results
        except Exception as e:
            print(f"Error getting relevant memories: {str(e)}")
            return []
    
    def _format_memories_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        """Format memories for inclusion in the prompt."""
        if not memories:
            return "No relevant memories found."
        
        # Separate memories by type
        assistant_info = []
        user_info = []
        other_high_importance = []
        regular = []
        
        for mem in memories:
            text = mem.get('text', '').strip()
            metadata = mem.get('metadata', {})
            
            # Skip empty memories
            if not text:
                continue
                
            # Format the memory with metadata if available
            meta_str = ""
            if metadata:
                meta_items = [f"{k}: {v}" for k, v in metadata.items() 
                            if k not in ['timestamp', 'source'] and v]
                if meta_items:
                    meta_str = f" ({', '.join(meta_items)})"
            
            formatted = f"- {text}{meta_str}"
            
            # Categorize by type and importance
            if 'assistant' in text.lower() and 'name' in text.lower():
                assistant_info.append(formatted)
            elif 'user' in str(metadata).lower() or 'user' in text.lower():
                user_info.append(formatted)
            elif metadata.get('importance') == 'high':
                other_high_importance.append(formatted)
            else:
                regular.append(formatted)
        
        # Combine all memories in a logical order
        all_memories = []
        
        # 1. Assistant info first
        if assistant_info:
            all_memories.append("About this assistant:")
            all_memories.extend(assistant_info)
        
        # 2. User info
        if user_info:
            if all_memories:
                all_memories.append("")  # Add a blank line between sections
            all_memories.append("About you:")
            all_memories.extend(user_info)
        
        # 3. Other high importance memories
        if other_high_importance:
            if all_memories:
                all_memories.append("")  # Add a blank line between sections
            all_memories.append("Important information:")
            all_memories.extend(other_high_importance)
        
        # 4. Regular memories
        if regular:
            if all_memories:
                all_memories.append("")  # Add a blank line between sections
            all_memories.append("Other relevant information:")
            all_memories.extend(regular)
        
        return "\n".join(all_memories)
    
    async def process_message(self, message: str) -> Any:
        """
        Process a user message and return the agent's response stream.
        Automatically includes relevant memories in the context.
        
        Args:
            message (str): User message
            
        Returns:
            Any: Agent's response stream
        """
        # Add message to conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Get relevant memories if auto_recall is enabled
        context = ""
        if self.auto_recall:
            # Search for relevant memories
            memories = await self._get_relevant_memories(message)
            
            # Format memories for the prompt
            if memories:
                context = self._format_memories_for_prompt(memories)
                # Add context to the message
                message_with_context = f"{message}\n\n{context}"
            else:
                message_with_context = message
        else:
            message_with_context = message
        
        try:
            # Get the response from the agent
            response = await self.agent.run_stream(task=message_with_context)
            
            # Add assistant's response to conversation history
            # Note: For streaming responses, we'd need to collect the full response first
            # For now, we'll just store the message that triggered the response
            self.conversation_history.append({"role": "assistant", "content": "[Response generated]"})
            
            return response
            
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            # Fall back to processing without memory if there's an error
            return self.agent.run_stream(task=message)
    
    async def close(self):
        """Close the model client connection."""
        await self.model_client.close()
