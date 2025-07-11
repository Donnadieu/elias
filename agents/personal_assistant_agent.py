"""
Personal Assistant Agent with RAG (Retrieval-Augmented Generation) capabilities.
Combines code execution, memory, and knowledge retrieval.
"""
import os
import asyncio
from typing import Any, Dict, Optional, List, Union
from datetime import datetime, timedelta

# Import AutoGen components
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Import the executor tool
from tools.executor_tool import run_python_code

# Import the RAG system
from .rag_system import RAGSystem, Document

# Import LLM configuration
from llm.deepseek_client import get_llm_config

# Try to load environment variables if dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using environment variables directly.")


class PersonalAssistantAgent:
    """
    A wrapper class for the AutoGen AssistantAgent that combines code execution
    and memory capabilities.
    """
    
    def __init__(
        self,
        name: str = "personal_assistant",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        system_message: Optional[str] = None,
        memory_collection: str = "personal_assistant_memories",
        qdrant_api_key: Optional[str] = None,
        qdrant_host_url: Optional[str] = None,
        rag_collection: str = "assistant_knowledge",
        use_deepseek: bool = True
    ):
        """
        Initialize the PersonalAssistantAgent with RAG capabilities.
        
        Args:
            name (str): Name of the agent
            model_name (Optional[str]): Name of the model to use for generation (overrides config)
            api_key (Optional[str]): API key (if None, will use environment variable)
            system_message (Optional[str]): System message for the agent
            memory_collection (str): Name of the memory collection to use
            qdrant_api_key (Optional[str]): Qdrant API key
            qdrant_host_url (Optional[str]): Qdrant host URL
            rag_collection (str): Name of the RAG collection to use
            use_deepseek (bool): Whether to use DeepSeek as the default provider
        """
        # Set the assistant's name and model
        self.name = name
        self.model_name = model_name
        
        # Default system message if none provided
        if system_message is None:
            system_message = f"""You are {name}, a helpful AI assistant with these capabilities:
            - Generate and execute Python code
            - Remember and recall information from conversations
            - Answer questions using your knowledge base
            
            Important information:
            - Your name is {name}
            - Always be helpful, accurate, and concise
            - Use provided context when available
            - Admit when you don't know something
            """
        
        # Get LLM configuration (DeepSeek by default, falls back to OpenAI)
        llm_config = get_llm_config(use_deepseek=use_deepseek)
        
        # Create model client with custom configuration if provided
        model_client_kwargs = {}
        
        # If we have a config list, extract all necessary parameters
        if "config_list" in llm_config and llm_config["config_list"]:
            config = llm_config["config_list"][0]
            
            # Extract model name
            if model_name:
                model_client_kwargs["model"] = model_name
            else:
                model_client_kwargs["model"] = config.get("model")
                
            # Extract API key
            if api_key:
                model_client_kwargs["api_key"] = api_key
            else:
                model_client_kwargs["api_key"] = config.get("api_key")
                
            # Extract base URL for DeepSeek
            if "base_url" in config:
                model_client_kwargs["base_url"] = config.get("base_url")
                
            # Extract model_info for non-OpenAI models
            if "model_info" in config:
                model_client_kwargs["model_info"] = config.get("model_info")
        else:
            # Fallback if no config is available
            if model_name:
                model_client_kwargs["model"] = model_name
            if api_key:
                model_client_kwargs["api_key"] = api_key
                
        self.model_client = OpenAIChatCompletionClient(
            **model_client_kwargs
        )
        
        # Initialize the RAG system
        self.rag = RAGSystem(
            collection_name=rag_collection,
            model_name="all-MiniLM-L6-v2",  # Embedding model
            qdrant_url=qdrant_host_url or os.environ.get("QDRANT_HOST_URL"),
            qdrant_api_key=qdrant_api_key or os.environ.get("QDRANT_API_KEY")
        )
        
        # Store assistant's name in memory (will be stored asynchronously when needed)
        
        # Define tools for the agent
        async def store_knowledge(text: str, metadata: Dict[str, Any] = None) -> str:
            """Store information in the knowledge base."""
            if metadata is None:
                metadata = {}
            doc = Document(
                text=text,
                metadata={
                    "type": "knowledge",
                    "source": "user",
                    "timestamp": datetime.utcnow().isoformat(),
                    **metadata
                }
            )
            doc_id = await self.rag.add_documents([doc])
            return f"Knowledge stored with ID: {doc_id[0] if doc_id else 'error'}"
        
        async def search_knowledge(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
            """Search the knowledge base."""
            results = await self.rag.search(query, top_k=top_k)
            return results or [{"text": "No relevant information found."}]
        
        # Create the assistant agent with tools
        self.agent = AssistantAgent(
            name=name,
            model_client=self.model_client,
            system_message=system_message,
            tools=[
                run_python_code,   # Code execution
                store_knowledge,   # Knowledge storage
                search_knowledge   # Knowledge retrieval
            ],
            reflect_on_tool_use=True,
            model_client_stream=True
        )
    
    async def get_assistant_name(self) -> str:
        """
        Get the assistant's name from the knowledge base.
        
        Returns:
            str: The assistant's name, or the default name if not found
        """
        try:
            print("Debug: Searching for assistant's name in knowledge base...")
            # Search for the assistant's name in the knowledge base
            results = await self.rag.search("assistant's name is", filter_metadata={"type": "assistant_info"}, top_k=1)
            
            print(f"Debug: Search results: {results}")
            
            if results and isinstance(results, list) and len(results) > 0 and 'text' in results[0]:
                # Extract the name from the stored text
                text = results[0]['text']
                print(f"Debug: Found text in knowledge base: {text}")
                
                if "assistant's name is" in text.lower():
                    name = text.split("assistant's name is")[1].strip()
                    # Remove any trailing punctuation
                    name = name.rstrip('.').strip()
                    if name:
                        self.name = name
                        print(f"Debug: Extracted name: {name}")
                        return name
            else:
                print("Debug: No valid results found in knowledge base")
                
        except Exception as e:
            print(f"Warning: Could not retrieve assistant name from knowledge base: {str(e)}")
        
        print(f"Debug: Returning default name: {self.name}")
        return self.name
    
    async def _store_assistant_info(self):
        """
        Store assistant's information in the RAG knowledge base.
        
        This method ensures there's only one assistant info document in the knowledge base.
        If an existing document is found, it will be updated with the current information.
        """
        try:
            print(f"Debug: Storing assistant info for name: {self.name}")
            
            # First, check if there's an existing assistant info document
            print("Debug: Searching for existing assistant info...")
            results = await self.rag.search(
                "assistant's name is", 
                filter_metadata={"type": "assistant_info"}, 
                top_k=1
            )
            
            print(f"Debug: Found {len(results)} existing assistant info documents")
            
            # Delete any existing assistant info documents
            if results and len(results) > 0 and 'id' in results[0]:
                print(f"Debug: Deleting existing assistant info document: {results[0]['id']}")
                await self.rag.delete_document(results[0]['id'])
            
            # Create a new document with the current information
            doc_text = f"The assistant's name is {self.name}"
            doc_metadata = {
                "type": "assistant_info",
                "importance": "high",
                "source": "system",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            print(f"Debug: Creating new assistant info document: {doc_text}")
            print(f"Debug: With metadata: {doc_metadata}")
            
            doc = Document(
                text=doc_text,
                metadata=doc_metadata
            )
            
            # Add the new document
            print("Debug: Adding new document to knowledge base...")
            doc_ids = await self.rag.add_documents([doc])
            print(f"Debug: Added document with IDs: {doc_ids}")
            
            # Verify the document was added
            if doc_ids and len(doc_ids) > 0:
                print("Debug: Successfully stored assistant info in knowledge base")
            else:
                print("Warning: Failed to store assistant info in knowledge base")
            
        except Exception as e:
            print(f"Error storing assistant info in knowledge base: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    async def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a document to the RAG knowledge base.
        
        Args:
            text: The text content to add
            metadata: Optional metadata for the document
            
        Returns:
            str: Confirmation message with document ID
        """
        try:
            if metadata is None:
                metadata = {}
                
            doc = Document(
                text=text,
                metadata={
                    "type": "document",
                    "source": "user_upload",
                    "timestamp": datetime.utcnow().isoformat(),
                    **metadata
                }
            )
            
            doc_ids = await self.rag.add_documents([doc])
            if doc_ids:
                return f"Document added to knowledge base with ID: {doc_ids[0]}"
            return "Failed to add document to knowledge base."
            
        except Exception as e:
            print(f"Error adding document: {e}")
            return f"Error adding document: {str(e)}"
    
    async def process_message(self, message: str) -> str:
        """
        Process a user message using RAG and return the assistant's response.
        
        Args:
            message: The user's message
            
        Returns:
            str: The assistant's response
        """
        try:
            # Clean and normalize the input message
            message = message.strip()
            if not message:
                return "I didn't catch that. Could you please rephrase?"

            # Handle name changes
            if self._is_name_change_request(message):
                return await self._handle_name_change(message)
            
            # Handle document ingestion (e.g., "remember that ..." or "remember: ...")
            if message.lower().startswith("remember that ") or message.lower().startswith("remember: "):
                # Extract the content after "remember that " or "remember: "
                prefix = "remember that " if "remember that " in message.lower() else "remember: "
                content = message[message.lower().find(prefix) + len(prefix):].strip()
                if content:
                    return await self.add_document(content)
                return "Please provide content to remember after 'remember that' or 'remember:'"
            
            # First, store the assistant's name if not already done
            if not hasattr(self, '_name_stored') or not self._name_stored:
                await self._store_assistant_info()
                self._name_stored = True
            
            # Enhanced system prompt with more context
            system_prompt = f"""You are {self.name}, a helpful and knowledgeable AI assistant. 
            Your goal is to provide accurate, helpful, and concise responses to the user's queries.
            
            Guidelines:
            1. Use the provided context to inform your response when relevant
            2. If the context doesn't contain the answer, use your general knowledge
            3. Be friendly, professional, and concise
            4. If you're unsure about something, say so
            5. For coding questions, provide clear examples when possible
            
            Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            # Check if we should use RAG or just the base model
            if hasattr(self, 'rag') and self.rag is not None:
                try:
                    # First, try to get relevant context from the knowledge base
                    context_results = await self.rag.search(
                        query=message,
                        top_k=3,
                        score_threshold=0.4,  # Slightly lower threshold to get more context
                        include_metadata=True
                    )
                    
                    # Format the context for the prompt
                    context = ""
                    if context_results:
                        context = "Relevant information from my knowledge base:\n"
                        for i, result in enumerate(context_results, 1):
                            text = result.get('text', '').strip()
                            if text:
                                source = result.get('metadata', {}).get('source', 'knowledge base')
                                context += f"{i}. {text} (Source: {source})\n"
                    
                    # If we have context, use it to generate a response
                    if context:
                        # Prepare the messages for the model
                        from autogen_core.models._types import SystemMessage, UserMessage
                        
                        # Create system message with instructions
                        system_msg = SystemMessage(content=system_prompt)
                        
                        # Create user message with context and question
                        user_content = f"""Context:
                        {context}
                        
                        Question: {message}
                        
                        Please provide a helpful response based on the context above and your general knowledge."""
                        user_msg = UserMessage(content=user_content, source="user")
                        
                        # Generate response using the LLM with context
                        try:
                            response = await self.model_client.create(
                                messages=[system_msg, user_msg],
                                extra_create_args={
                                    'temperature': 0.7,
                                    'max_tokens': 500
                                }
                            )
                            
                            # Extract the response text
                            if hasattr(response, 'content') and response.content:
                                return response.content.strip()
                            
                        except Exception as e:
                            print(f"Error generating response with context: {str(e)}")
                            import traceback
                            traceback.print_exc()
                    
                    # If no relevant context was found or error occurred, try without context
                    try:
                        from autogen_core.models._types import SystemMessage, UserMessage
                        
                        # Create system message with instructions
                        system_msg = SystemMessage(content=system_prompt)
                        
                        # Create user message with the original question
                        user_msg = UserMessage(content=message, source="user")
                        
                        # Generate response using just the base model
                        response = await self.model_client.create(
                            messages=[system_msg, user_msg],
                            extra_create_args={
                                'temperature': 0.7,
                                'max_tokens': 500
                            }
                        )
                        
                        if hasattr(response, 'content') and response.content:
                            return response.content.strip()
                        
                        return "I'm not sure how to respond to that. Could you provide more details?"
                        
                    except Exception as e:
                        print(f"Error generating response: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        return "I'm having some technical difficulties right now. Could you try asking again in a different way?"
                    
                except Exception as e:
                    print(f"Error in RAG processing: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # Fall back to basic response if RAG fails
                    return "I'm having some technical difficulties right now. Could you try asking again in a different way?"
            else:
                # Fallback if RAG isn't available
                return f"I'm {self.name}. I'm currently running in a limited mode. You said: {message}"
            
        except Exception as e:
            print(f"Error in process_message: {str(e)}")
            return "I'm sorry, I encountered an error processing your message. Please try again."
    
    def _is_name_change_request(self, message: str) -> bool:
        """Check if the message is requesting a name change."""
        return any(cmd in message.lower() for cmd in ["your name is", "call you"])
    
    async def _handle_name_change(self, message: str) -> str:
        """
        Handle requests to change the assistant's name.
        
        Args:
            message: The user's message containing the name change request
            
        Returns:
            str: Response message confirming the name change or requesting clarification
        """
        try:
            # Extract the new name from the message
            if "your name is" in message.lower():
                new_name = message.lower().split("your name is")[1].strip()
            elif "call you" in message.lower():
                new_name = message.lower().split("call you")[1].strip()
            else:
                return "I'm not sure what name you'd like me to use. Please say 'your name is [name]'."
            
            # Clean up the name
            if new_name:
                new_name = new_name.strip(" .!?\"'")
                if new_name:
                    # Update the name in the knowledge base and system
                    await self._update_assistant_name(new_name)
                    return f"Got it! I'll go by {new_name} from now on."
            
            return "I didn't catch the name. Could you please repeat that?"
            
        except Exception as e:
            print(f"Error updating assistant name: {str(e)}")
            return f"I'm sorry, I encountered an error updating my name: {str(e)}. Please try again later."
    
    async def _handle_memory_operations(self, message: str) -> Optional[str]:
        """Handle explicit memory operations (remember/recall)."""
        lower_msg = message.lower()
        
        # Handle 'remember' command
        if lower_msg.startswith("remember:"):
            memory_text = message[len("remember:"):].strip()
            if not memory_text:
                return "Please provide something to remember after 'remember:'"
            
            # Store the memory with metadata
            memory = {
                "text": memory_text,
                "metadata": {
                    "type": "user_memory",
                    "timestamp": datetime.now().isoformat(),
                    "importance": "medium",
                    "source": "explicit"
                }
            }
            try:
                await self.memory_store.store(memory)
                return f"I'll remember that: {memory_text}"
            except Exception as e:
                print(f"Error storing memory: {e}")
                return "I had trouble saving that to memory."
        
        # Handle 'recall' command
        elif lower_msg.startswith("recall:"):
            query = message[len("recall:"):].strip()
            if not query:
                return "Please provide a search query after 'recall:'"
            
            try:
                results = await self.memory_store.search(query, top_k=3)
                if results:
                    response = ["I found these related memories:"]
                    for i, result in enumerate(results, 1):
                        response.append(f"{i}. {result.get('text', 'No text available')}")
                    return "\n".join(response)
                return "I couldn't find any related memories."
            except Exception as e:
                print(f"Error recalling memories: {e}")
                return "I had trouble searching my memories."
        
        return None
    
    async def _get_enhanced_context(self, message: str) -> str:
        """Retrieve and format relevant context from memory."""
        try:
            # Get relevant memories with higher threshold for relevance
            memories = await self.memory_store.search(message, top_k=3)
            
            if not memories:
                return ""
                
            # Format context with relevance scores
            context_lines = ["Relevant context from our conversation:"]
            for mem in memories:
                if mem.get('score', 0) >= 0.4:  # Higher threshold for better relevance
                    text = mem.get('text', '').strip()
                    if text:
                        # Add source/timestamp if available
                        meta = mem.get('metadata', {})
                        source = meta.get('source', 'memory')
                        timestamp = meta.get('timestamp', '')
                        if timestamp:
                            context_lines.append(f"- {text} (source: {source}, {timestamp})")
                        else:
                            context_lines.append(f"- {text}")
            
            return "\n".join(context_lines) if len(context_lines) > 1 else ""
            
        except Exception as e:
            print(f"Error getting context: {e}")
            return ""
    
    async def _generate_response(self, message: str, context: str = "") -> str:
        """Generate a response using the agent with the given context."""
        try:
            # Prepare the message with context
            if context:
                message_with_context = f"{message}\n\n{context}"
            else:
                message_with_context = message
            
            # Get response from the agent
            response = await self.agent.run_stream(task=message_with_context)
            
            # Handle both generator and direct responses
            if hasattr(response, '__aiter__'):
                # For streaming responses, collect all chunks
                response_text = []
                async for chunk in response:
                    if hasattr(chunk, 'text'):
                        response_text.append(chunk.text)
                    elif isinstance(chunk, str):
                        response_text.append(chunk)
                return ''.join(response_text)
            
            return str(response) if response else "I'm not sure how to respond to that."
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm having trouble generating a response right now. Could you try again?"
    
    def _get_context_for_message(self, message: str) -> str:
        """Get relevant context from memory for the given message."""
        try:
            # Get relevant memories
            memories = self.memory_store.search(message, top_k=3)
            
            # Filter out low-relevance results and format them
            context = []
            for mem in memories:
                if mem.get('score', 0) >= 0.3:  # Only include reasonably relevant memories
                    text = mem.get('text', '').strip()
                    if text:
                        context.append(f"- {text}")
            
            return "\n".join(context) if context else ""
            
        except Exception as e:
            print(f"Error getting context: {str(e)}")
            return ""
    
    async def _update_assistant_name(self, new_name: str):
        """
        Update the assistant's name and store it in the knowledge base.
        
        Args:
            new_name (str): The new name for the assistant
        """
        # Update the instance variable
        old_name = self.name
        self.name = new_name
        
        # Update the system message with the new name
        new_system_message = f"""You are {new_name}, a helpful AI assistant with these capabilities:
            - Generate and execute Python code
            - Remember and recall information from conversations
            - Answer questions using your knowledge base
            
            Important information:
            - Your name is {new_name}
            - Always be helpful, accurate, and concise
            - Use provided context when available
            - Admit when you don't know something
            """
        
        # Update the agent's system message if possible
        if hasattr(self, 'agent') and hasattr(self.agent, 'update_system_message'):
            self.agent.update_system_message(new_system_message)
            
        # Store the updated name in the knowledge base
        try:
            # First, try to find and update the existing assistant info
            results = await self.rag.search("assistant's name is", 
                                         filter_metadata={"type": "assistant_info"}, 
                                         top_k=1)
            
            if results and 'id' in results[0]:
                # Update the existing document
                doc_id = results[0]['id']
                await self.rag.delete_document(doc_id)
            
            # Add the updated document
            await self._store_assistant_info()
            
        except Exception as e:
            print(f"Warning: Could not update assistant name in knowledge base: {str(e)}")
            # Revert the name if we couldn't update the knowledge base
            self.name = old_name
            if hasattr(self, 'agent') and hasattr(self.agent, 'update_system_message'):
                self.agent.update_system_message(
                    f"""You are {old_name}, a helpful AI assistant with these capabilities:
                    - Generate and execute Python code
                    - Remember and recall information from conversations
                    - Answer questions using your knowledge base
                    
                    Important information:
                    - Your name is {old_name}
                    - Always be helpful, accurate, and concise
                    - Use provided context when available
                    - Admit when you don't know something
                    """
                )
            raise
        
        # Store the name in memory
        try:
            memory = {
                "text": f"The assistant's name is {new_name}",
                "metadata": {
                    "type": "assistant_info",
                    "timestamp": datetime.now().isoformat(),
                    "importance": "high"
                }
            }
            self.memory_store.store(memory)
        except Exception as e:
            print(f"Warning: Could not store assistant name in memory: {e}")
    
    async def close(self):
        """Clean up resources."""
        if hasattr(self, 'memory_store') and hasattr(self.memory_store, 'close'):
            await self.memory_store.close()
        await self.model_client.close()
