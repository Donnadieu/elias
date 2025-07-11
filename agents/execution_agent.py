"""
Execution Agent for the personal assistant using AutoGen's AgentChat framework.
"""
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from typing import Any, Dict, Optional

# Import the executor tool
from tools.executor_tool import run_python_code

# Try to load environment variables if dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using environment variables directly.")
    # Continue without dotenv


class ExecutionAgent:
    """
    A wrapper class for the AutoGen AssistantAgent that can execute Python code.
    """
    
    def __init__(
        self,
        name: str = "execution_agent",
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        system_message: str = "You are a helpful AI assistant that can generate and execute Python code based on user instructions."
    ):
        """
        Initialize the ExecutionAgent.
        
        Args:
            name (str): Name of the agent
            model_name (str): Name of the model to use
            api_key (Optional[str]): OpenAI API key (if None, will use environment variable)
            system_message (str): System message for the agent
        """
        # Create model client
        model_client_kwargs = {}
        if api_key:
            model_client_kwargs["api_key"] = api_key
            
        self.model_client = OpenAIChatCompletionClient(
            model=model_name,
            **model_client_kwargs
        )
        
        # Create the assistant agent with the run_python_code tool
        self.agent = AssistantAgent(
            name=name,
            model_client=self.model_client,
            system_message=system_message,
            tools=[run_python_code],  # Pass tools directly in the constructor
            reflect_on_tool_use=True,
            model_client_stream=True,  # Enable streaming tokens
        )
    
    async def process_message(self, message: str) -> Any:
        """
        Process a user message and return the agent's response stream.
        
        Args:
            message (str): User message
            
        Returns:
            Any: Agent's response stream
        """
        # The run_stream method returns an awaitable stream that can be passed to Console
        return self.agent.run_stream(task=message)
    
    async def close(self):
        """Close the model client connection."""
        await self.model_client.close()
