"""
Execution Agent for the personal assistant using AutoGen's AgentChat framework.
"""
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from typing import Any, Dict, Optional

# Import the executor tool
from tools.executor_tool import run_python_code
# Import LLM configuration
from llm.deepseek_client import get_llm_config

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
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        system_message: str = "You are a helpful AI assistant that can generate and execute Python code based on user instructions.",
        use_deepseek: bool = True
    ):
        """
        Initialize the ExecutionAgent.
        
        Args:
            name (str): Name of the agent
            model_name (Optional[str]): Name of the model to use (overrides config)
            api_key (Optional[str]): API key (if None, will use environment variable)
            system_message (str): System message for the agent
            use_deepseek (bool): Whether to use DeepSeek as the default provider
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
