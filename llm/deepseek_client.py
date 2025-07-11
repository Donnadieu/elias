"""
DeepSeek LLM client configuration and utilities.
"""
import os
from typing import Dict, Any, Optional


def get_deepseek_config() -> Dict[str, Any]:
    """
    Get the DeepSeek LLM configuration.
    
    Returns:
        Dict containing the DeepSeek configuration for AutoGen.
        
    Raises:
        ValueError: If required environment variables are not set.
    """
    base_url = os.getenv("DEEPSEEK_BASE_URL")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # Default model if not specified
    
    if not all([base_url, api_key]):
        raise ValueError(
            "DeepSeek configuration is incomplete. "
            "Please set DEEPSEEK_BASE_URL and DEEPSEEK_API_KEY environment variables."
        )
    
    # Ensure the base URL has the correct format
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    
    # Model info is required for non-OpenAI models
    model_info = {
        "name": model,
        "family": "deepseek",  # Required field in v0.4.7+
        "max_tokens": 4096,  # Default max tokens for DeepSeek models
        "has_chat_completions": True,
        "has_completions": True,
        "has_embeddings": False,
        "has_chat_completion_stream": True,
        "has_completion_stream": True,
        "supports_functions": True,
        "supports_tools": True,
        "vision": False,  # Required field in v0.4.7+
        "function_calling": "auto",  # Required field in v0.4.7+
        "json_output": True  # Required field in v0.4.7+
    }
    
    return {
        "config_list": [{
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "api_type": "openai",  # DeepSeek uses OpenAI-compatible API
            "model_info": model_info  # Required for non-OpenAI models
        }],
        "timeout": 120,
        "cache_seed": 42  # For reproducibility
    }


def get_llm_config(use_deepseek: bool = True) -> Dict[str, Any]:
    """
    Get LLM configuration, defaulting to DeepSeek but falling back to OpenAI if needed.
    
    Args:
        use_deepseek: Whether to try using DeepSeek first.
        
    Returns:
        Dict containing the LLM configuration.
        
    Raises:
        RuntimeError: If no valid LLM configuration can be created.
    """
    if use_deepseek:
        try:
            return get_deepseek_config()
        except ValueError as e:
            print(f"Warning: {str(e)} Falling back to OpenAI if configured.")
    
    # Fall back to OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError(
            "No valid LLM configuration found. "
            "Please set either DeepSeek or OpenAI API keys in your environment."
        )
    
    return {
        "config_list": [{
            "model": os.getenv("OPENAI_MODEL", "gpt-4"),
            "api_key": openai_api_key,
            "api_type": "openai"
        }],
        "timeout": 120,
        "cache_seed": 42
    }
