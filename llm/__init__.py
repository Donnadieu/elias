"""
LLM (Large Language Model) package for the AI Assistant.

This package contains LLM client implementations and configurations.
"""

from .deepseek_client import get_llm_config, get_deepseek_config

__all__ = ['get_llm_config', 'get_deepseek_config']
