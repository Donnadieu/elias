"""Python code execution tool for the personal assistant."""
import io
import sys
import signal
import builtins
import importlib
import random
import string
import math
import datetime
import re
import json
import time
import itertools
import functools
import collections
from contextlib import redirect_stdout, contextmanager
from typing import Dict, Any, Optional, List, Union


@contextmanager
def timeout(seconds: int):
    """Context manager for timeout."""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Execution timed out after {seconds} seconds")
    
    # Set the timeout handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Cancel the timeout if we exit normally
        signal.alarm(0)


# Define a list of allowed modules that can be safely imported
ALLOWED_MODULES = [
    "random", "string", "math", "datetime", "re", "json", "time", 
    "itertools", "functools", "collections", "statistics", "decimal",
    "fractions", "heapq", "bisect", "array", "enum", "copy", "pprint",
    "textwrap", "uuid", "calendar", "zoneinfo", "dataclasses"
]

# Create a safe import function that only allows importing from the allowed list
def safe_import(name, *args, **kwargs):
    """A safe wrapper around __import__ that only allows importing from the allowed list."""
    if name in ALLOWED_MODULES:
        return importlib.import_module(name)
    else:
        raise ImportError(f"Import of '{name}' is not allowed for security reasons. Only these modules can be imported: {', '.join(ALLOWED_MODULES)}")

async def run_python_code(code: str, timeout_seconds: int = 10) -> str:
    """
    Safely executes Python code in a restricted context and returns the result.
    
    Args:
        code (str): Python code to execute
        timeout_seconds (int): Maximum execution time in seconds
        
    Returns:
        str: Execution results or error message
    """
    # Create string buffer to capture stdout
    buffer = io.StringIO()
    
    # Create a restricted globals dictionary for execution
    # Pre-import common modules to make them available in the execution environment
    restricted_globals = {
        # Common modules
        "random": random,
        "string": string,
        "math": math,
        "datetime": datetime,
        "re": re,
        "json": json,
        "time": time,
        "itertools": itertools,
        "functools": functools,
        "collections": collections,
        
        # Add the safe import function
        "__import__": safe_import,
        "import_module": safe_import,
        
        # Essential built-ins
        "range": range,
        "len": len,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "print": print,
        "sum": sum,
        "min": min,
        "max": max,
        "enumerate": enumerate,
        "zip": zip,
        "round": round,
        "sorted": sorted,
        "reversed": reversed,
        "abs": abs,
        "all": all,
        "any": any,
        "chr": chr,
        "ord": ord,
        "divmod": divmod,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "hasattr": hasattr,
        "getattr": getattr,
        "setattr": setattr,
        "delattr": delattr,
        "pow": pow,
        "type": type,
    }
    
    # Create a safe __builtins__ dictionary without dangerous functions
    safe_builtins = {}
    for name in dir(builtins):
        if name not in [
            "open",  # Restrict file operations
            "exec", "eval", "compile",  # Restrict code execution
            "input", "breakpoint",  # Restrict interactive features
            "globals", "locals", "vars",  # Restrict access to execution environment
            "memoryview", "object", "super",  # Restrict access to Python internals
        ]:
            try:
                safe_builtins[name] = getattr(builtins, name)
            except AttributeError:
                pass
    
    # Replace the built-in __import__ with our safe version
    safe_builtins["__import__"] = safe_import
    
    restricted_globals["__builtins__"] = safe_builtins
    
    try:
        # Redirect stdout to our buffer and set a timeout
        with redirect_stdout(buffer), timeout(timeout_seconds):
            # Execute the code in the restricted context
            exec(code, restricted_globals)
        
        # Get the captured output
        output = buffer.getvalue().strip()
        if output:
            return f"Execution successful. Output:\n\n{output}"
        else:
            # Try to get the last expression's value
            lines = code.strip().split('\n')
            if lines:
                last_line = lines[-1].strip()
                if last_line and not last_line.startswith(('#', '"""', "'''")) and '=' not in last_line:
                    try:
                        with timeout(2):  # Short timeout for expression evaluation
                            result = eval(last_line, restricted_globals)
                            if result is not None:
                                return f"Execution successful. Result:\n\n{result}"
                    except Exception:
                        pass  # Ignore errors in trying to evaluate the last line
            
            return "Execution successful. No output was produced."
    except TimeoutError as e:
        return f"Execution failed: {str(e)}"
    except Exception as e:
        return f"Execution failed with error:\n\n{type(e).__name__}: {str(e)}"
