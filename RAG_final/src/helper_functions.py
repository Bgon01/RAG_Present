# src/helper_functions.py

import time
import random
import asyncio
from typing import Any, Callable
from functools import wraps

def retry_with_exponential_backoff(
    func: Callable, max_retries: int = 5, initial_delay: float = 1, backoff_factor: float = 2
) -> Any:
    """Retries a function with exponential backoff in case of exceptions."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(delay + random.uniform(0, 0.1))
                delay *= backoff_factor
    return wrapper
