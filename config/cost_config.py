"""
Cost configuration for LLM pricing to support ROI tracking.
Prices based on Groq / Llama 3 rates.
"""

# Model pricing per 1M tokens (USD)
# Note: These values can be updated as provider prices change.
PRICING = {
    "llama-3.3-70b-versatile": {
        "prompt": 0.59,
        "completion": 0.79
    },
    "llama3-70b-8192": {
        "prompt": 0.59,
        "completion": 0.79
    },
    "llama3-8b-8192": {
        "prompt": 0.05,
        "completion": 0.08
    }
}

DEFAULT_PRICING = {
    "prompt": 0.50,
    "completion": 0.50
}

def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate estimated cost in USD"""
    pricing = PRICING.get(model_name, DEFAULT_PRICING)
    
    prompt_cost = (prompt_tokens / 1_000_000) * pricing["prompt"]
    completion_cost = (completion_tokens / 1_000_000) * pricing["completion"]
    
    return prompt_cost + completion_cost
