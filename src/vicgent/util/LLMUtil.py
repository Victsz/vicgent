"""LLM utility functions for model creation and management."""

import os
from typing import Optional
from langchain.chat_models import init_chat_model


def create_model_anthropic(
    model_name: str,
    base_url: Optional[str] = None,
    temperature: float = 0,
    api_key_env_var: str = "API_KEY",
    **kwargs
):
    """
    Create a standardized Anthropic LLM model instance.
    
    Args:
        model_name: The name of the model to use (e.g., "Pro/deepseek-ai/DeepSeek-V3")
        base_url: Base URL for the API endpoint. If None, uses environment variable ANTHROPIC_BASE_URL
        temperature: Sampling temperature (0 for deterministic output)
        api_key_env_var: Environment variable name for API key
        **kwargs: Additional arguments to pass to init_chat_model
        
    Returns:
        A configured Anthropic chat model instance
        
    Example:
        >>> tllm = create_model_anthropic(
        ...     model_name="Pro/deepseek-ai/DeepSeek-V3",
        ...     temperature=0
        ... )
    """
    # Get base URL from environment if not provided
    if base_url is None:
        base_url = os.getenv("ANTHROPIC_BASE_URL")

    
    # Get API key from environment
    api_key = os.getenv(api_key_env_var)
    if api_key:
        os.environ['ANTHROPIC_AUTH_TOKEN'] = api_key
    print(f"URL: {base_url}, API key: {api_key[0:10]}")
    # Create and return the model with fixed model_provider
    return init_chat_model(
        model=model_name,
        base_url=base_url.strip(),
        model_provider="anthropic",
        temperature=temperature,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/home/victor/workspace/playgrounds/langchain/.agent.env",override=True)
    from langchain_core.output_parsers import StrOutputParser
    # Create model using environment variables
    tllm = create_model_anthropic(
        model_name=os.getenv("TMODEL", "Pro/deepseek-ai/DeepSeek-V3")
    )
    
    print("Testing model connection...")
    # response = tllm.invoke("are you ready? answer with yes or no")
    parser = StrOutputParser()
    chain = (tllm|parser)
    rsp_content = chain.invoke("are you ready? answer with yes or no")

    print(f"Model response: {rsp_content=}")
    print("Anthropic model created successfully!")