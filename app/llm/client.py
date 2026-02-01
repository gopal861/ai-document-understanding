# app/llm/client.py
import os
from openai import OpenAI

class LLMClient:
    """
    Client for OpenAI LLM API.
    
    Handles prompt formatting and API calls with error handling.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize OpenAI client.
        
        Args:
            model: OpenAI model to use (default: gpt-4o-mini)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Please set it before running the application."
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        """
        Generate text using OpenAI API.
        
        Args:
            prompt: Input prompt for the model
        
        Returns:
            Generated text response
        
        Raises:
            Exception: If API call fails
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful document understanding assistant. Answer only based on the provided context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=500
            )
            
            # Extract text from response
            return response.choices[0].message.content
        
        except Exception as e:
            # Log error and re-raise
            raise Exception(f"OpenAI API call failed: {str(e)}")


