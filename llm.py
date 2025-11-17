import os
import openai

from config.config import OPENAI_API_KEY

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

class LLM:
    def __init__(self, model_name='gpt-3.5-turbo'):
        self.model_name = model_name

    def chat(self, system_prompt: str, user_prompt: str, temperature=0.0, max_tokens=512):
        try:
            if not OPENAI_API_KEY:
                # simple fallback - deterministic canned response
                return 'OpenAI API key not provided. Set OPENAI_API_KEY in environment or config/config.py.'
            resp = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return resp['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f'LLM call failed: {e}'
