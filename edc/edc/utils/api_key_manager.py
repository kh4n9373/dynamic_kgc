import openai
import time
import logging
import random
from collections import deque
import os

class APIKeyManager:
    def __init__(self, api_keys):
        self.api_keys = deque(api_keys)
        self.rate_limited_keys = {}
        self.cooldown_period = 60
    
    def get_available_key(self):
        current_time = time.time()
        cooled_down_keys = [k for k, t in list(self.rate_limited_keys.items()) if current_time > t]
        
        for key in cooled_down_keys:
            self.api_keys.append(key)
            del self.rate_limited_keys[key]
            
        if len(self.api_keys) == 0:
            if self.rate_limited_keys:
                min_wait = min(self.rate_limited_keys.values()) - current_time
                if min_wait > 0:
                    logging.warning(f"All API keys are rate limited. Waiting {min_wait:.2f} seconds.")
                    time.sleep(min_wait + 1) 
                    return self.get_available_key()  
            else:
                raise Exception("No API keys available and none are cooling down")
        
        return self.api_keys[0]
    
    def mark_rate_limited(self, key):
        if key in self.api_keys:
            self.api_keys.remove(key)
        self.rate_limited_keys[key] = time.time() + self.cooldown_period
        logging.warning(f"API key rate limited. Cooling down for {self.cooldown_period} seconds.")
    
    def rotate_key(self):
        if len(self.api_keys) > 1:
            self.api_keys.rotate(1)
        return self.api_keys[0]


def openai_chat_completion(model, system_prompt, history, temperature=0, max_tokens=512, key_manager=None):

    if key_manager is None:
        raise ValueError("APIKeyManager instance is required")
    
    if system_prompt is not None:
        messages = [{"role": "system", "content": system_prompt}] + history
    else:
        messages = history
    
    max_retries = 5
    retry_count = 0
    backoff_time = 2  
    
    while retry_count < max_retries:
        try:
            api_key = key_manager.get_available_key()
            
            openai.api_key = api_key
            # Get base URL from environment variable or use default
            llm_base_url = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
            openai.base_url = llm_base_url
            
            response = openai.chat.completions.create(
                model=model, 
                messages=messages, 
                temperature=temperature, 
                max_tokens=max_tokens
            )
            
            logging.debug(f"Model: {model}\nPrompt:\n {messages}\n Result: {response.choices[0].message.content}")
            return response.choices[0].message.content
            
        except openai.RateLimitError:
            key_manager.mark_rate_limited(api_key)
            retry_count += 1
            
        except Exception as e:
            key_manager.rotate_key()
            retry_count += 1
            wait_time = backoff_time * (2 ** (retry_count - 1)) 
            logging.warning(f"Error: {str(e)}. Retrying in {wait_time} seconds with a different key.")
            time.sleep(wait_time)
    
    raise Exception(f"Failed to get a response after {max_retries} retries with all available API keys")


if __name__ == "__main__":
    api_keys = ["key1", "key2", "key3"]
    key_manager = APIKeyManager(api_keys)
    
    result = openai_chat_completion(
        model="gpt-3.5-turbo", 
        system_prompt="You are a helpful assistant.", 
        history=[{"role": "user", "content": "Hello!"}],
        key_manager=key_manager
    )
    print(result)