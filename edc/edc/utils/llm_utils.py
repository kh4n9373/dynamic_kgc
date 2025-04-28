import os
import openai
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import ast
from sentence_transformers import SentenceTransformer
from typing import List
import gc
import torch
import logging
import ctranslate2
from transformers import AutoTokenizer
import threading

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import argparse 
import time
from dotenv import load_dotenv
import os 
load_dotenv()

import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

thread_local = threading.local()

class ApiKeyManager:
    def __init__(self):
        self.keys = self._load_api_keys()
        self.key_lock = threading.Lock()
        self.key_status = {key: {"errors": 0, "last_used": 0, "cooling_down": False} for key in self.keys}
        
    def _load_api_keys(self):
        """Load API keys from environment variables"""
        keys = []
        # Look for GEMINI_API_KEY and GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.
        main_key = os.getenv("GEMINI_API_KEY")
        if main_key:
            keys.append(main_key)
        
        # Look for additional keys
        i = 1
        while True:
            key = os.getenv(f"GEMINI_API_KEY_{i}")
            if key:
                keys.append(key)
                i += 1
            else:
                break
        
        print(f"Loaded {len(keys)} API keys")
        return keys
    
    def get_key(self):
        """Get the next available API key"""
        with self.key_lock:
            current_time = time.time()
            available_keys = [
                key for key, status in self.key_status.items() 
                if not status["cooling_down"] or (current_time - status["last_used"] > 60)  # 60 second cooldown
            ]
            
            if not available_keys:
                # All keys are cooling down, wait and try the least recently used one
                key = min(self.key_status.items(), key=lambda x: x[1]["last_used"])[0]
                time.sleep(2)  # Wait a bit before using a cooling down key
            else:
                # Use the key with the fewest recent errors
                key = min(available_keys, key=lambda k: self.key_status[k]["errors"])
            
            self.key_status[key]["last_used"] = current_time
            return key
    
    def mark_error(self, key, error_type):
        """Mark an API key as having an error"""
        with self.key_lock:
            if error_type == 429:  # Rate limit error
                self.key_status[key]["errors"] += 5
                self.key_status[key]["cooling_down"] = True
                print(f"API key {key[:5]}... is rate limited, cooling down")
            else:
                self.key_status[key]["errors"] += 1
    
    def mark_success(self, key):
        """Mark an API key as having a successful request"""
        with self.key_lock:
            self.key_status[key]["errors"] = max(0, self.key_status[key]["errors"] - 1)
            self.key_status[key]["cooling_down"] = False

api_key_manager = ApiKeyManager()

def free_model(model: AutoModelForCausalLM = None, tokenizer: AutoTokenizer = None):
    try:
        model.cpu()
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(e)


def get_embedding_e5mistral(model, tokenizer, sentence, task=None):
    model.eval()
    device = model.device

    if task != None:
        # It's a query to be embed
        sentence = get_detailed_instruct(task, sentence)

    sentence = [sentence]

    max_length = 4096
    batch_dict = tokenizer(
        sentence, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True
    )
    batch_dict["input_ids"] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict["input_ids"]]
    batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors="pt")

    batch_dict.to(device)

    embeddings = model(**batch_dict).detach().cpu()

    assert len(embeddings) == 1

    return embeddings[0]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


# def get_embedding_sts(model: SentenceTransformer, text: str, prompt_name=None, prompt=None):
#     embedding = model.encode(text, prompt_name=prompt_name, prompt=prompt)
#     return embedding

def get_embedding_sts(text: str, device="cpu"): 
    model_name = "BAAI/bge-m3"
    model_save_path = "bge_m3"
    # model_path = "bge_model_ctranslate2_base"

    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if device == "cuda":
        translator = ctranslate2.Encoder(
            model_save_path, device=device, compute_type="float16"
        )  # or "cuda" for GPU
    else:
        translator = ctranslate2.Encoder(model_save_path, device=device)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].tolist()[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    output = translator.forward_batch([tokens])

    
    last_hidden_state = output.last_hidden_state
    last_hidden_state = np.array(last_hidden_state)
    last_hidden_state = torch.as_tensor(last_hidden_state, device = device)[0]

    last_hidden_state = torch.nn.functional.normalize(last_hidden_state, p=2, dim=1)

    if device == "cuda":
        embeddings = last_hidden_state.detach().cpu().tolist()[0]
    else:
        embeddings = last_hidden_state.detach().tolist()[0]

    return embeddings

def parse_raw_entities(raw_entities: str):
    parsed_entities = []
    left_bracket_idx = raw_entities.index("[")
    right_bracket_idx = raw_entities.index("]")
    try:
        parsed_entities = ast.literal_eval(raw_entities[left_bracket_idx : right_bracket_idx + 1])
    except Exception as e:
        pass
    logging.debug(f"Entities {raw_entities} parsed as {parsed_entities}")
    return parsed_entities


def parse_raw_triplets(raw_triplets: str):
    # Look for enclosing brackets
    unmatched_left_bracket_indices = []
    matched_bracket_pairs = []

    collected_triples = []
    for c_idx, c in enumerate(raw_triplets):
        if c == "[":
            unmatched_left_bracket_indices.append(c_idx)
        if c == "]":
            if len(unmatched_left_bracket_indices) == 0:
                continue
            # Found a right bracket, match to the last found left bracket
            matched_left_bracket_idx = unmatched_left_bracket_indices.pop()
            matched_bracket_pairs.append((matched_left_bracket_idx, c_idx))
    for l, r in matched_bracket_pairs:
        bracketed_str = raw_triplets[l : r + 1]
        try:
            parsed_triple = ast.literal_eval(bracketed_str)
            if len(parsed_triple) == 3 and all([isinstance(t, str) for t in parsed_triple]):
                if all([e != "" and e != "_" for e in parsed_triple]):
                    collected_triples.append(parsed_triple)
            elif not all([type(x) == type(parsed_triple[0]) for x in parsed_triple]):
                for e_idx, e in enumerate(parsed_triple):
                    if isinstance(e, list):
                        parsed_triple[e_idx] = ", ".join(e)
                collected_triples.append(parsed_triple)
        except Exception as e:
            pass
    logger.debug(f"Triplets {raw_triplets} parsed as {collected_triples}")
    return collected_triples


def parse_relation_definition(raw_definitions: str):
    descriptions = raw_definitions.split("\n")
    relation_definition_dict = {}

    for description in descriptions:
        if ":" not in description:
            continue
        index_of_colon = description.index(":")
        relation = description[:index_of_colon].strip()

        relation_description = description[index_of_colon + 1 :].strip()

        if relation == "Answer":
            continue

        relation_definition_dict[relation] = relation_description
    logger.debug(f"Relation Definitions {raw_definitions} parsed as {relation_definition_dict}")
    return relation_definition_dict


def is_model_openai(model_name):
    return "gpt" or "gemini" in model_name


def generate_completion_transformers(
    input: list,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_token=256,
    answer_prepend="",
):
    device = model.device
    tokenizer.pad_token = tokenizer.eos_token

    messages = tokenizer.apply_chat_template(input, add_generation_prompt=True, tokenize=False) + answer_prepend

    model_inputs = tokenizer(messages, return_tensors="pt", padding=True, add_special_tokens=False).to(device)

    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_token,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )

    generation = model.generate(**model_inputs, generation_config=generation_config)
    sequences = generation["sequences"]
    generated_ids = sequences[:, model_inputs["input_ids"].shape[1] :]
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    logging.debug(f"Prompt:\n {messages}\n Result: {generated_texts}")
    return generated_texts


def openai_chat_completion_with_key(model, messages, temperature=0, max_tokens=512, retry_count=0):
    """Make an OpenAI chat completion request with a specific API key"""
    if not hasattr(thread_local, "api_key"):
        thread_local.api_key = api_key_manager.get_key()
    
    openai.api_key = thread_local.api_key
    openai.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    
    try:
        # print("MESSAGES: \n", messages)
        response = openai.chat.completions.create(
            model=model, 
            messages=messages, 
            temperature=temperature, 
            max_tokens=max_tokens
        )
        api_key_manager.mark_success(thread_local.api_key)
        # print("RESPONSE: \n", response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        error_code = getattr(e, "status_code", 0)
        api_key_manager.mark_error(thread_local.api_key, error_code)
        
        if error_code == 429 or retry_count >= 2:
            thread_local.api_key = api_key_manager.get_key()
        
        wait_time = min(2 ** retry_count, 30)
        print(f"Error: {e}, retrying in {wait_time}s (retry {retry_count+1})")
        time.sleep(wait_time)
        
        if retry_count < 5:  # Maximum 5 retries
            return openai_chat_completion_with_key(model, messages, temperature, max_tokens, retry_count + 1)
        else:
            raise Exception(f"Failed after 5 retries: {e}")

def openai_chat_completion(model, system_prompt, history, temperature=0, max_tokens=512):
    if system_prompt is not None:
        messages = [{"role": "system", "content": system_prompt}] + history
    else:
        messages = history
    
    logging.debug(f"Model: {model}\nPrompt:\n {messages}")
    result = openai_chat_completion_with_key(model, messages, temperature, max_tokens)
    logging.debug(f"Result: {result}")
    return result

def batch_openai_chat_completion(model, system_prompt, batch_histories, temperature=0, max_tokens=512, max_workers=5):
    results = [None] * len(batch_histories)
    
    all_messages = []
    for history in batch_histories:
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}] + history
        else:
            messages = history
        all_messages.append(messages)
    
    def process_request(idx, messages):
        try:
            result = openai_chat_completion_with_key(model, messages, temperature, max_tokens)
            results[idx] = result
            return idx, result
        except Exception as e:
            logging.error(f"Error processing request {idx}: {e}")
            results[idx] = f"ERROR: {str(e)}"
            return idx, f"ERROR: {str(e)}"
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_request, i, msgs) for i, msgs in enumerate(all_messages)]
        for future in as_completed(futures):
            try:
                idx, result = future.result()
                logging.debug(f"Completed request {idx}")
            except Exception as e:
                logging.error(f"Exception in thread: {e}")
    
    return results
