import pandas as pd
import csv
from transformers import AutoTokenizer
import ctranslate2
import torch
import numpy as np
from typing import Union, List, Dict, Optional
import re
from tqdm import tqdm
import gc
import os
import argparse
from functools import lru_cache


def parse_arguments():
    parser = argparse.ArgumentParser(description='Construct final dataset by merging original and EDC outputs with similarity calculations')
    
    # File path arguments
    parser.add_argument('--original', type=str, default='preprocessed_dataset/webnlg_simple.csv',
                        help='Path to original dataset CSV file')
    parser.add_argument('--edc', type=str, default='preprocessed_edc_result/edc_output.csv',
                        help='Path to EDC output CSV file')
    parser.add_argument('--output', type=str, default='raw_dataset_constructed/merged_output.csv',
                        help='Path for output merged CSV file')
    
    # Hardware optimization arguments
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda', 'mps'], default='auto',
                        help='Device to use for computation (auto, cpu, cuda, mps)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing embeddings')
    parser.add_argument('--cache_size', type=int, default=10000,
                        help='Number of embeddings to cache')
    
    # Model parameters
    parser.add_argument('--similarity_threshold', type=float, default=0.9,
                        help='Similarity threshold for matching relations')
    parser.add_argument('--model_name', type=str, default='BAAI/bge-m3',
                        help='Name of the embedding model to use')
    parser.add_argument('--model_path', type=str, default='embedding_model',
                        help='Path to the saved model')
    
    # Predefined configurations
    parser.add_argument('--config', type=str, choices=['webnlg_simple', 'webnlg', 'wiki-nre', 'rebel'],
                        help='Use predefined dataset configuration')
    
    args = parser.parse_args()
    
    # Handle predefined configurations
    if args.config:
        if args.config == 'webnlg_simple':
            args.original = 'preprocessed_dataset/webnlg_simple.csv'
            args.edc = 'preprocessed_edc_result/edc_webnlg_simple.csv'
            args.output = 'raw_dataset_constructed/webnlg_simple.csv'
        elif args.config == 'webnlg':
            args.original = 'preprocessed_dataset/webnlg.csv'
            args.edc = 'preprocessed_edc_result/edc_webnlg.csv'
            args.output = 'raw_dataset_constructed/webnlg.csv'
        elif args.config == 'wiki-nre':
            args.original = 'preprocessed_dataset/wiki-nre.csv'
            args.edc = 'preprocessed_edc_result/edc_wiki-nre.csv'
            args.output = 'raw_dataset_constructed/wiki-nre.csv'
        elif args.config == 'rebel':
            args.original = 'preprocessed_dataset/rebel.csv'
            args.edc = 'preprocessed_edc_result/edc_rebel.csv'
            args.output = 'raw_dataset_constructed/rebel.csv'
    
    # Auto-detect best available device
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    
    return args


def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Set file paths from arguments
    original_csv = args.original
    edc_csv = args.edc
    output_csv = args.output
    SIMILARITY_THRESHOLD = args.similarity_threshold
    
    # Set up hardware configuration
    BATCH_SIZE = args.batch_size
    CACHE_SIZE = args.cache_size
    DEVICE = args.device
    
    # Display configuration
    print("\nConfiguration:")
    print(f"  Original CSV:        {original_csv}")
    print(f"  EDC CSV:             {edc_csv}")
    print(f"  Output CSV:          {output_csv}")
    print(f"  Similarity Threshold: {SIMILARITY_THRESHOLD}")
    print(f"  Device:              {DEVICE}")
    print(f"  Batch Size:          {BATCH_SIZE}")
    print(f"  Cache Size:          {CACHE_SIZE}")
    print(f"  Model Name:          {args.model_name}")
    print(f"  Model Path:          {args.model_path}\n")
    
    # Initialize model and tokenizer
    print("Initializing embedding model...")
    model_name = args.model_name
    model_save_path = args.model_path
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configure ctranslate2 for hardware acceleration
    ct2_device = "cpu"
    if DEVICE == "cuda" and torch.cuda.is_available():
        ct2_device = "cuda"
    translator = ctranslate2.Encoder(model_save_path, device=ct2_device)
    
    # Embedding computation functions
    def compute_embedding(text: str) -> List[float]:
        """Compute embedding for a single text without caching"""
        if not text:
            return [0.0] * 768
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].tolist()[0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        output = translator.forward_batch([tokens])
        
        last_hidden_state = output.last_hidden_state
        last_hidden_state = np.array(last_hidden_state)
        
        tensor_device = torch.device(DEVICE)
        last_hidden_state = torch.as_tensor(last_hidden_state, device=tensor_device)[0]
        last_hidden_state = torch.nn.functional.normalize(last_hidden_state, p=2, dim=1)

        if DEVICE != "cpu":
            embeddings = last_hidden_state.detach().cpu().tolist()[0]
        else:
            embeddings = last_hidden_state.detach().tolist()[0]

        return embeddings
    
    # Set up caching for efficiency
    lru_cache_compute_embedding = lru_cache(maxsize=CACHE_SIZE)(compute_embedding)

    def batch_compute_embeddings(texts: List[str]) -> Dict[str, List[float]]:
        """Compute embeddings for a batch of texts with caching"""
        if not texts:
            return {}
        
        results = {}
        texts_to_compute = []
        text_indices = []
        
        for i, text in enumerate(texts):
            if not text:
                results[i] = np.zeros(768)
            else:
                try:
                    cached = lru_cache_compute_embedding(text)
                    if cached is not None:
                        results[i] = cached
                    else:
                        texts_to_compute.append(text)
                        text_indices.append(i)
                except Exception:
                    texts_to_compute.append(text)
                    text_indices.append(i)
        
        if texts_to_compute:
            inputs = tokenizer(texts_to_compute, return_tensors="pt", padding=True, truncation=True)
            all_tokens = []
            
            for i in range(len(texts_to_compute)):
                input_ids = inputs["input_ids"][i].tolist()
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                all_tokens.append(tokens)
            
            batch_outputs = []
            for i in range(0, len(all_tokens), BATCH_SIZE):
                batch = all_tokens[i:i+BATCH_SIZE]
                outputs = translator.forward_batch(batch)
                batch_outputs.extend(outputs.last_hidden_state)
            
            for i, (idx, output) in enumerate(zip(text_indices, batch_outputs)):
                tensor_device = torch.device(DEVICE)
                hidden_state = torch.as_tensor(np.array([output]), device=tensor_device)
                normalized = torch.nn.functional.normalize(hidden_state, p=2, dim=1)
                
                if DEVICE != "cpu":
                    embedding = normalized.detach().cpu().numpy()[0]
                else:
                    embedding = normalized.detach().numpy()[0]
                
                embedding_list = embedding.tolist()
                results[idx] = embedding_list
                try:
                    lru_cache_compute_embedding(texts_to_compute[i])
                except Exception:
                    pass
        
        ordered_results = [results[i] for i in range(len(texts))]
        return ordered_results

    def cosine_similarity(vec1: Union[List[float], np.ndarray],
                        vec2: Union[List[float], np.ndarray]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            vec1 = np.asarray(vec1, dtype=np.float32)
            vec2 = np.asarray(vec2, dtype=np.float32)
            
            if vec1.ndim != 1 or vec2.ndim != 1:
                print("Warning: Input must be 1-dimensional vectors")
                return 0.0
                
            if vec1.shape[0] != vec2.shape[0]:
                print(f"Warning: Vectors must have the same dimensions (got {vec1.shape[0]} and {vec2.shape[0]})")
                return 0.0
                
            norm_vec1 = np.linalg.norm(vec1)
            norm_vec2 = np.linalg.norm(vec2)
            
            if norm_vec1 == 0.0 or norm_vec2 == 0.0:
                return 0.0
                
            dot_product = np.dot(vec1, vec2)
            similarity = dot_product / (norm_vec1 * norm_vec2)
            
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0

    def normalize_sentence(text: str) -> str:
        """Normalize sentence by converting to lowercase and removing non-alphabetic characters"""
        if not isinstance(text, str): 
            return ""
        lower_text = text.lower()
        alpha_only = re.sub(r'[^a-z]', '', lower_text)
        return alpha_only

    # Function to clear memory
    def clear_memory():
        """Clean up memory to prevent OOM issues"""
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()
        elif DEVICE == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Starting the processing with {DEVICE.upper()} optimization...")

    try:
        print(f"Reading file: {original_csv}")
        df_original = pd.read_csv(original_csv, dtype=str).fillna('')
        print(f"Read {len(df_original)} rows from {original_csv}")

        print(f"Reading file: {edc_csv}")
        df_edc = pd.read_csv(edc_csv, dtype=str).fillna('')
        print(f"Read {len(df_edc)} rows from {edc_csv}")

        print("Normalizing 'sentence' column...")
        df_original['normalized_sentence'] = df_original['sentence'].apply(normalize_sentence)
        df_edc['normalized_sentence'] = df_edc['sentence'].apply(normalize_sentence)
        print("Sentence normalization complete.")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        exit()
    except Exception as e:
        print(f"Error reading or normalizing CSV file: {e}")
        exit()

    # Merge based on normalized sentences
    print("Merging data based on normalized sentences...")
    try:
        df_merged = pd.merge(
            df_original,
            df_edc.drop(columns=['sentence']),
            how='inner',
            on='normalized_sentence',
            suffixes=('_orig', '_edc')
        )
        print(f"Merge complete. Total pairs created: {len(df_merged)}.")
        if len(df_merged) > 10 * max(len(df_original), len(df_edc)) and len(df_merged) > 1000:
            print(f"Warning: Unusually large number of rows after merge ({len(df_merged)}).")
        if df_merged.empty:
            print("No matching sentences found between the two files. Exiting.")
            exit()
    except KeyError as e:
        print(f"Error: Column 'normalized_sentence' not found for merge - {e}.")
        exit()
    except Exception as e:
        print(f"Error during merge process: {e}")
        exit()

    # --- OPTIMIZED SIMILARITY CALCULATION WITH BATCH PROCESSING ---
    print(f"Calculating similarities with {DEVICE.upper()} optimization...")

    # Collect all definition texts for batch processing
    all_def_orig = df_merged['relation_definition'].fillna('').tolist()
    all_def_edc = df_merged['extracted_relation_definition'].fillna('').tolist()

    # Process in batches to avoid memory issues
    total_rows = len(df_merged)
    similarities = []

    # Process embeddings in batches
    print("Computing embeddings in batches...")
    for batch_start in tqdm(range(0, total_rows, BATCH_SIZE), desc="Processing batches"):
        batch_end = min(batch_start + BATCH_SIZE, total_rows)
        
        # Get texts for this batch
        batch_def_orig = all_def_orig[batch_start:batch_end]
        batch_def_edc = all_def_edc[batch_start:batch_end]
        
        # Compute embeddings for this batch
        emb_orig_batch = batch_compute_embeddings(batch_def_orig)
        emb_edc_batch = batch_compute_embeddings(batch_def_edc)
        
        # Process each row in the batch
        for i in range(len(batch_def_orig)):
            idx = batch_start + i
            
            try:
                # Get embeddings from batch results
                emb_orig = emb_orig_batch[i]
                emb_edc = emb_edc_batch[i]
                
                # Calculate similarity
                sim_score = cosine_similarity(emb_orig, emb_edc)
                similarities.append(sim_score)
                
            except Exception as e:
                print(f"\nError processing row at index {idx}: {e}")
                similarities.append(None)
        
        # Clean up memory after each batch
        if (batch_start + BATCH_SIZE) % (BATCH_SIZE * 10) == 0:
            clear_memory()

    # Add similarity column to DataFrame
    df_merged['definition_similarity'] = pd.Series(similarities).fillna(-1.0)
    print("\nSimilarity calculation complete.")

    # Prepare output data
    print("Preparing output data...")
    try:
        output_columns = [
            'line_number', 'sentence', 'subject','extracted_subject',
            'object','extracted_object', 'relation','extracted_relation',
            'relation_definition', 'extracted_relation_definition',
            'definition_similarity'
        ]
        df_final = df_merged.rename(columns={'relation_orig': 'relation'})
        final_columns_ordered = [col for col in output_columns if col in df_final.columns]
        df_final = df_final[final_columns_ordered]
    except KeyError as e:
        print(f"Error: Expected column not found in merged DataFrame - {e}.")
        print("Available columns:", df_merged.columns.tolist())
        exit()
    except Exception as e:
        print(f"Error preparing output columns: {e}")
        exit()

    # Write results to CSV
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        
        print(f"Writing results to file: {output_csv}")
        df_final.to_csv(output_csv, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
        print(f"Complete! Results saved to '{output_csv}'.")
        
        # Report stats
        print(f"Processing statistics:")
        print(f"- Total rows processed: {len(df_merged)}")
        print(f"- Average similarity: {np.mean([s for s in similarities if s is not None]):.4f}")
        
    except Exception as e:
        print(f"Error writing final CSV file: {e}")

    # Final cleanup
    clear_memory()
    print(f"{DEVICE.upper()} optimization complete!")


if __name__ == "__main__":
    main()