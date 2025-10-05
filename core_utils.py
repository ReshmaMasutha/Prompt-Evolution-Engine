# core_utils.py

import os
import json
import numpy as np
import textstat
import time
from sentence_transformers import SentenceTransformer

# NOTE: The OpenAI imports are kept but the API key line is commented out 
# to ensure it doesn't even try to connect, though they aren't used in the core functions below.
# import openai 
# openai.api_key = os.getenv('OPENAI_API_KEY') 
embed_model = SentenceTransformer('all-MiniLM-L6-v2') 

# --- Core Utility Functions ---

def generate_variations_with_meta(base_prompt, n=6, model='gpt-4o-mini'):
    """
    REVISED: Replaces the OpenAI call with a simple, cost-free local function.
    It generates placeholder variations to keep the app structure working.
    """
    
    # 1. Create a delay to simulate the time an API call would take (optional)
    time.sleep(1.5) 

    # 2. Generate placeholder prompt variations based on the base prompt
    #    The 'tags' field is included to match the original expected JSON structure.
    variations = []
    
    # Generate the requested number of variations (n)
    for i in range(1, n + 1):
        # Create a placeholder prompt that is clearly an evolved version
        new_prompt = f"Optimize the original prompt for a professional audience: {base_prompt} - V{i}"
        
        # Add a unique ID and a placeholder tag
        variations.append({
            'id': i, 
            'prompt': new_prompt, 
            'tags': f'Placeholder, V{i}'
        })
        
    return variations


def get_model_response(prompt, model='gpt-4o-mini', temperature=0.7):
    """
    REVISED: Replaces the OpenAI call with a static, cost-free placeholder response.
    It simulates a successful LLM output.
    """
    # Simulate a delay
    time.sleep(0.5) 

    # Create a generic placeholder response that includes the original prompt for context
    placeholder_response = (
        f"This is a placeholder response generated without using a paid API key. "
        f"The prompt used was: '{prompt}'. "
        f"In a live version, this output would be the full, detailed answer from the LLM. "
        f"For demonstration, we are providing a medium-length text that can be scored."
        f"\n\nHere is some filler text to ensure the Length Score metric works: "
        f"The system is functional, but the content is static for cost reasons. "
        f"The main purpose of the evolution engine is to test the prompt structure itself, "
        f"which this framework allows us to do with dummy responses."
    )
    return placeholder_response


# --- The rest of the scoring functions remain the same as they use local libraries only ---

def semantic_similarity(text_a, text_b):
    """Calculates cosine similarity between two text embeddings (0.0 to 1.0)."""
    try:
        # NOTE: This uses the local Sentence Transformer, which is FREE and fast.
        embeddings = embed_model.encode([text_a, text_b])
        a_emb = embeddings[0]
        b_emb = embeddings[1]
        
        dot_product = np.dot(a_emb, b_emb)
        norm_product = np.linalg.norm(a_emb) * np.linalg.norm(b_emb)
        
        if norm_product == 0:
            return 0.0
        
        # We cap the output to 4 decimal places for clean formatting
        return round(float(dot_product / norm_product), 4)
    except Exception as e:
        print(f"Similarity error: {e}")
        return 0.0


def calculate_metrics(base_prompt, response_text):
    """Calculates all scoring metrics for a single response."""
    
    # --- 1. Semantic Relevance (S) ---
    S = semantic_similarity(base_prompt, response_text)
    
    # --- 2. Readability (R) ---
    try:
        # Flesch score is a local calculation (FREE)
        flesch_score = textstat.flesch_reading_ease(response_text)
        # Normalize the score between 0.0 and 1.0 (60 is high, 30 is low)
        R = min(1.0, max(0.0, (flesch_score - 30) / 60)) 
    except:
        R = 0.5 # Fallback if textstat fails
    
    # --- 3. Length Appropriateness (L) ---
    word_count = len(response_text.split())
    # Normalize the score (1.0 is 100 words, more is capped at 1.0)
    L = min(1.0, word_count / 100)
    
    # --- 4. Final Weighted Score ---
    FINAL_SCORE = (0.40 * S) + (0.30 * R) + (0.30 * L) 
    
    return {
        'Semantic_Relevance': round(S * 100, 1), # Return as percentage for app.py
        'Readability': round(R * 100, 1),     # Return as percentage for app.py
        'Length_Score': round(L * 100, 1),    # Return as percentage for app.py
        'Final_Score': round(FINAL_SCORE * 100, 1) # Return as percentage for app.py
    }