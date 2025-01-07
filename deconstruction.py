import json
from datetime import datetime
from bson import ObjectId
import os
import os.path as osp
import time
from typing import List, Dict, Union
from openai import OpenAI
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

def json_serializer(obj):
    """Custom JSON serializer for objects not serializable by default."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, ObjectId):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")
    
def retry_api_call(func, retries=3, delay=5):
    """Retry the API call if it fails."""
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if attempt < retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise e

def sanitize_text(text):
    """Sanitize the text by replacing or removing control characters."""
    sanitized_text = re.sub(r'[\t\n\r\f\v]', ' ', text)
    return sanitized_text

def extract_keywords(text, num_keywords=5):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.isalnum() and word.lower() not in stop_words]
    
    freq_dist = nltk.FreqDist(filtered_words)
    keywords = [word for word, _ in freq_dist.most_common(num_keywords)]
    return keywords
    
def get_embedding(text, model="second-state/Nomic-embed-text-v1.5-Embedding-GGUF"):
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio", timeout=600.0)
    text = text.replace("\n", " ")
    return retry_api_call(lambda: client.embeddings.create(input=[text], model=model).data[0].embedding)
    
def get_response_from_llm(prompt, model, history=None, temperature=0.7):
    try:
        if history is None:
            history = [{"role": "system", "content": "You are a scholar who analyzes scientific texts and uses the method of deconstruction in your work. Proceed with the given texts as follows: Break the text into parts: Divide the text into separate elements, blocks, or fragments to reveal its structure and hidden contradictions. Rearrange and recombine: Change the order of the text parts either randomly or meaningfully, creating new configurations of meaning. Seek multiplicity of interpretations: Abandon the search for a single 'correct' meaning; instead, explore various possible readings. Critique binary oppositions: Identify and question traditional oppositions (e.g., 'author/reader,' 'form/content'). Activate the role of the reader: Engage yourself as a reader in the process of creating meaning, becoming a co-author of the text. Experiment and play: Use creative and playful methods to uncover new, unexpected interpretations. Respect the text: Remember that deconstruction does not destroy the text but reveals its new facets and potential. Additionally, follow these guidelines to actively develop ideas from previous iterations and filter out unproductive ones: Develop previous ideas: Actively build upon ideas that emerged in earlier iterations. Explore how they can be expanded, refined, or connected to other concepts. Evaluate ideas critically: Assess the productivity of each idea. Discard ideas that are repetitive, irrelevant, or do not contribute to a deeper understanding of the text. Reflect on progress: After each iteration, reflect on which ideas have been most fruitful and why. Use this reflection to guide the next steps in the analysis. Iterative refinement: Continuously revisit and refine previous interpretations, ensuring that each iteration adds depth and clarity to the analysis. Your goal is to create a rich, multi-layered understanding of the text by iteratively deconstructing, recombining, and interpreting its elements, while actively developing productive ideas and discarding those that do not contribute to the analysis."}]

        history.append({"role": "user", "content": prompt})

        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio", timeout=6000.0)

        completion = retry_api_call(lambda: client.chat.completions.create(
            model=model,
            messages=history,
            temperature=temperature,
            max_tokens = 30,
            stream=True,
        ))

        new_message = {"role": "assistant", "content": ""}

        for chunk in completion:
            if chunk.choices[0].delta.content:
                new_message["content"] += chunk.choices[0].delta.content

        history.append(new_message)
        return new_message["content"], history

    except Exception as e:
        print(f"Error in LLM response: {e}")
        return None, history

def deconstruct_text(text, model, num_iterations=3):
    """Deconstruct the text iteratively using the LLM."""
    deconstructions = []
    system_message = "You are a scholar who analyzes scientific texts and uses the method of deconstruction in your work. Proceed with the given texts as follows: Break the text into parts: Divide the text into separate elements, blocks, or fragments to reveal its structure and hidden contradictions. Rearrange and recombine: Change the order of the text parts either randomly or meaningfully, creating new configurations of meaning. Seek multiplicity of interpretations: Abandon the search for a single 'correct' meaning; instead, explore various possible readings. Critique binary oppositions: Identify and question traditional oppositions (e.g., 'author/reader,' 'form/content'). Activate the role of the reader: Engage yourself as a reader in the process of creating meaning, becoming a co-author of the text. Experiment and play: Use creative and playful methods to uncover new, unexpected interpretations. Respect the text: Remember that deconstruction does not destroy the text but reveals its new facets and potential. Additionally, follow these guidelines to actively develop ideas from previous iterations and filter out unproductive ones: Develop previous ideas: Actively build upon ideas that emerged in earlier iterations. Explore how they can be expanded, refined, or connected to other concepts. Evaluate ideas critically: Assess the productivity of each idea. Discard ideas that are repetitive, irrelevant, or do not contribute to a deeper understanding of the text. Reflect on progress: After each iteration, reflect on which ideas have been most fruitful and why. Use this reflection to guide the next steps in the analysis. Iterative refinement: Continuously revisit and refine previous interpretations, ensuring that each iteration adds depth and clarity to the analysis. Your goal is to create a rich, multi-layered understanding of the text by iteratively deconstructing, recombining, and interpreting its elements, while actively developing productive ideas and discarding those that do not contribute to the analysis."
    initial_prompt = f"{system_message}\n\nText to deconstruct:\n\"\"\"\n{text}\n\"\"\"\n\nBegin by breaking the text into parts and analyzing its structure."

    for iteration in range(num_iterations):
        print(f"Deconstruction iteration {iteration + 1}/{num_iterations}")
        response, history = get_response_from_llm(initial_prompt, model=model)
        deconstructions.append(response)

        if iteration < num_iterations - 1:
            initial_prompt = f"Iteration {iteration + 2}/{num_iterations + 10}.\nReflect on the deconstruction so far and try to uncover deeper meanings or new interpretations.\nCurrent deconstruction:\n{response}\nOriginal text:\n{text}\n"

    return deconstructions

def analyze_text_with_deconstruction(text: str, model: str = "hermes-3-llama-3.2-3b", num_iterations: int = 1, attached_docs: List[str] = None, workspace_info=None, user_info=None) -> List[str]:
    """
    Analyze the given text using iterative deconstruction.
    
    Args:
        text (str): The text to analyze.
        model (str): The LLM model to use for deconstruction.
        num_iterations (int): The number of deconstruction iterations.
        attached_docs (List[str]): Attached documents to include in the analysis.
        workspace_info (dict): Filtered information about the workspace.
        user_info (dict): Filtered information about the user.
    
    Returns:
        List[str]: A list of deconstructed interpretations of the text.
    """
    print("DECONSTRUCTION WORKS")
    if attached_docs:
        text += "\n\nAttached Documents:\n" + "\n".join(attached_docs)
    if workspace_info:
        text += f"\n\nworkspace Info:\n{json.dumps(workspace_info, default=json_serializer)}"
    if user_info:
        text += f"\n\nUser Info:\n{json.dumps(user_info, default=json_serializer)}"
    
    return deconstruct_text(text, model=model, num_iterations=num_iterations)