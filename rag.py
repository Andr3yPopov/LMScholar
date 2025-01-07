import os
import json
import numpy as np
from scipy.spatial.distance import cosine
from heapq import nlargest
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def get_embedding(text, model="second-state/Nomic-embed-text-v1.5-Embedding-GGUF"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def read_json_files(directory):
    json_files = [pos_json for pos_json in os.listdir(directory) if pos_json.endswith('.json')]
    data = []
    for file_name in json_files:
        with open(os.path.join(directory, file_name), 'r', encoding='utf-8') as file:
            data.append(json.load(file))
    return data

def calculate_cosine_similarity(query_embedding, embedding):
    return 1 - cosine(query_embedding, embedding)

def find_top_similar_files(directory, query_embedding, top_n=5):
    data = read_json_files(directory)
    similarities = []
    
    for item in data:
        embedding = np.array(item['embedding'])
        similarity = calculate_cosine_similarity(query_embedding, embedding)
        similarities.append((similarity, item['text'], item['pdf_link'], embedding))
    
    top_similarities = nlargest(top_n, similarities, key=lambda x: x[0])
    return top_similarities

directory_path = "articles_json"  # Specify the path to the directory with JSON files
top_n = 1  # Number of files to search

def result(directory_path, input_text, top_n):
    query_embedding = get_embedding(input_text)  # Get embedding from text
    top_files = find_top_similar_files(directory_path, query_embedding, top_n)
    texts = [text for _, text, _, _ in top_files]
    links = [link for _, _, link, _ in top_files]
    return texts, links

