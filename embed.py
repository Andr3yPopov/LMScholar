import os
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def get_embedding(text, model="second-state/Nomic-embed-text-v1.5-Embedding-GGUF"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def process_files(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith(".txt"):
            input_filepath = os.path.join(input_directory, filename)
            with open(input_filepath, 'r', encoding='utf-8') as file:
                text = file.read()
            
            embedding = get_embedding(text)
            data = {
                "embedding": embedding,
                "text": text
            }
            
            output_filename = f"{os.path.splitext(filename)[0]}.json"
            output_filepath = os.path.join(output_directory, output_filename)
            with open(output_filepath, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, ensure_ascii=False, indent=4)
            
            print(f"Processed {filename} and saved to {output_filename}")


if __name__ == "__main__":
    input_directory = "articles_txt"
    output_directory = "articles_json"
    process_files(input_directory, output_directory)
