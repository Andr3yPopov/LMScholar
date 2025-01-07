import arxiv
import requests
import re
import fitz
import os
import json
from openai import OpenAI
from transformers import AutoTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from user_db import StartupHelperDB
from datetime import datetime, timezone

def is_article_in_db(workspace_name, article_title):
    """
    Проверяет, существует ли статья с таким же названием в базе данных для данной команды.
    
    Args:
        workspace_name (str): Название команды.
        article_title (str): Название статьи.
    
    Returns:
        bool: True, если статья уже существует, иначе False.
    """
    db_connection_string = "mongodb://localhost:27017/"
    db = StartupHelperDB(db_connection_string)
    
    try:
        workspace = db.get_workspace_by_name(workspace_name)
        if not workspace:
            raise ValueError("Workspace not found")
        
        collection = db.init_collection('documents')
        article = collection.find_one({"workspace_id": workspace["_id"], "document_name": article_title})
        
        return article is not None
    except Exception as e:
        print(f"Error checking if article is in DB: {e}")
        return False
        
def save_article_to_db(workspace_name, article_title, article_url, article_content, creator_email, document_type="article"):
    db_connection_string = "mongodb://localhost:27017/"
    db = StartupHelperDB(db_connection_string)
        
    try:
        print("start of function")
        workspace = db.get_workspace_by_name(workspace_name)
        if not workspace:
            raise ValueError("workspace not found")
        collection = db.init_collection('documents')
        existing_article = collection.find_one({
            "workspace_id": workspace["_id"],
            "document_name": article_title,
            "document_type": document_type
        })
        
        if existing_article:
            print(f"Article '{article_title}' already exists in the database. Skipping save.")
            return False

        key_file_path = "aes_key.bin"
        db.load_aes_key_from_file(key_file_path)
        encrypted_content = db.encrypt_aes(article_content.encode('utf-8'))
        article_data = {
            "workspace_id": workspace["_id"],
            "document_name": article_title,
            "document_url": article_url,
            "document_content": encrypted_content,
            "document_type": document_type,
            "creator_email": creator_email,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }

        db.init_collection('documents').insert_one(article_data)
        print("stamp 3")
        print(f"Article '{article_title}' saved to database as a single object.")
        return True
    except Exception as e:
        print(f"Failed to save article to database: {e}")
        import traceback
        traceback.print_exc()
        raise
            
def get_embedding(text, model="second-state/Nomic-embed-text-v1.5-Embedding-GGUF"):
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def count_tokens(text, tokenizer_path="tokenizer"):
    # Загрузка токенизатора из локальной папки
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokens = tokenizer.encode(text)
    return len(tokens)

def truncate_text_to_tokens(text, max_tokens, tokenizer_path="tokenizer"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = tokenizer.decode(tokens)
    return text

def split_text_into_chunks(text, max_tokens=500, tokenizer_path="tokenizer"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    current_token_count = 0
    
    for paragraph in paragraphs:
        paragraph_tokens = tokenizer.encode(paragraph)
        if current_token_count + len(paragraph_tokens) > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
            current_token_count = len(paragraph_tokens)
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
            current_token_count += len(paragraph_tokens)
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def download_articles(topic, year=2020, pdf_dir='pdf_files', txt_dir='txt_files', json_dir='json_files', max_results=20, workspace_name=None, creator_email=None, document_type="article", save_to_db=1):
    client = arxiv.Client()
    
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    
    search_max_results = max_results * 3
    
    search = arxiv.Search(
        query=topic,
        max_results=search_max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    articles = []
    
    for result in client.results(search):
        publication_year = result.published.year
        if publication_year >= year:
            print(f'Title: {result.title}')
            print(f'PDF Link: {result.pdf_url}')
            
            pdf_filename = re.sub(r'[^\w\s]', '', result.title)
            pdf_filename = re.sub(r'\s+', '_', pdf_filename).strip()
            pdf_path = os.path.join(pdf_dir, f"{pdf_filename}.pdf")
            txt_filename = f"{pdf_filename}.txt"
            txt_path = os.path.join(txt_dir, txt_filename)
            json_filename = f"{pdf_filename}.json"
            json_path = os.path.join(json_dir, json_filename)
            
            try:
                if workspace_name and is_article_in_db(workspace_name, result.title):
                    print(f"Article '{result.title}' already exists in the database. Skipping download.")
                    continue
                
                pdf_response = requests.get(result.pdf_url)
                pdf_response.raise_for_status()
                
                with open(pdf_path, 'wb') as f:
                    f.write(pdf_response.content)
                
                try:
                    with fitz.open(pdf_path) as doc:
                        with open(txt_path, 'w', encoding='utf-8') as txt_file:
                            txt_file.write(f"Title: {result.title}\n\n")
                            for page_num in range(len(doc)):
                                page = doc.load_page(page_num)
                                text = page.get_text()
                                txt_file.write(text)
                    
                    with open(txt_path, 'r', encoding='utf-8') as txt_file:
                        text = txt_file.read()
                    
                    text = truncate_text_to_tokens(text, max_tokens=1000)
                    
                    token_count = count_tokens(text)
                    if token_count > 1000:
                        print(f'Article "{result.title}" exceeds 2000 tokens ({token_count} tokens). Processing chunks...')
                        
                        chunks = split_text_into_chunks(text, max_tokens=500)
                        query_embedding = get_embedding(topic)
                        chunk_embeddings = [get_embedding(c) for c in chunks]
                        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
                        sorted_chunks = [c for _, c in sorted(zip(similarities, chunks), reverse=True)]
                        
                        concatenated_text = "\n\n".join(sorted_chunks[:3])
                        
                        concatenated_embedding = get_embedding(concatenated_text)
                        
                        json_data = {
                            "embedding": concatenated_embedding,
                            "text": concatenated_text,
                            "pdf_link": result.pdf_url,
                            "pdf_path": pdf_path,
                            "txt_path": txt_path
                        }
                        
                        with open(json_path, 'w', encoding='utf-8') as json_file:
                            json.dump(json_data, json_file, ensure_ascii=False, indent=4)
                        
                        print(f'Article saved as {json_path}')
                    else:
                        embedding = get_embedding(text)
                        json_data = {
                            "embedding": embedding,
                            "text": text,
                            "pdf_link": result.pdf_url,
                            "pdf_path": pdf_path,
                            "txt_path": txt_path
                        }
                        
                        with open(json_path, 'w', encoding='utf-8') as json_file:
                            json.dump(json_data, json_file, ensure_ascii=False, indent=4)
                        
                        print(f'Article saved as {json_path}')

                    if save_to_db == 1 and workspace_name and creator_email:
                        print("######################")
                        save_article_to_db(
                            workspace_name,
                            result.title,
                            result.pdf_url,
                            text,
                            creator_email
                        )
                    
                    articles.append({
                        "title": result.title,
                        "text": text,
                        "pdf_link": result.pdf_url,
                        "pdf_path": pdf_path,
                        "txt_path": txt_path
                    })
                
                except RuntimeError as e:
                    print(f"Error processing PDF file {pdf_path}: {e}")
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                    if os.path.exists(txt_path):
                        os.remove(txt_path)
                    if os.path.exists(json_path):
                        os.remove(json_path)
                
            except requests.RequestException as e:
                print(f"Error downloading PDF file: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                if os.path.exists(json_path):
                    os.remove(json_path)
    
    if articles:
        query_embedding = get_embedding(topic)
        article_embeddings = [get_embedding(article["text"]) for article in articles]
        similarities = cosine_similarity([query_embedding], article_embeddings)[0]
        
        sorted_articles = [article for _, article in sorted(zip(similarities, articles), reverse=True)]
        
        return sorted_articles[:max_results]
    else:
        print("No articles found.")
        return []
                    
def process_text(text, query, max_tokens=1000):
    token_count = count_tokens(text)
    
    if token_count <= max_tokens:
        return text
    else:
        print(f'Text exceeds {max_tokens} tokens ({token_count} tokens). Processing chunks...')
        chunks = split_text_into_chunks(text, max_tokens=500)
        query_embedding = get_embedding(query)
        chunk_embeddings = [get_embedding(c) for c in chunks]
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        sorted_chunks = [c for _, c in sorted(zip(similarities, chunks), reverse=True)]
        concatenated_text = "\n\n".join(sorted_chunks[:3])
        
        return concatenated_text
