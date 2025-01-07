import json5
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
from download_articles import download_articles
import re
import json
import json5

nltk.download('stopwords')
nltk.download('punkt')

def extract_json_from_response(text):
    # Ищем JSON в тексте с помощью регулярного выражения
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return match.group(0)
    return None
    
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
    
    # Простая частотная модель: выбираем наиболее часто встречающиеся слова
    freq_dist = nltk.FreqDist(filtered_words)
    keywords = [word for word, _ in freq_dist.most_common(num_keywords)]
    return keywords
    
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio", timeout=600.0)

def get_embedding(text, model="second-state/Nomic-embed-text-v1.5-Embedding-GGUF", client=client):
    text = text.replace("\n", " ")
    return retry_api_call(lambda: client.embeddings.create(input=[text], model=model).data[0].embedding)

def get_response_from_llm(prompt, model, history=None, temperature=0.7, client=client):
    try:
        if history is None:
            history = [{"role": "system", "content": "You are an intelligent assistant."}]

        history.append({"role": "user", "content": prompt})

        completion = retry_api_call(lambda: client.chat.completions.create(
            model=model,
            messages=history,
            temperature=temperature,
            max_tokens=500,
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

def generate_ideas_for_paper(query, model, num_reflections=3, max_attempts=1, max_retries=3, workspace_info=None, user_info=None):
    idea_archive = []
    system_message = "You are a creative assistant helping to generate innovative ideas based on a given query. Provide unique and practical ideas that address the query."

    # Добавляем информацию о проекте и пользователе в промпт
    project_description = workspace_info.get('project_description', 'N/A') if workspace_info else 'N/A'
    user_role = user_info.get('role', 'N/A') if user_info else 'N/A'
    user_competencies = ', '.join(user_info.get('competencies', [])) if user_info else 'N/A'

    # Первоначальный запрос с шаблоном выходных данных
    initial_prompt = f"""
    {system_message}

    Query:
    \"\"\"
    {query}
    \"\"\"

    Project Description:
    \"\"\"
    {project_description}
    \"\"\"

    User Role:
    \"\"\"
    {user_role}
    \"\"\"

    User Competencies:
    \"\"\"
    {user_competencies}
    \"\"\"

    Generate ideas to address this query. Respond **only** with a JSON array of ideas, where each idea is an object with a 'title' and 'description' field.

    Example of the expected output format:
    [
        {{
            "title": "Idea 1",
            "description": "This is the first idea."
        }},
        {{
            "title": "Idea 2",
            "description": "This is the second idea."
        }}
    ]

    Your response must **strictly** follow this JSON format. Do not include any additional text or explanations.
    """

    for attempt in range(max_attempts):
        try:
            print(f"Generating idea {len(idea_archive) + 1}/{max_attempts}")
            text, msg_history = get_response_from_llm(initial_prompt, model=model)
            print(text)
            
            # Извлекаем JSON из ответа
            json_text = extract_json_from_response(text)
            if not json_text:
                print("No valid JSON found in the response.")
                continue

            # Парсим JSON с повторными попытками
            for retry in range(max_retries):
                try:
                    ideas_json = json5.loads(json_text)
                    ideas = [f"{idea['title']}: {idea['description']}" for idea in ideas_json]
                    idea_archive.extend(ideas)
                    break  # Успешно декодировали JSON, выходим из цикла повторных попыток
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Failed to parse JSON (attempt {retry + 1}/{max_retries}): {e}")
                    if retry < max_retries - 1:
                        # Повторная генерация JSON
                        text, msg_history = get_response_from_llm(initial_prompt, model=model, history=msg_history)
                        json_text = extract_json_from_response(text)
                        if not json_text:
                            print("No valid JSON found in the response during retry.")
                            continue
                    else:
                        print("Max retries reached. Skipping this attempt.")
                        continue

            # Улучшение идей через рефлексию
            for i in range(1, num_reflections):
                reflection_prompt = f"""
                Iteration {i + 1}/{num_reflections}.
                Reflect on the ideas generated so far and try to improve or refine them.
                Current ideas:
                {', '.join(idea_archive)}

                Respond **only** with a JSON array of improved ideas, where each idea is an object with a 'title' and 'description' field.
                Your response must **strictly** follow the JSON format. Do not include any additional text or explanations.
                """

                text, msg_history = get_response_from_llm(reflection_prompt, model=model, history=msg_history)
                json_text = extract_json_from_response(text)
                if not json_text:
                    print("No valid JSON found in the response during reflection.")
                    continue

                # Парсим JSON с повторными попытками
                for retry in range(max_retries):
                    try:
                        new_ideas_json = json5.loads(json_text)
                        new_ideas = [f"{idea['title']}: {idea['description']}" for idea in new_ideas_json]
                        idea_archive = list(set(idea_archive + new_ideas))
                        break  # Успешно декодировали JSON, выходим из цикла повторных попыток
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"Failed to parse JSON during reflection (attempt {retry + 1}/{max_retries}): {e}")
                        if retry < max_retries - 1:
                            # Повторная генерация JSON
                            text, msg_history = get_response_from_llm(reflection_prompt, model=model, history=msg_history)
                            json_text = extract_json_from_response(text)
                            if not json_text:
                                print("No valid JSON found in the response during retry.")
                                continue
                        else:
                            print("Max retries reached. Skipping this reflection.")
                            continue

                if "I am done" in text:
                    print(f"Idea generation converged after {i + 1} iterations.")
                    break

            break
        except Exception as e:
            print(f"Failed to generate ideas: {e}")
            continue

    return idea_archive

def check_idea_novelty(ideas, max_num_iterations=5, json_dir='articles_for_ideas') -> List[str]:
    novel_ideas = []  # Список для хранения новых идей

    for idx, idea in enumerate(ideas):
        print(f"\nChecking novelty of idea {idx + 1}: {idea}")
        novel = True
        
        # Получаем эмбеддинг идеи
        try:
            idea_embedding = get_embedding(idea)
        except Exception as e:
            print(f"Failed to get embedding for idea {idx + 1}: {e}")
            continue
        keywords = extract_keywords(idea)
        keyword_query = " ".join(keywords)
        # Скачиваем статьи для текущей идеи
        download_articles(keyword_query, json_dir=json_dir, max_results=1, save_to_db=0)

        # Проверяем косинусное сходство с каждой загруженной статьей
        article_files = [osp.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]

        for article_file in article_files:
            try:
                with open(article_file, 'r') as f:
                    article_data = json.load(f)
                
                article_embedding = np.array(article_data['embedding'])

                # Вычисляем косинусное сходство
                similarity = cosine_similarity([idea_embedding], [article_embedding])[0][0]

                if similarity > 0.8:
                    print(f"Idea {idx + 1} is not novel. Similarity: {similarity:.2f}")
                    novel = False
                    break  # Прекращаем проверку, если нашли схожую статью
            except Exception as e:
                print(f"Error processing article {article_file}: {e}")
                continue

        if novel:
            print(f"Idea {idx + 1} is novel.")
            novel_ideas.append(idea)

    return novel_ideas

def generate_novel_ideas_for_section(section: str, model: str = "hermes-3-llama-3.2-3b", num_reflections: int = 1, max_attempts: int = 1, max_num_iterations: int = 1, workspace_info=None, user_info=None) -> List[str]:
    """
    Функция генерирует новые идеи для раздела научной статьи и проверяет их новизну.

    Args:
        section (str): Раздел научной статьи.
        model (str): Модель LLM, используемая для генерации идей.
        num_reflections (int): Количество циклов улучшения идей.
        max_attempts (int): Максимальное количество попыток генерации идей.
        max_num_iterations (int): Максимальное количество итераций проверки новизны.
        workspace_info (dict): Информация о проекте.
        user_info (dict): Информация о пользователе.

    Returns:
        List[str]: Список новых идей, признанных новыми.
    """
    # Генерация идей
    ideas = generate_ideas_for_paper(section, model=model, num_reflections=num_reflections, max_attempts=max_attempts, workspace_info=workspace_info, user_info=user_info)
    
    # Проверка идей на новизну
    novel_ideas = check_idea_novelty(ideas, max_num_iterations=max_num_iterations)
    
    return novel_ideas
#print(generate_novel_ideas_for_section("AI project with LLM"))