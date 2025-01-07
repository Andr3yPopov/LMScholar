import re
from openai import OpenAI
import time
import httpx
from transformers import AutoTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the OpenAI client with increased timeout
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio", timeout=httpx.Timeout(1000000.0))  # Увеличенный таймаут


def trim_chat_history(chat_history, max_tokens=500):
    """
    Обрезает историю чата до максимального количества токенов.
    
    Args:
        chat_history (list): История чата в формате [{"role": "user/bot", "content": "текст сообщения"}].
        max_tokens (int): Максимальное количество токенов.
    
    Returns:
        list: Обрезанная история чата.
    """
    total_tokens = 0
    trimmed_history = []
    
    # Идём с конца истории, чтобы сохранить последние сообщения
    for message in reversed(chat_history):
        message_tokens = count_tokens(message["content"])
        if total_tokens + message_tokens > max_tokens:
            break
        trimmed_history.insert(0, message)  # Вставляем в начало, чтобы сохранить порядок
        total_tokens += message_tokens
    
    return trimmed_history
    
def sanitize_text(text):
    """Sanitize the text by replacing or removing control characters."""
    sanitized_text = re.sub(r'[\t\n\r\f\v]', ' ', text)
    return sanitized_text

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

def get_embedding(text, model="second-state/Nomic-embed-text-v1.5-Embedding-GGUF"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def count_tokens(text, tokenizer_path="tokenizer"):
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

def compress_text(text, model="hermes-3-llama-3.2-3b"):
    """
    Сжимает текст, выделяя основную информацию в виде тезисов.
    
    Args:
        text (str): Исходный текст.
        model (str): Модель LLM для обработки текста.
    
    Returns:
        str: Сжатый текст в виде тезисов.
    """
    prompt = f"""
    You are an AI assistant that summarizes text into concise bullet points. Extract the most important information from the following text and present it as bullet points:
    
    Text:
    \"\"\"
    {text}
    \"\"\"
    
    Respond with a list of bullet points summarizing the key points. Do not include any additional explanations or comments.
    """
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )
    
    return response.choices[0].message.content
    
def process_text(text, query, max_tokens=2000):
    token_count = count_tokens(text)
    
    if token_count <= max_tokens:
        return text
    else:
        print(f'Text exceeds {max_tokens} tokens ({token_count} tokens). Processing chunks...')
        
        # Разделение текста на отрывки по 500 токенов
        chunks = split_text_into_chunks(text, max_tokens=500)
        
        # Получение эмбеддинга поискового запроса
        query_embedding = get_embedding(query)
        
        # Вычисление эмбеддингов отрывков
        chunk_embeddings = [get_embedding(c) for c in chunks]
        
        # Вычисление косинусного сходства
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        
        # Сортировка отрывков по косинусному сходству
        sorted_chunks = [c for _, c in sorted(zip(similarities, chunks), reverse=True)]
        
        # Конкатенация наиболее релевантных отрывков
        concatenated_text = "\n\n".join(sorted_chunks[:3])  # Выбираем 3 наиболее релевантных отрывка
        
        return concatenated_text

def basic_chat(query, attached_docs=None, chat_history=None, workspace_info=None, user_info=None):
    """
    Функция для базового чата с использованием системного промпта и учётом контекста (500 последних токенов).
    
    Args:
        query (str): Запрос пользователя.
        attached_docs (List[str]): Прикрепленные документы.
        chat_history (list): История чата в формате [{"role": "user/bot", "content": "текст сообщения"}].
        workspace_info (dict): Информация о команде.
        user_info (dict): Информация о текущем пользователе.
    
    Returns:
        Генератор: Потоковый ответ от модели.
    """
    compressed_docs = [compress_text(doc) for doc in attached_docs] if attached_docs else []
    
    # Формируем контекст из последних 500 токенов
    if chat_history:
        # Преобразуем историю чата в строку
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
        # Обрезаем историю до 500 токенов
        history_str = truncate_text_to_tokens(history_str, max_tokens=500)
    else:
        history_str = ""

    # Формируем системный промпт с учётом информации о команде и пользователе
    system_prompt = f"""
    You are a helpful and intelligent AI assistant. Your goal is to provide clear, concise, and accurate answers to user queries. Follow these guidelines:
    1. Be polite and professional in your responses.
    2. If you don't know the answer, admit it and suggest where the user might find the information.
    3. Keep your responses focused and avoid unnecessary details unless asked.
    4. If the query is ambiguous, ask clarifying questions.
    5. Use simple and understandable language.
    
    **workspace Information:**
    - workspace Name: {workspace_info.get('workspace_name', 'N/A')}
    - Project Description: {workspace_info.get('project_description', 'N/A')}
    
    **User Information:**
    - Name: {user_info.get('name', 'N/A')}
    - Role: {user_info.get('role', 'N/A')}
    - Competencies: {', '.join(user_info.get('competencies', []))}
    """

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    # Добавляем историю чата, если она есть
    if history_str:
        messages.append({"role": "user", "content": history_str})

    # Добавляем текущий запрос пользователя
    messages.append({"role": "user", "content": query})

    # Добавляем прикрепленные документы, если они есть
    if compressed_docs:
        messages.append({"role": "user", "content": "Attached Documents:\n" + "\n".join(compressed_docs)})

    completion = client.chat.completions.create(
        model="hermes-3-llama-3.2-3b",
        messages=messages,
        temperature=0.7,
        stream=True
    )
    
    return completion

def extract_information(texts, query):
    analysis_results = []
    if not texts:
        return []

    for text in texts:
        sanitized_text = sanitize_text(text)
        processed_text = process_text(sanitized_text, query)

        history = [
            {"role": "system", "content": """
            You are an advanced academic writing assistant specializing in the analysis of scientific articles. Your primary task is to critically analyze the provided texts and extract the most relevant information based on the user's query. Follow these key principles when analyzing content:

            1. **Formal and Rigorous Academic Style**: Use a formal tone and adhere to academic standards in your responses.
            2. **Critical Analysis**: Evaluate the provided sources critically, identifying strengths, weaknesses, and potential gaps in the research.
            3. **Novel Insights**: Highlight novel ideas, findings, or methodologies that contribute to the field.
            4. **Relevant Examples**: Provide concrete examples, case studies, or data points to support your analysis.
            5. **Reconciliation of Conflicts**: Address conflicting findings by proposing theories or explanations to resolve contradictions.
            6. **Clarity and Relevance**: Ensure your response is clear, concise, and directly relevant to the user's query. Avoid redundancy and irrelevant information.
            7. **Original Ideas**: Incorporate original hypotheses or ideas where applicable, adding value to the academic discussion.
            8. **Avoid Generalizations**: Focus on detailed analysis rather than broad generalizations.

            When analyzing information from the sources, follow this structure:
            1. **Title and Authors**: Mention the title of the article and the names of the authors.
            2. **Scientific Ideas**: Summarize the key scientific ideas presented in the article, providing a detailed description of each.
            3. **Advantages and Disadvantages**: Discuss the advantages and disadvantages of the ideas or methodologies presented.
            4. **Examples and Illustrations**: Provide examples, case studies, or data points that illustrate the ideas. Include code snippets, formulas, or other relevant details.
            5. **Key Quotes**: Cite key quotes from the article that highlight the most important sections or findings.

            Your goal is to provide a comprehensive and insightful analysis that helps the user understand the article's contributions and implications.
            """},
            {"role": "user", "content": f"The topic of the article is '{query}'. The goal is to analyze the following text and extract the most relevant information based on the user's query, offering critical insights, detailed examples, and identifying gaps in the current research: {processed_text}"}
        ]

        def generate_completion():
            try:
                completion = retry_api_call(lambda: client.chat.completions.create(
                    model="hermes-3-llama-3.2-3b",
                    messages=history,
                    temperature=0.8,
                    stream=True
                ))
                for chunk in completion:
                    content = chunk.choices[0].delta.content or ""
                    if content:
                        yield content
            except Exception as e:
                print(f"Error during API call: {e}")
                yield ""

        analysis_results.append(generate_completion())

    return analysis_results
    