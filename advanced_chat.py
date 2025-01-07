import json
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from download_articles import download_articles
from new_ideas import generate_novel_ideas_for_section
from deconstruction import analyze_text_with_deconstruction
from lib import process_text, retry_api_call, count_tokens, split_text_into_chunks, get_embedding
import re
from flask import Response, stream_with_context

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
intermediate_results = {}

def save_intermediate_results(key, data):
    """
    Сохраняет промежуточные результаты по ключу.
    
    Args:
        key (str): Ключ для сохранения результатов.
        data (dict): Данные для сохранения.
    """
    intermediate_results[key] = data

def get_intermediate_results(key):
    """
    Возвращает сохранённые промежуточные результаты по ключу.
    
    Args:
        key (str): Ключ для получения результатов.
    
    Returns:
        dict: Сохранённые данные или None, если ключ не найден.
    """
    return intermediate_results.get(key)
    
    
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
    
def generate_search_query(user_query, compressed_docs, model="hermes-3-llama-3.2-3b"):
    """
    Формирует корректный поисковый запрос из запроса пользователя с учетом сжатых документов.
    
    Args:
        user_query (str): Запрос пользователя.
        compressed_docs (list): Список сжатых текстов документов.
        model (str): Модель LLM для обработки запроса.
    
    Returns:
        str: Корректный поисковый запрос.
    """
    docs_summary = "\n".join(compressed_docs)
    
    prompt = f"""
    You are an AI assistant that helps to refine search queries based on user input and additional context. Given the user query and a summary of relevant documents, generate a precise search query for finding scientific articles.
    
    User Query:
    \"\"\"
    {user_query}
    \"\"\"
    
    Summary of Relevant Documents:
    \"\"\"
    {docs_summary}
    \"\"\"
    
    Respond with a single search query that combines the user query and the context from the documents. Do not include any additional explanations or comments.
    """
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=100
    )
    
    return response.choices[0].message.content
    



def decide_tools(query, attached_docs):
    """Решает, какие инструменты использовать для обработки запроса."""
    compressed_docs = [compress_text(doc) for doc in attached_docs] if attached_docs else []
    
    prompt = f"""
    You are an AI assistant that decides which tools to use for processing a user query. 
    Available tools:
    1. Idea generation - for generating new ideas
    2. Article analysis - for downloading and analyzing scientific articles
    3. Deconstruction - for deep text analysis and unconventional ideas
    
    Query: {query}
    Attached documents: {compressed_docs}
    
    Respond with a JSON object indicating which tools to use:
    {{
        "idea_generation": boolean,
        "article_analysis": boolean,
        "deconstruction": boolean
    }}
    """
    
    response = client.chat.completions.create(
        model="hermes-3-llama-3.2-3b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=100
    )
    
    response_json = json.loads(response.choices[0].message.content)
    
    def normalize_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() == "true"
        if isinstance(value, int):
            return value == 1
        return False
    
    normalized_tools = {
        "idea_generation": normalize_bool(response_json.get("idea_generation", False)),
        "article_analysis": normalize_bool(response_json.get("article_analysis", False)),
        "deconstruction": normalize_bool(response_json.get("deconstruction", False)),
    }
    
    return normalized_tools

def extract_information(texts, query, workspace_info=None, user_info=None):
    """Анализирует тексты и возвращает результаты анализа в виде строки."""
    analysis_results = []
    if not texts:
        return "No texts provided for analysis."

    for text in texts:
        processed_text = process_text(text, query)

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
            {"role": "user", "content": f"""
            The topic of the article is '{query}'. The goal is to analyze the following text and extract the most relevant information based on the user's query, offering critical insights, detailed examples, and identifying gaps in the current research: {processed_text}

            **Workspace Information:**
            - Workspace Name: {workspace_info.get('workspace_name', 'N/A')}
            - Project Description: {workspace_info.get('project_description', 'N/A')}

            **User Information:**
            - Name: {user_info.get('name', 'N/A')}
            - Role: {user_info.get('role', 'N/A')}
            - Competencies: {', '.join(user_info.get('competencies', []))}
            """}
        ]

        try:
            completion = retry_api_call(lambda: client.chat.completions.create(
                model="hermes-3-llama-3.2-3b",
                messages=history,
                temperature=0.8,
                stream=False
            ))
            analysis_results.append(completion.choices[0].message.content)
        except Exception as e:
            print(f"Error during API call: {e}")
            analysis_results.append("Error: Unable to analyze the text.")

    return "\n\n".join(analysis_results)




def parse_chain_response(response):
    """
    Парсит ответ LLM, определяющий цепочку рассуждений и результаты для сохранения.
    
    Args:
        response (str): Ответ LLM в формате, похожем на JSON.
    
    Returns:
        tuple: (list, list) — список инструментов и список результатов для сохранения.
    
    Raises:
        ValueError: Если ответ не удалось распарсить или он имеет неверный формат.
    """
    try:
        cleaned_response = response.strip()
        cleaned_response = re.sub(r"chain:\s*", '"chain":', cleaned_response)
        cleaned_response = re.sub(r"save:\s*", '"save":', cleaned_response)
        cleaned_response = re.sub(r"'", '"', cleaned_response)
        cleaned_response = re.sub(r"\s+", " ", cleaned_response)
        
        if not cleaned_response.startswith("{"):
            cleaned_response = "{" + cleaned_response + "}"
        
        response_json = json.loads(cleaned_response)
        
        if not isinstance(response_json, dict):
            raise ValueError("Response is not a JSON object.")
        
        chain = response_json.get("chain", [])
        save = response_json.get("save", [])
        
        if not isinstance(chain, list) or not isinstance(save, list):
            raise ValueError("Chain and save should be lists.")
        
        return chain, save
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error: {e}")



def build_chain_prompt(query, attached_docs):
    """
    Формирует промпт для LLM, чтобы определить минимально необходимую цепочку рассуждений и указать, какие результаты сохранить.
    
    Args:
        query (str): Запрос пользователя.
        attached_docs (list): Прикрепленные документы.
    
    Returns:
        str: Промпт для LLM.
    """
    prompt = f"""
    You are an AI assistant that decides which tools to use for processing a user query. 
    Available tools:
    1. Idea generation - for generating new ideas
    2. Article analysis - for downloading and analyzing scientific articles
    3. Deconstruction - for deep text analysis and unconventional ideas
    
    Based on the user query and attached documents, determine the minimal chain of tools to use. 
    Use only the tools that are absolutely necessary to answer the query.
    
    **Rules:**
    1. For simple queries like "generate ideas for an AI project", use only the "Idea generation" tool.
    2. For more complex queries, use no more than three tools.
    3. If the query does not require analysis or deconstruction, do not include those tools.
    4. If no attached documents are provided, avoid using the "Article analysis" tool unless explicitly needed.
    
    Additionally, specify which intermediate results should be saved for future use.
    
    Respond with a JSON object containing two fields:
    1. "chain": A list of tool names in the order they should be used.
    2. "save": A list of tool names whose results should be saved.
    
    **Important Instructions:**
    - Use double quotes (`"`) for all keys and string values in JSON.
    - Do not use single quotes (`'`) in JSON.
    - Ensure the JSON is properly formatted with correct commas and brackets.
    - Do not add any extra text, comments, or explanations outside the JSON object.
    - Avoid trailing commas in lists or objects.
    - Do not include any Markdown syntax (e.g., ```json```).
    - If the response is empty, return an empty JSON object: {{}}.
    
    Example of a valid response for a simple query:
    {{
        "chain": ["Idea generation"],
        "save": ["Idea generation"]
    }}
    
    Example of a valid response for a complex query:
    {{
        "chain": ["Article analysis", "Deconstruction"],
        "save": ["Deconstruction"]
    }}
    
    User Query:
    \"\"\"
    {query}
    \"\"\"
    
    Attached Documents:
    \"\"\"
    {attached_docs}
    \"\"\"
    Your response must **strictly** follow the JSON format. Do not include any additional text, markdown, formatting marks or explanations.
    """
    return prompt

def generate_idea_prompt(user_query, attached_docs, workspace_info, user_info):
    """
    Формирует запрос на генерацию идей с использованием сообщения пользователя, прикреплённых документов,
    информации о рабочем пространстве и данных о пользователе.
    
    Args:
        user_query (str): Сообщение пользователя.
        attached_docs (list): Список прикреплённых документов.
        workspace_info (dict): Информация о рабочем пространстве.
        user_info (dict): Данные о пользователе.
    
    Returns:
        str: Сформированный запрос для генерации идей.
    """
    compressed_docs = [compress_text(doc) for doc in attached_docs] if attached_docs else []
    
    prompt = f"""
    You are an AI assistant that helps to generate ideas based on user input, attached documents, workspace information, and user profile. 
    Use the following information to create a detailed and specific prompt for generating ideas:
    
    **User Query:**
    {user_query}
    
    **Attached Documents:**
    {compressed_docs if compressed_docs else "No attached documents"}
    
    **Workspace Information:**
    - Workspace Name: {workspace_info.get('workspace_name', 'N/A')}
    - Project Description: {workspace_info.get('project_description', 'N/A')}
    
    **User Information:**
    - Name: {user_info.get('name', 'N/A')}
    - Role: {user_info.get('role', 'N/A')}
    - Competencies: {', '.join(user_info.get('competencies', []))}
    
    Create a detailed and specific prompt for generating ideas that combines the user's query, the context from the attached documents, the workspace's project description, and the user's competencies. 
    The prompt should be clear, concise, and directly relevant to the user's needs.
    
    Respond with the prompt only. Do not include any additional explanations or comments.
    """
    
    response = client.chat.completions.create(
        model="hermes-3-llama-3.2-3b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )
    
    return response.choices[0].message.content

def process_query(query, attached_docs=None, chat_history=None, max_retries=3, user_chain=None, user_save_results=None, workspace_info=None, user_info=None):
    """Обрабатывает запрос в расширенном режиме."""
    if attached_docs is None:
        attached_docs = []
    
    if chat_history:
        chat_history = trim_chat_history(chat_history)
    
    if user_chain:
        chain = user_chain
        save_results = user_save_results
    else:
        chain_prompt = build_chain_prompt(query, attached_docs)
        
        retries = 0
        while retries < max_retries:
            try:
                chain_response = client.chat.completions.create(
                    model="hermes-3-llama-3.2-3b",
                    messages=[{"role": "user", "content": chain_prompt}],
                    temperature=0.3,
                    max_tokens=100,
                    stream=False 
                )
                
                chain_response_content = chain_response.choices[0].message.content
                chain, save_results = parse_chain_response(chain_response_content)
                print("Chain of tools (auto-generated):", chain)
                print("Results to save (auto-generated):", save_results)
                break  
            except ValueError as e:
                print(f"Error parsing chain response (attempt {retries + 1}):", e)
                retries += 1
                if retries >= max_retries:
                    return f"Error: Failed to parse chain response after {max_retries} attempts."
                continue
    
    results = {}
    previous_output = '' 

    history_str = ""
    if chat_history:
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])

    for tool in chain:
        print(f"Processing tool: {tool}")
        if tool == "Idea generation":
            # Генерация новых идей
            if isinstance(previous_output, list):
                previous_output = "\n".join(str(item) for item in previous_output)
            # Добавляем историю чата к запросу
            full_query = f"{query}\n\nChat History:\n{history_str}\n\n{previous_output}"
            results['ideas'] = generate_novel_ideas_for_section(full_query, workspace_info=workspace_info, user_info=user_info)
            if tool in save_results:
                save_intermediate_results('ideas', results['ideas'])  
            previous_output = results['ideas']  
        elif tool == "Article analysis":
            compressed_docs = [compress_text(doc) for doc in attached_docs] if attached_docs else []
            search_query = generate_search_query(f"{query}\n\nChat History:\n{history_str}", compressed_docs)
            articles = download_articles(search_query, max_results=1)
            results['articles'] = []
            for article in articles:
                analysis = extract_information([article['text']], f"{query}\n\nChat History:\n{history_str}", workspace_info=workspace_info, user_info=user_info)
                results['articles'].append({"text": article['text'], "analysis": analysis})
            if tool in save_results:
                save_intermediate_results('articles', results['articles'])
            previous_output = results['articles']
        elif tool == "Deconstruction":
            print("DECONSTRUCTION")
            results['deconstruction'] = []
            if isinstance(previous_output, list):
                previous_output = "\n".join(str(item) for item in previous_output)
            input_for_deconstruction = f"{query}\n\nChat History:\n{history_str}\n\n{previous_output}"
            if attached_docs:
                input_for_deconstruction += "\n\nAttached Documents:\n" + "\n\n".join(attached_docs)
            
            filtered_workspace_info = {
                "workspace_name": workspace_info.get('workspace_name', 'N/A'),
                "project_description": workspace_info.get('project_description', 'N/A')
            }
            filtered_user_info = {
                "name": user_info.get('name', 'N/A'),
                "role": user_info.get('role', 'N/A'),
                "competencies": ', '.join(user_info.get('competencies', []))
            }
            
            analysis = analyze_text_with_deconstruction(input_for_deconstruction, workspace_info=filtered_workspace_info, user_info=filtered_user_info)
            results['deconstruction'].append(analysis)
            
            if tool in save_results:
                save_intermediate_results('deconstruction', results['deconstruction']) 
            previous_output = results['deconstruction']  
    
    prompt = f"""
    You are an AI assistant. Use the following information to answer the user's query:
    Query: {query}
    Analysis results: {json.dumps(results)}
    Intermediate results: {json.dumps(intermediate_results)}
    Chat history: {json.dumps(chat_history) if chat_history else "No chat history"}
    
    **workspace Information:**
    - workspace Name: {workspace_info.get('workspace_name', 'N/A')}
    - Project Description: {workspace_info.get('project_description', 'N/A')}
    
    **User Information:**
    - Name: {user_info.get('name', 'N/A')}
    - Role: {user_info.get('role', 'N/A')}
    - Competencies: {', '.join(user_info.get('competencies', []))}
    
    Provide a comprehensive answer based on the analysis and chat history. Do not mention the tools used.
    """
    
    # Генерируем финальный ответ с использованием LLM
    response = client.chat.completions.create(
        model="hermes-3-llama-3.2-3b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        stream=True
    )

    return response