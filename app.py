from flask import Flask, render_template, request, redirect, url_for, session, send_file, Response, stream_with_context
from user_db import StartupHelperDB
from download_articles import download_articles 
from advanced_chat import process_query
from lib import basic_chat
import os
import io
from lib import extract_information
import re
import json

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 10000000
app.secret_key = os.urandom(24)
db_connection_string = "mongodb://localhost:27017/"
db = StartupHelperDB(db_connection_string)
key_file_path = "aes_key.bin"

@app.route('/')
def index():
    if 'email' not in session:
        return redirect(url_for('login'))

    user = db.get_member_by_email(session['email'])
    workspaces = db.get_workspaces_by_creator_email(session['email'])
    member_workspaces = db.get_member_workspaces(session['email'])
    invitations = db.get_invitations_by_email(session['email'])

    created_workspaces = [workspace['workspace_name'] for workspace in workspaces]

    return render_template('index.html', user=user, workspaces=workspaces, member_workspaces=member_workspaces, created_workspaces=created_workspaces, invitations=invitations)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']
        competencies = request.form['competencies'].split(',')

        try:
            db.create_member(name, email, password, role, competencies)
            return redirect(url_for('login'))
        except ValueError as e:
            return f"Registration failed: {e}"
        except Exception as e:
            return f"Registration failed: {e}"

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        try:
            if db.authenticate_member(email, password):
                session['email'] = email
                return redirect(url_for('index'))
            else:
                return "Invalid credentials"
        except Exception as e:
            return f"Login failed: {e}"

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('login'))

@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if 'email' not in session:
        return redirect(url_for('login'))

    user = db.get_member_by_email(session['email'])

    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        role = request.form['role']
        new_competency = request.form.get('new_competency')

        try:
            db.change_member_name(session['email'], name)
            db.change_member_role(session['email'], role)
            if new_competency:
                db.add_competency(session['email'], new_competency)
            return redirect(url_for('edit_profile'))
        except Exception as e:
            return f"Profile update failed: {e}"

    return render_template('edit_profile.html', user=user)

@app.route('/delete_competency/<competency>', methods=['GET'])
def delete_competency(competency):
    if 'email' not in session:
        return redirect(url_for('login'))

    email = session['email']

    try:
        db.delete_competency(email, competency)
        return redirect(url_for('edit_profile'))
    except Exception as e:
        return f"Failed to delete competency: {e}"

@app.route('/create_workspace', methods=['GET', 'POST'])
def create_workspace():
    if 'email' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        workspace_name = request.form['workspace_name']
        description = request.form['description']
        project_description = request.form['project_description']
        creator_email = session['email']

        try:
            db.create_workspace(workspace_name, description, project_description, creator_email)
            return redirect(url_for('index'))
        except ValueError as e:
            return f"Failed to create workspace: {e}"
        except Exception as e:
            return f"Failed to create workspace: {e}"

    return render_template('create_workspace.html')

@app.route('/edit_workspace/<workspace_name>', methods=['GET', 'POST'])
def edit_workspace(workspace_name):
    if 'email' not in session:
        return redirect(url_for('login'))

    workspace = db.get_workspace_by_name(workspace_name)

    if workspace['creator_email'] != session['email']:
        return "You do not have permission to edit this workspace."

    if request.method == 'POST':
        new_workspace_name = request.form['workspace_name']
        description = request.form['description']
        project_description = request.form['project_description']

        try:
            db.change_workspace_name(workspace_name, new_workspace_name)
            db.change_workspace_description(new_workspace_name, description)
            db.change_workspace_project_description(new_workspace_name, project_description)
            return redirect(url_for('index'))
        except Exception as e:
            return f"Failed to edit workspace: {e}"

    members = db.get_members_by_workspace_name(workspace_name)

    return render_template('edit_workspace.html', workspace=workspace, members=members)

@app.route('/invite_member/<workspace_name>', methods=['POST'])
def invite_member(workspace_name):
    if 'email' not in session:
        return redirect(url_for('login'))

    email = request.form['email']

    if email == session['email']:
        return "You cannot invite yourself to the workspace"

    try:
        db.invite_member_to_workspace(workspace_name, email)
        return redirect(url_for('edit_workspace', workspace_name=workspace_name))
    except ValueError as e:
        return f"Failed to invite member: {e}"
    except Exception as e:
        return f"Failed to invite member: {e}"

@app.route('/accept_invitation/<workspace_name>', methods=['POST'])
def accept_invitation(workspace_name):
    if 'email' not in session:
        return redirect(url_for('login'))

    email = session['email']

    try:
        db.accept_invitation(workspace_name, email)
        return redirect(url_for('index'))
    except Exception as e:
        return f"Failed to accept invitation: {e}"

@app.route('/reject_invitation/<workspace_name>', methods=['POST'])
def reject_invitation(workspace_name):
    if 'email' not in session:
        return redirect(url_for('login'))

    email = session['email']

    try:
        db.reject_invitation(workspace_name, email)
        return redirect(url_for('index'))
    except Exception as e:
        return f"Failed to reject invitation: {e}"

@app.route('/view_workspace/<workspace_name>', methods=['GET'])
def view_workspace(workspace_name):
    if 'email' not in session:
        return redirect(url_for('login'))

    workspace_info = db.get_workspace_info_by_name(workspace_name)

    return render_template('view_workspace.html', workspace_info=workspace_info, search_articles_url=url_for('search_articles', workspace_name=workspace_name))
    
@app.route('/leave_workspace/<workspace_name>', methods=['POST'])
def leave_workspace(workspace_name):
    if 'email' not in session:
        return redirect(url_for('login'))

    email = session['email']

    try:
        result = db.leave_workspace(email, workspace_name)
        if result:
            return redirect(url_for('index'))
        else:
            return "Failed to leave workspace"
    except Exception as e:
        return f"Failed to leave workspace: {e}"

@app.route('/remove_member/<workspace_name>/<email>', methods=['POST'])
def remove_member(workspace_name, email):
    if 'email' not in session:
        return redirect(url_for('login'))

    creator_email = session['email']
    workspace = db.get_workspace_by_name(workspace_name)

    if workspace['creator_email'] != creator_email:
        return "You do not have permission to remove members from this workspace."

    try:
        result = db.remove_member_from_workspace(workspace_name, email)
        if result:
            return redirect(url_for('edit_workspace', workspace_name=workspace_name))
        else:
            return "Failed to remove member from workspace"
    except Exception as e:
        return f"Failed to remove member from workspace: {e}"
        
@app.route('/upload_document/<workspace_name>', methods=['GET', 'POST'])
def upload_document(workspace_name):
    if 'email' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'document' not in request.files:
            return "No file part"

        file = request.files['document']
        if file.filename == '':
            return "No selected file"

        if file:
            document_name = file.filename
            document_content = file.read()

            try:
                db.create_document("document", workspace_name, document_name, document_content, key_file_path, session['email'])
                return redirect(url_for('view_workspace', workspace_name=workspace_name))
            except Exception as e:
                return f"Failed to upload document: {e}"

    return render_template('upload_document.html', workspace_name=workspace_name)

@app.route('/download_document/<workspace_name>/<document_name>')
def download_document(workspace_name, document_name):
    if 'email' not in session:
        return redirect(url_for('login'))

    try:
        document_content = db.download_document(workspace_name, document_name, key_file_path)
        if document_content == "Document not found":
            return "Document not found"

        return send_file(
            io.BytesIO(document_content),
            as_attachment=True,
            download_name=document_name
        )
    except Exception as e:
        return f"Failed to download document: {e}"

@app.route('/delete_document/<workspace_name>/<document_name>', methods=['POST'])
def delete_document(workspace_name, document_name):
    if 'email' not in session:
        return redirect(url_for('login'))

    try:
        result = db.delete_document(workspace_name, document_name, session['email'])
        if result:
            return redirect(url_for('view_workspace', workspace_name=workspace_name))
        else:
            return "Failed to delete document"
    except Exception as e:
        return f"Failed to delete document: {e}"

@app.route('/view_document/<workspace_name>/<document_name>')
def view_document(workspace_name, document_name):
    if 'email' not in session:
        return redirect(url_for('login'))

    try:
        document_content = db.get_document_content_as_text(workspace_name, document_name, key_file_path)
        if document_content == "Document not found":
            return "Document not found"
        elif document_content == "Document is not encoded as text":
            return "Document is not encoded as text"
        elif document_content == "Document content is not text":
            return "Document content is not text"

        return render_template('view_document.html', workspace_name=workspace_name, document_name=document_name, document_content=document_content)
    except Exception as e:
        return f"Failed to view document: {e}"
        
@app.route('/delete_workspace/<workspace_name>', methods=['POST'])
def delete_workspace(workspace_name):
    if 'email' not in session:
        return redirect(url_for('login'))

    try:
        result = db.delete_workspace(workspace_name)
        if result:
            return redirect(url_for('index'))
        else:
            return "Failed to delete workspace"
    except Exception as e:
        return f"Failed to delete workspace: {e}"

@app.route('/search_articles/<workspace_name>', methods=['POST'])
def search_articles(workspace_name):
    if 'email' not in session:
        return redirect(url_for('login'))

    topic = request.form['topic']
    year = int(request.form['year'])
    max_results = int(request.form['max_results'])

    try:
        articles = download_articles(topic, year, max_results=max_results, workspace_name=workspace_name, creator_email=session['email'], document_type="article")
        
        articles_html = ""
        for article in articles:
            articles_html += f"""
                <li>
                    <a href="{article['pdf_link']}" target="_blank">{article['title']}</a>
                    <form method="POST" action="{{ url_for('delete_article', workspace_name=workspace_name, article_name=article['title']) }}">
                        <button type="submit">Delete Article</button>
                    </form>
                </li>
            """
        return articles_html
    except Exception as e:
        return f"Failed to download articles: {e}"

@app.route('/delete_article/<workspace_name>/<article_name>', methods=['POST'])
def delete_article(workspace_name, article_name):
    if 'email' not in session:
        return redirect(url_for('login'))

    try:
        result = db.delete_article(workspace_name, article_name, session['email'])
        if result:
            return redirect(url_for('view_workspace', workspace_name=workspace_name))
        else:
            return "Failed to delete article"
    except Exception as e:
        return f"Failed to delete article: {e}"


@app.route('/chat/<workspace_name>', methods=['GET', 'POST'])
def chat(workspace_name):
    if 'email' not in session:
        return redirect(url_for('login'))

    documents = db.get_documents_by_workspace_name(workspace_name)
    articles = db.get_articles_by_workspace_name(workspace_name)

    chat_history = db.load_chat_history(session['email'], key_file_path)
    workspace_info = db.get_workspace_info_by_name(workspace_name)
    user_info = db.get_member_by_email(session['email'])

    if request.method == 'POST':
        query = request.form.get('query')
        selected_document_names = request.form.getlist('document_ids')
        selected_article_names = request.form.getlist('article_ids')
        advanced_mode = request.form.get('advanced_mode', 'off') == 'on'
        user_chain = json.loads(request.form.get('chain', '[]'))
        user_save_results = json.loads(request.form.get('save_results', '[]'))

        texts = []
        docs = selected_document_names + selected_article_names
        docs = [x for x in docs if x != ""]
        for doc_name in docs:
            text = db.get_document_content_as_text(workspace_name, doc_name, key_file_path)
            if text:
                texts.append(text)

        chat_history.append({"role": "user", "content": query})

        if advanced_mode:
            response = process_query(
                query,
                attached_docs=texts,
                chat_history=chat_history,
                user_chain=user_chain,
                user_save_results=user_save_results,
                workspace_info=workspace_info,
                user_info=user_info
            )

            def generate():
                bot_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        bot_response += content
                        yield content.encode('utf-8')

                chat_history.append({"role": "bot", "content": bot_response})

                db.save_chat_history(session['email'], chat_history, key_file_path)

            return Response(stream_with_context(generate()), content_type='text/event-stream')
        else:
            response = basic_chat(query, attached_docs=texts, chat_history=chat_history, workspace_info=workspace_info, user_info=user_info)

            def generate():
                bot_response = ""
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        bot_response += content
                        yield content.encode('utf-8')

                chat_history.append({"role": "bot", "content": bot_response})

                db.save_chat_history(session['email'], chat_history, key_file_path)

            return Response(stream_with_context(generate()), content_type='text/event-stream')

    return render_template('chat.html', workspace_name=workspace_name, documents=documents, articles=articles, chat_history=chat_history)

@app.route('/delete_chat_history/<workspace_name>', methods=['POST'])
def delete_chat_history(workspace_name):
    if 'email' not in session:
        return redirect(url_for('login'))

    try:
        db.delete_chat_history(session['email'])
        return redirect(url_for('chat', workspace_name=workspace_name))
    except Exception as e:
        return f"Failed to delete chat history: {e}"
        
if __name__ == '__main__':
    ssl_cert = 'cert.pem'
    ssl_key = 'key.pem'
    app.run(host='0.0.0.0', port=443, ssl_context=(ssl_cert, ssl_key))