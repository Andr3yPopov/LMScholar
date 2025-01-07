import pymongo
import hashlib
from datetime import datetime, timezone
import logging
import base64
import os
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import fitz
import json

logging.basicConfig(level=logging.ERROR)

class StartupHelperDB:
    def __init__(self, db_connection_string):
        try:
            self.client = pymongo.MongoClient(db_connection_string)
            self.db = self.client['startup_helper']
            self.aes_key = None
        except pymongo.errors.ConnectionFailure as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            raise

    def init_collection(self, collection_name):
        try:
            return self.db[collection_name]
        except pymongo.errors.InvalidName as e:
            logging.error(f"Invalid collection name: {e}")
            raise

    def generate_aes_key(self, password, salt=None):
        try:
            if salt is None:
                salt = os.urandom(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            self.aes_key = kdf.derive(password.encode())
            return base64.b64encode(salt).decode('utf-8')
        except Exception as e:
            logging.error(f"Failed to generate AES key: {e}")
            raise

    def save_aes_key_to_file(self, file_path):
        try:
            if self.aes_key is None:
                raise ValueError("AES key is not generated")
            with open(file_path, 'wb') as key_file:
                key_file.write(self.aes_key)
        except Exception as e:
            logging.error(f"Failed to save AES key to file: {e}")
            raise

    def load_aes_key_from_file(self, file_path):
        try:
            with open(file_path, 'rb') as key_file:
                self.aes_key = key_file.read()
        except Exception as e:
            logging.error(f"Failed to load AES key from file: {e}")
            raise

    def encrypt_aes(self, plaintext, iv=None):
        try:
            if iv is None:
                iv = os.urandom(16)
            cipher = Cipher(algorithms.AES(self.aes_key), modes.CFB(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(plaintext) + encryptor.finalize()
            return base64.b64encode(iv + ciphertext)
        except Exception as e:
            logging.error(f"Failed to encrypt data: {e}")
            raise

    def decrypt_aes(self, ciphertext):
        try:
            ciphertext = base64.b64decode(ciphertext)
            iv = ciphertext[:16]
            actual_ciphertext = ciphertext[16:]
            cipher = Cipher(algorithms.AES(self.aes_key), modes.CFB(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            plaintext = decryptor.update(actual_ciphertext) + decryptor.finalize()
            return plaintext
        except Exception as e:
            logging.error(f"Failed to decrypt data: {e}")
            raise
            
    def create_workspace(self, workspace_name, description, project_description, creator_email):
        try:
            collection = self.init_collection('workspaces')
            if collection.find_one({"workspace_name": workspace_name}):
                raise ValueError("workspace name already exists")
            workspace = {
                "workspace_name": workspace_name,
                "description": description,
                "project_description": project_description,
                "creator_email": creator_email,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            workspace_id = collection.insert_one(workspace).inserted_id

            self.assign_member_to_workspace(creator_email, workspace_name)

            return workspace_id
        except Exception as e:
            logging.error(f"Failed to create workspace: {e}")
            raise

    def get_workspace_by_name(self, workspace_name):
        try:
            collection = self.init_collection('workspaces')
            return collection.find_one({"workspace_name": workspace_name})
        except Exception as e:
            logging.error(f"Failed to get workspace by name: {e}")
            raise

    def change_workspace_name(self, current_workspace_name, new_workspace_name):
        try:
            collection = self.init_collection('workspaces')
            workspace = collection.find_one({"workspace_name": current_workspace_name})
            if not workspace:
                return "workspace not found"
            collection.update_one({"workspace_name": current_workspace_name}, {"$set": {"workspace_name": new_workspace_name}})
            return True
        except Exception as e:
            logging.error(f"Failed to change workspace name: {e}")
            raise

    def change_workspace_description(self, workspace_name, new_description):
        try:
            collection = self.init_collection('workspaces')
            workspace = collection.find_one({"workspace_name": workspace_name})
            if not workspace:
                return "workspace not found"
            collection.update_one({"workspace_name": workspace_name}, {"$set": {"description": new_description}})
            return True
        except Exception as e:
            logging.error(f"Failed to change workspace description: {e}")
            raise

    def change_workspace_project_description(self, workspace_name, new_project_description):
        try:
            collection = self.init_collection('workspaces')
            workspace = collection.find_one({"workspace_name": workspace_name})
            if not workspace:
                return "workspace not found"
            collection.update_one({"workspace_name": workspace_name}, {"$set": {"project_description": new_project_description}})
            return True
        except Exception as e:
            logging.error(f"Failed to change workspace project description: {e}")
            raise

    def change_workspace_creator_email(self, workspace_name, new_creator_email):
        try:
            collection = self.init_collection('workspaces')
            workspace = collection.find_one({"workspace_name": workspace_name})
            if not workspace:
                return "workspace not found"
            collection.update_one({"workspace_name": workspace_name}, {"$set": {"creator_email": new_creator_email}})
            return True
        except Exception as e:
            logging.error(f"Failed to change workspace creator email: {e}")
            raise

    def create_member(self, name, email, password, role, competencies):
        try:
            collection = self.init_collection('members')
            if collection.find_one({"email": email}):
                raise ValueError("Email already exists")
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            member = {
                "name": name,
                "email": email,
                "password_hash": password_hash,
                "role": role,
                "competencies": competencies,
                "workspaces": [],  
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            return collection.insert_one(member).inserted_id
        except Exception as e:
            logging.error(f"Failed to create member: {e}")
            raise
            
    def assign_member_to_workspace(self, email, workspace_name):
        try:
            workspace = self.get_workspace_by_name(workspace_name)
            if not workspace:
                raise ValueError("workspace not found")

            collection = self.init_collection('members')
            result = collection.update_one({"email": email}, {"$addToSet": {"workspaces": workspace_name}})
            return result.modified_count > 0
        except Exception as e:
            logging.error(f"Failed to assign member to workspace: {e}")
            raise

    def get_member(self, member_id):
        try:
            collection = self.init_collection('members')
            return collection.find_one({"_id": member_id})
        except Exception as e:
            logging.error(f"Failed to get member: {e}")
            raise

    def get_members_by_workspace_name(self, workspace_name):
        try:
            workspace = self.get_workspace_by_name(workspace_name)
            if not workspace:
                return "workspace not found"

            creator_email = workspace["creator_email"]

            members_collection = self.init_collection('members')
            members = members_collection.find({"workspaces": workspace_name})

            members_info = []
            for member in members:
                member_info = {
                    "name": member["name"],
                    "email": member["email"],
                    "role": member["role"],
                    "competencies": member["competencies"],
                    "created_at": member["created_at"],
                    "updated_at": member["updated_at"]
                }
                members_info.append(member_info)

            return members_info
        except Exception as e:
            logging.error(f"Failed to get members by workspace name: {e}")
            raise

    def create_document(self, document_type, workspace_name, document_name, document_content, key_file_path, creator_email):
        try:
            workspace = self.get_workspace_by_name(workspace_name)
            if not workspace:
                raise ValueError("workspace not found")

            self.load_aes_key_from_file(key_file_path)
            encrypted_content = self.encrypt_aes(document_content)

            collection = self.init_collection('documents')
            document = {
                "workspace_id": workspace["_id"],
                "document_name": document_name,
                "document_content": encrypted_content,
                "creator_email": creator_email,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "document_type": document_type
            }
            return collection.insert_one(document).inserted_id
        except Exception as e:
            logging.error(f"Failed to create document: {e}")
            raise

    def get_document(self, workspace_name, document_name):
        try:
            workspace = self.get_workspace_by_name(workspace_name)
            if not workspace:
                return "workspace not found"
            workspace_id = workspace["_id"]
            collection = self.init_collection('documents')
            return collection.find_one({"workspace_id": workspace_id, "document_name": document_name})
        except Exception as e:
            logging.error(f"Failed to get document: {e}")
            raise

    def get_document_content(self, document_id, document_type):
        try:
            document = self.get_document(document_id)
            if not document:
                print("DB STAMP 1")
                return "Document not found"

            encrypted_content = document["document_content"]
            decode_as_text = document.get("decode_as_text", False)

            if decode_as_text:
                decrypted_content = self.decrypt_aes(encrypted_content)
                return decrypted_content
            else:
                return encrypted_content
        except Exception as e:
            logging.error(f"Failed to get document content: {e}")
            raise

    def encrypt_and_save_document(self, document_type, key_file_path, workspace_name=None, document_name=None, document_content=None, document_url=None, annotation=None, generated_by_ai=False, decode_as_text=True):
        try:
            self.load_aes_key_from_file(key_file_path)
            if document_content:
                document_content = self.encrypt_aes(document_content)
            return self.create_document(document_type, workspace_name, document_name, document_content, document_url, annotation, generated_by_ai, decode_as_text)
        except Exception as e:
            logging.error(f"Failed to encrypt and save document: {e}")
            raise

    def decrypt_and_read_document(self, workspace_name, document_name, key_file_path):
        try:
            self.load_aes_key_from_file(key_file_path)
            document = self.get_document(workspace_name, document_name)
            if not document:
                print("DB STAMP 2")
                return "Document not found"

            encrypted_content = document["document_content"]
            decode_as_text = document.get("decode_as_text", False)

            if decode_as_text:
                decrypted_content = self.decrypt_aes(encrypted_content)
                return decrypted_content
            else:
                return encrypted_content
        except Exception as e:
            logging.error(f"Failed to decrypt and read document: {e}")
            raise

    def get_workspace_documents(self, workspace_name):
        try:
            workspace = self.get_workspace_by_name(workspace_name)
            if not workspace:
                return "workspace not found"
            workspace_id = workspace["_id"]

            collection = self.init_collection('documents')
            documents = collection.find({"workspace_id": workspace_id})

            document_names = [doc["document_name"] for doc in documents]
            return document_names
        except Exception as e:
            logging.error(f"Failed to get workspace documents: {e}")
            raise

    def authenticate_member(self, email, password):
        try:
            collection = self.init_collection('members')
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            member = collection.find_one({"email": email, "password_hash": password_hash})
            return member is not None
        except Exception as e:
            logging.error(f"Failed to authenticate member: {e}")
            raise

    def get_member_by_email(self, email):
        try:
            collection = self.init_collection('members')
            return collection.find_one({"email": email})
        except Exception as e:
            logging.error(f"Failed to get member by email: {e}")
            raise
            
    def delete_document(self, workspace_name, document_name):
       try:
           workspace = self.get_workspace_by_name(workspace_name)
           if not workspace:
               return "workspace not found"
           workspace_id = workspace["_id"]
           collection = self.init_collection('documents')
           result = collection.delete_one({"workspace_id": workspace_id, "document_name": document_name})
           return result.deleted_count > 0
       except Exception as e:
           logging.error(f"Failed to delete document: {e}")
           raise
           
    def delete_member(self, workspace_name, email):
        try:
            collection = self.init_collection('members')
            result = collection.update_one({"email": email}, {"$pull": {"workspaces": workspace_name}})
            return result.modified_count > 0
        except Exception as e:
            logging.error(f"Failed to delete member: {e}")
            raise

    def delete_workspace(self, workspace_name):
        try:
            collection = self.init_collection('workspaces')
            workspace = collection.find_one({"workspace_name": workspace_name})
            if not workspace:
                return "workspace not found"
            workspace_id = workspace["_id"]

            documents_collection = self.init_collection('documents')
            documents_collection.delete_many({"workspace_id": workspace_id})

            members_collection = self.init_collection('members')
            members_collection.update_many({}, {"$pull": {"workspaces": workspace_name}})

            result = collection.delete_one({"workspace_name": workspace_name})
            return result.deleted_count > 0
        except Exception as e:
            logging.error(f"Failed to delete workspace: {e}")
            raise

    def delete_competency(self, email, competency_description):
        try:
            collection = self.init_collection('members')
            member = collection.find_one({"email": email})
            if not member:
                return "Member not found"
            competencies = member["competencies"]
            if competency_description in competencies:
                competencies.remove(competency_description)
                collection.update_one({"email": email}, {"$set": {"competencies": competencies}})
                return True
            return False
        except Exception as e:
            print("########")
            print("error delete")
            print("########")
            logging.error(f"Failed to delete competency: {e}")
            raise

    def add_competency(self, email, new_competency):
        try:
            collection = self.init_collection('members')
            member = collection.find_one({"email": email})
            if not member:
                return "Member not found"
            competencies = member["competencies"]
            if new_competency not in competencies:
                competencies.append(new_competency)
                collection.update_one({"email": email}, {"$set": {"competencies": competencies}})
                return True
            return False
        except Exception as e:
            print("########")
            print("error add")
            print("########")
            logging.error(f"Failed to add competency: {e}")
            raise

    def change_member_role(self, email, new_role):
        try:
            collection = self.init_collection('members')
            member = collection.find_one({"email": email})
            if not member:
                return "Member not found"
            collection.update_one({"email": email}, {"$set": {"role": new_role}})
            return True
        except Exception as e:
            logging.error(f"Failed to change member role: {e}")
            raise

    def change_member_name(self, email, new_name):
        try:
            collection = self.init_collection('members')
            member = collection.find_one({"email": email})
            if not member:
                return "Member not found"
            collection.update_one({"email": email}, {"$set": {"name": new_name}})
            return True
        except Exception as e:
            logging.error(f"Failed to change member name: {e}")
            raise

    def change_workspace_name(self, current_workspace_name, new_workspace_name):
        try:
            collection = self.init_collection('workspaces')
            workspace = collection.find_one({"workspace_name": current_workspace_name})
            if not workspace:
                return "workspace not found"
            collection.update_one({"workspace_name": current_workspace_name}, {"$set": {"workspace_name": new_workspace_name}})
            return True
        except Exception as e:
            logging.error(f"Failed to change workspace name: {e}")
            raise

    def delete_document(self, workspace_name, document_name):
        try:
            workspace = self.get_workspace_by_name(workspace_name)
            if not workspace:
                return "workspace not found"
            workspace_id = workspace["_id"]
            collection = self.init_collection('documents')
            result = collection.delete_one({"workspace_id": workspace_id, "document_name": document_name})
            return result.deleted_count > 0
        except Exception as e:
            logging.error(f"Failed to delete document: {e}")
            raise

    def change_document_name(self, workspace_name, current_document_name, new_document_name):
        try:
            workspace = self.get_workspace_by_name(workspace_name)
            if not workspace:
                return "workspace not found"
            workspace_id = workspace["_id"]
            collection = self.init_collection('documents')
            document = collection.find_one({"workspace_id": workspace_id, "document_name": current_document_name})
            if not document:
                print("DB STAMP 3")
                return "Document not found"
            collection.update_one({"workspace_id": workspace_id, "document_name": current_document_name}, {"$set": {"document_name": new_document_name}})
            return True
        except Exception as e:
            logging.error(f"Failed to change document name: {e}")
            raise
    def get_workspaces_by_creator_email(self, creator_email):
        try:
            collection = self.init_collection('workspaces')
            workspaces = collection.find({"creator_email": creator_email})
            return list(workspaces)
        except Exception as e:
            logging.error(f"Failed to get workspaces by creator email: {e}")
            raise
            
    def get_articles_by_workspace_name(self, workspace_name):
        try:
            workspace = self.get_workspace_by_name(workspace_name)
            if not workspace:
                return "workspace not found"
            workspace_id = workspace["_id"]

            collection = self.init_collection('documents')
            articles = collection.find({"workspace_id": workspace_id, "document_type": "article"})  # Фильтруем только статьи

            article_list = []
            for article in articles:
                article_list.append({
                    "document_name": article["document_name"],
                    "document_url": article["document_url"],
                    "creator_email": article["creator_email"]
                })

            return article_list
        except Exception as e:
            logging.error(f"Failed to get articles by workspace name: {e}")
            raise
            
            
    def invite_member_to_workspace(self, workspace_name, email):
        try:
            workspace = self.get_workspace_by_name(workspace_name)
            if not workspace:
                return "workspace not found"

            if workspace["creator_email"] == email:
                raise ValueError("You cannot invite yourself to the workspace")

            collection = self.init_collection('invitations')
            if collection.find_one({"workspace_name": workspace_name, "email": email, "status": "pending"}):
                raise ValueError("Invitation already sent")

            invitation = {
                "workspace_name": workspace_name,
                "email": email,
                "status": "pending",
                "created_at": datetime.now(timezone.utc)
            }
            return collection.insert_one(invitation).inserted_id
        except Exception as e:
            logging.error(f"Failed to invite member to workspace: {e}")
            raise

    def get_invitations_by_email(self, email):
        try:
            collection = self.init_collection('invitations')
            invitations = collection.find({"email": email, "status": "pending"})
            return list(invitations)
        except Exception as e:
            logging.error(f"Failed to get invitations by email: {e}")
            raise

    def accept_invitation(self, workspace_name, email):
        try:
            collection = self.init_collection('invitations')
            invitation = collection.find_one({"workspace_name": workspace_name, "email": email, "status": "pending"})
            if not invitation:
                return "Invitation not found"
            collection.update_one({"workspace_name": workspace_name, "email": email}, {"$set": {"status": "accepted"}})

            self.assign_member_to_workspace(email, workspace_name)

            collection.delete_one({"workspace_name": workspace_name, "email": email})

            return True
        except Exception as e:
            logging.error(f"Failed to accept invitation: {e}")
            raise

    def reject_invitation(self, workspace_name, email):
        try:
            collection = self.init_collection('invitations')
            invitation = collection.find_one({"workspace_name": workspace_name, "email": email, "status": "pending"})
            if not invitation:
                return "Invitation not found"

            collection.update_one({"workspace_name": workspace_name, "email": email}, {"$set": {"status": "rejected"}})

            collection.delete_one({"workspace_name": workspace_name, "email": email})

            return True
        except Exception as e:
            logging.error(f"Failed to reject invitation: {e}")
            raise
        
    def get_documents_by_workspace_name(self, workspace_name):
        try:
            workspace = self.get_workspace_by_name(workspace_name)
            if not workspace:
                return "workspace not found"
            workspace_id = workspace["_id"]

            collection = self.init_collection('documents')
            documents = collection.find({"workspace_id": workspace_id, "document_type": "document"})

            document_list = []
            for document in documents:
                document_list.append({
                    "document_name": document["document_name"],
                    "creator_email": document["creator_email"]
                })

            return document_list
        except Exception as e:
            logging.error(f"Failed to get documents by workspace name: {e}")
            raise
        
    def get_workspace_info_by_name(self, workspace_name):
        try:
            workspace = self.get_workspace_by_name(workspace_name)
            if not workspace:
                return "workspace not found"

            members = self.get_members_by_workspace_name(workspace_name)
            documents = self.get_documents_by_workspace_name(workspace_name)
            articles = self.get_articles_by_workspace_name(workspace_name)

            workspace_info = {
                "workspace_name": workspace["workspace_name"],
                "description": workspace["description"],
                "project_description": workspace["project_description"],
                "creator_email": workspace["creator_email"],
                "created_at": workspace["created_at"],
                "updated_at": workspace["updated_at"],
                "members": members,
                "documents": documents,
                "articles": articles
            }

            return workspace_info
        except Exception as e:
            logging.error(f"Failed to get workspace info by name: {e}")
            raise
            
    def get_member_workspaces(self, email):
        try:
            collection = self.init_collection('members')
            member = collection.find_one({"email": email})
            if not member:
                return "Member not found"

            workspaces = member.get("workspaces", [])
            if not workspaces:
                return []

            workspace_collection = self.init_collection('workspaces')
            member_workspaces = workspace_collection.find({"workspace_name": {"$in": workspaces}})
            return list(member_workspaces)
        except Exception as e:
            logging.error(f"Failed to get member workspaces: {e}")
            raise
            
    def leave_workspace(self, email, workspace_name):
        try:
            workspace = self.get_workspace_by_name(workspace_name)
            if not workspace:
                return "workspace not found"

            if workspace["creator_email"] == email:
                return "Creator cannot leave the workspace"

            collection = self.init_collection('members')
            result = collection.update_one({"email": email}, {"$pull": {"workspaces": workspace_name}})
            return result.modified_count > 0
        except Exception as e:
            logging.error(f"Failed to leave workspace: {e}")
            raise
            
    def remove_member_from_workspace(self, workspace_name, email):
        try:
            workspace = self.get_workspace_by_name(workspace_name)
            if not workspace:
                return "workspace not found"

            collection = self.init_collection('members')
            result = collection.update_one({"email": email}, {"$pull": {"workspaces": workspace_name}})
            return result.modified_count > 0
        except Exception as e:
            logging.error(f"Failed to remove member from workspace: {e}")
            raise
    
    def upload_document(self, workspace_name, document_name, document_content, key_file_path):
        try:
            workspace = self.get_workspace_by_name(workspace_name)
            if not workspace:
                return "workspace not found"

            self.load_aes_key_from_file(key_file_path)
            encrypted_content = self.encrypt_aes(document_content)

            collection = self.init_collection('documents')
            document = {
                "workspace_id": workspace["_id"],
                "document_name": document_name,
                "document_content": encrypted_content,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            return collection.insert_one(document).inserted_id
        except Exception as e:
            logging.error(f"Failed to upload document: {e}")
            raise

    def download_document(self, workspace_name, document_name, key_file_path):
        try:
            workspace = self.get_workspace_by_name(workspace_name)
            if not workspace:
                return "workspace not found"

            collection = self.init_collection('documents')
            document = collection.find_one({"workspace_id": workspace["_id"], "document_name": document_name})
            if not document:
                print("DB STAMP 4")
                return "Document not found"

            self.load_aes_key_from_file(key_file_path)
            decrypted_content = self.decrypt_aes(document["document_content"])

            return decrypted_content
        except Exception as e:
            logging.error(f"Failed to download document: {e}")
            raise
            
    def delete_document(self, workspace_name, document_name, creator_email):
        try:
            workspace = self.get_workspace_by_name(workspace_name)
            if not workspace:
                return "workspace not found"

            collection = self.init_collection('documents')
            document = collection.find_one({"workspace_id": workspace["_id"], "document_name": document_name})
            if not document:
                print("DB STAMP 5")
                return "Document not found"

            if document["creator_email"] != creator_email:
                return "You do not have permission to delete this document."

            result = collection.delete_one({"workspace_id": workspace["_id"], "document_name": document_name})
            return result.deleted_count > 0
        except Exception as e:
            logging.error(f"Failed to delete document: {e}")
            raise
            
    def is_text_content(self, content):
        try:
            print(content.decode('utf-8'))
            return True
        except UnicodeDecodeError:
            try:
                content.decode('cp1251')
                return True
            except UnicodeDecodeError:
                return True

    def get_document_content_as_text(self, workspace_name, document_name, key_file_path):
        try:
            print(f"Searching for document: {document_name} in workspace: {workspace_name}")
            workspace = self.get_workspace_by_name(workspace_name)
            if not workspace:
                print("workspace not found")
                return "workspace not found"

            collection = self.init_collection('documents')
            document = collection.find_one({"workspace_id": workspace["_id"], "document_name": document_name})
            print("=============================================")
            print(document_name)
            print("=============================================")
            if not document:
                print("Document not found")
                return "Document not found"

            print(f"Document found: {document}")
            self.load_aes_key_from_file(key_file_path)
            encrypted_content = document["document_content"]

            decrypted_content = self.decrypt_aes(encrypted_content)

            try:
                text_content = decrypted_content.decode('utf-8')
                print(text_content)
                print("=============================================")
                return text_content
                print("=============================================")
            except UnicodeDecodeError:
                try:
                    text_content = decrypted_content.decode('cp1251')
                    return text_content
                except UnicodeDecodeError:
                    return "Document content is not text"
        except Exception as e:
            logging.error(f"Failed to get document content as text: {e}")
            raise
            
    def upload_article(self, workspace_name, article_title, article_content, creator_email):
        try:
            workspace = self.get_workspace_by_name(workspace_name)
            if not workspace:
                return "workspace not found"

            self.load_aes_key_from_file(key_file_path)

            encrypted_content = self.encrypt_aes(article_content.encode('utf-8'))

            collection = self.init_collection('documents')
            document = {
                "workspace_id": workspace["_id"],
                "document_name": f"{article_title}.txt", 
                "document_content": encrypted_content,
                "document_type": "article",
                "creator_email": creator_email,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            return collection.insert_one(document).inserted_id
        except Exception as e:
            logging.error(f"Failed to upload article: {e}")
            raise
            
    def save_article_link(self, workspace_name, article_title, article_url, creator_email, document_type="article"):
        try:
            workspace = self.get_workspace_by_name(workspace_name)
            if not workspace:
                return "workspace not found"

            collection = self.init_collection('documents')
            document = {
                "workspace_id": workspace["_id"],
                "document_name": f"{article_title}.pdf",
                "document_url": article_url,
                "creator_email": creator_email,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "document_type": document_type  
            }
            return collection.insert_one(document).inserted_id
        except Exception as e:
            logging.error(f"Failed to save article link: {e}")
            raise

    def delete_article(self, workspace_name, article_name, creator_email):
        try:
            workspace = self.get_workspace_by_name(workspace_name)
            if not workspace:
                return "workspace not found"

            collection = self.init_collection('documents')
            article = collection.find_one({"workspace_id": workspace["_id"], "document_name": article_name})
            if not article:
                return "Article not found"

            if article["creator_email"] != creator_email:
                return "You do not have permission to delete this article."

            result = collection.delete_one({"workspace_id": workspace["_id"], "document_name": article_name})
            return result.deleted_count > 0
        except Exception as e:
            logging.error(f"Failed to delete article: {e}")
            raise
            
    def get_document_type(self, workspace_name, document_name):
        """
        Возвращает тип документа: "document" для пользовательских документов или "article" для статей.
        """
        try:
            workspace = self.get_workspace_by_name(workspace_name)
            if not workspace:
                return "workspace not found"

            collection = self.init_collection('documents')
            document = collection.find_one({"workspace_id": workspace["_id"], "document_name": document_name})
            if not document:
                print("DB STAMP 7")
                return "Document not found"

            return document.get("document_type", "document")
        except Exception as e:
            logging.error(f"Failed to get document type: {e}")
            raise

    def save_chat_history(self, email, chat_history, key_file_path):
        try:
            self.load_aes_key_from_file(key_file_path)
            chat_history_str = json.dumps(chat_history)
            encrypted_history = self.encrypt_aes(chat_history_str.encode('utf-8'))
            collection = self.init_collection('chat_history')
            collection.insert_one({
                "email": email,
                "history": encrypted_history,
                "created_at": datetime.now(timezone.utc)
            })
            return True
        except Exception as e:
            logging.error(f"Failed to save chat history: {e}")
            return False

    def load_chat_history(self, email, key_file_path):
        try:
            self.load_aes_key_from_file(key_file_path)
            collection = self.init_collection('chat_history')
            chat_records = collection.find({"email": email}).sort("created_at", pymongo.ASCENDING)
            
            chat_history = []
            for record in chat_records:
                if "history" in record:
                    decrypted_history = self.decrypt_aes(record["history"])
                    chat_history.extend(json.loads(decrypted_history.decode('utf-8')))
            
            return chat_history
        except Exception as e:
            logging.error(f"Failed to load chat history: {e}")
            return []
            
    def delete_chat_history(self, email):
        try:
            collection = self.init_collection('chat_history')
            result = collection.delete_many({"email": email})
            return result.deleted_count > 0
        except Exception as e:
            logging.error(f"Failed to delete chat history: {e}")
            return False