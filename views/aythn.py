from types import NoneType
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import json
from langchain.chains import LLMChain
import requests
from datetime import datetime
from utils.message_converter import MessageConverterWithDateTime
import config
from database import models
# Load environment variables
load_dotenv()

# Global variables for reuse (simplified)

def load_domain_qa_data(file_path: str = "domain-q&a.json"):
    """
    Load domain Q&A data from JSON file and format it for the system prompt.
    
    Args:
        file_path: Path to the domain Q&A JSON file
        
        
    Returns:
        tuple: (formatted string for prompt, dict of user inputs to responses with next_route)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            qa_data = json.load(file)
        
        domains_text = "DOMAIN KNOWLEDGE:\nYou have access to the following domain-specific information to help users:\n\n"
        qa_mapping = {}  # Map user inputs to responses with next_route
        
        for domain_name, domain_info in qa_data.get("domains", {}).items():
            # Format domain name (replace underscores with spaces and capitalize)
            formatted_domain = domain_name.replace('_', ' ').title()
            domains_text += f"{formatted_domain.upper()}:\n"
            
            questions = domain_info.get("questions", [])
            for question_data in questions:
                question = question_data.get("question", "")
                answers = question_data.get("answers", {})
                
                domains_text += f"- {question}\n"
                
                for answer_key, answer_info in answers.items():
                    response = answer_info.get("response", "")
                    next_route = answer_info.get("next_route", "/")
                    domains_text += f"  - If they say '{answer_key}': Respond with \"{response}\"\n"
                    # Store mapping for quick lookup
                    qa_mapping[answer_key.lower()] = {
                        "response": response,
                        "next_route": next_route
                    }
                
                domains_text += "\n"
        
        return domains_text, qa_mapping
        
    except FileNotFoundError:
        print(f"Warning: Domain Q&A file {file_path} not found")
        return "DOMAIN KNOWLEDGE:\nNo domain-specific information available.\n\n", {}
    except json.JSONDecodeError as e:
        print(f"Error parsing domain Q&A file: {e}")
        return "DOMAIN KNOWLEDGE:\nError loading domain information.\n\n", {}


# PostgreSQL (PGVector) Connection
PG_CONN_STRING = config.PG_CONN_STRING

# Load domain Q&A data
domain_knowledge, qa_mapping = load_domain_qa_data()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
You are Aythn, a friendly and helpful conversational AI assistant. Your goal is to have natural, human-like conversations with users while helping them with their questions and needs.

Key guidelines for your interactions:
1. Be warm, empathetic, and conversational - like talking to a knowledgeable friend
2. Use natural language with appropriate casual expressions and contractions
3. Show genuine interest in the user's responses
4. Ask follow-up questions to better understand their needs
5. Keep responses concise but engaging - avoid overly long explanations unless needed
6. If you don't have specific information, be honest about it and offer to help in other ways

{domain_knowledge}

Remember: You're here to help and have a pleasant conversation, not just to provide robotic responses. Make the user feel heard and valued. Use the domain knowledge naturally in conversation flow.
            """
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ]
)

def run_agent(user_input: str, lead_id: int, max_number_of_messages: int=5):
    """
    Interact with the agent using LangChain and handle memory and initialization.
    Args:
        user_input (str): The query or command provided by the user.
        lead_id (int): Unique identifier for the lead.
        max_number_of_messages (int): Maximum number of messages to maintain in memory.

    Returns:
        str: The agent's response based on the user input.
    """
    # Initialize message history with lead_id
    session_id = str(lead_id)
    chat_history_table = "messages"
    message_history = SQLChatMessageHistory(
        session_id=session_id,
        connection_string=PG_CONN_STRING,
        table_name=chat_history_table,
        custom_message_converter=MessageConverterWithDateTime(chat_history_table),
    )

    # Initializing buffer memory from the message history
    memory = ConversationBufferWindowMemory(
        chat_memory=message_history,
        memory_key="chat_history",
        return_messages=True,
        k=int(max_number_of_messages),
    )

    # Initialize the llm model for direct conversation
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # Create a simple chain with memory
    chain = LLMChain(llm=chat_model, prompt=prompt, memory=memory)

    # Interact with the agent
    try:
        response = chain.invoke(input=user_input)
        response_text = response['text']
        
        # Determine next_route based on Q&A data - match user input to find appropriate route
        next_route = "/"  # default
        user_input_lower = user_input.lower().strip()
        
        # Check if user input matches any key in qa_mapping
        if user_input_lower in qa_mapping:
            next_route = qa_mapping[user_input_lower]["next_route"]
        else:
            # Try partial matching - check if any key in qa_mapping contains the user input
            for key, value in qa_mapping.items():
                if user_input_lower in key or key in user_input_lower:
                    next_route = value["next_route"]
                    break
        
        # Return both response and next_route
        return {
            "text": response_text,
            "next_route": next_route
        }
    except Exception as e:
        print(f"Error in run_agent: {e}")
        return {
            "text": "I apologize, but I'm having trouble processing your request right now. Please try again in a moment.",
            "next_route": "/"
        }



def check_subscription_status():
    """
    Check if the webhook is subscribed to leadgen service.
    If no subscription exists, create one automatically.
    
    Returns:
        dict: Response payload from Facebook API containing subscription details
    """
    try:
        app_id = config.META_APP_ID
        access_token = config.META_ACCESS_TOKEN

        # First, check if subscription exists
        url = f"https://graph.facebook.com/v24.0/{app_id}/subscriptions"
        params = {
            "access_token": access_token
        }

        resp = requests.get(url, params=params)
        try:
            payload = resp.json()
        except Exception:
            payload = {"text": resp.text}

        # Subscription exists, return parsed subscription info and raw payload for visibility
        return {
            "subscription_status": "active",
            "subscription_data": payload
        }
        
    except Exception as e:
        return {"error": str(e)}



def subscribe_webhook():
    """
    Subscribe to Facebook Page Webhook for leadgen events.
    
    Returns:
        dict: Response payload from Facebook API
    """
    try:
        app_id = config.META_APP_ID
        callback_url = config.WEBHOOK_CALLBACK_URL
        verify_token = config.WEBHOOK_VERIFY_TOKEN
        fields = "leadgen"
        access_token = config.META_ACCESS_TOKEN

        # Validate required configuration
        missing = []
        if not app_id:
            missing.append("META_APP_ID")
        if not access_token:
            missing.append("META_ACCESS_TOKEN")
        if not callback_url:
            missing.append("WEBHOOK_CALLBACK_URL")
        if not verify_token:
            missing.append("WEBHOOK_VERIFY_TOKEN")
        if missing:
            return {
                "error": "Missing required configuration",
                "missing": missing
            }

        url = f"https://graph.facebook.com/v24.0/{app_id}/subscriptions"
        data = {
            "object": "page",
            "callback_url": callback_url,
            "verify_token": verify_token,
            "fields": fields,
            "access_token": access_token,
        }

        try:
            resp = requests.post(url, data=data)
            payload = resp.json()
        except Exception:
            return{
                "error": resp.text
            }

        response= check_subscription_status()
        if response.get("error"):
            return {
                "error": response.get("error")
            }
        else:
            return {
                "subscription_status": "active",
                "message": "Subscription active",
                "subscription_data": payload,
            }
    except Exception as e:
        return {"error": str(e)}


def fetch_lead_details(lead_id: int):
    """
    Fetch the full lead info (name, email, phone, etc.) from Graph API
    """
    url = f"https://graph.facebook.com/v24.0/{lead_id}"
    params = {
        "access_token": config.PAGE_ACCESS_TOKEN  # Page access token with leads_retrieval permission
    }
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        return resp.json()
    else:
        error_data = resp.json() if resp.text else {}
        error_msg = error_data.get("error", {}).get("message", resp.text)
        error_code = error_data.get("error", {}).get("code", resp.status_code)
        print(f"Failed to fetch lead ID {lead_id}: [{error_code}] {error_msg}")
        return None


def send_initial_greeting(lead_id: int, lead_name: str = None):
    """
    Send an initial greeting message to a new lead to start the conversation.
    This initializes the chat session and sends a greeting from the agent.
    
    Args:
        lead_id (int): The ID of the lead in the database
        lead_name (str, optional): The name of the lead for personalization
    """
    try:
        # Send a simple greeting message as user input to trigger agent's response
        # The agent will respond naturally with a friendly greeting based on the system prompt
        initial_user_message = "Hello"
        
        # Use run_agent to initialize the conversation and get agent's greeting response
        result = run_agent(initial_user_message, lead_id)
        
        if result and result.get('text'):
            print(f"Initial greeting sent to lead {lead_id} (name: {lead_name}): {result.get('text', 'No response')[:100]}...")
        else:
            print(f"Warning: No response received for initial greeting to lead {lead_id}")
        
        return result
    except Exception as e:
        print(f"Error sending initial greeting to lead {lead_id}: {e}")
        return None


def save_lead_to_db(lead_data):
    """
    Extracts lead fields and stores them in your Leads table
    """
    session = None
    try:
        field_data = {item["name"]: item["values"][0] for item in lead_data["field_data"]}
        name = field_data.get("full_name")
        email = field_data.get("email")

        # Create engine and session
        engine = create_engine(PG_CONN_STRING)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Create Lead model dynamically
        Base = declarative_base()
        lead_model = models.create_leads_model('leads', Base)
        
        new_lead = lead_model(
            name=name,
            email=email,
            eligible=None,
            created_at=datetime.now()
        )
        session.add(new_lead)
        session.commit()
        
        # Get the lead ID after commit (it's auto-generated)
        lead_db_id = new_lead.id
        
        return new_lead, lead_db_id
    except Exception as e:
        print(f"Error saving lead: {e}")
        if session:
            session.rollback()
        return None, None
    finally:
        if session:
            session.close()

def store_lead_data(data: dict):
    """
    Store lead data in the database and automatically send initial greeting.
    """
    try:
            # Loop through entries and changes
            for entry in data.get("entry", []):
                for change in entry.get("changes", []):
                    if change.get("field") == "leadgen":
                        lead_id = change["value"]["leadgen_id"]
                        print(f"Fetching lead details for ID: {lead_id}")

                        # lead_data = fetch_lead_details(lead_id)
                        # mock data for leads
                        lead_data={
                            "id": lead_id,
                            "field_data": [
                                {"name": "full_name", "values": ["John Doe"]},
                                {"name": "email", "values": ["john@gmail.com"]}
                            ]
                        }

                        if lead_data:
                            new_lead, lead_db_id = save_lead_to_db(lead_data)
                            if new_lead and lead_db_id:
                                print(f"Lead saved successfully: {new_lead.name} ({new_lead.email})")
                                
                                # Automatically send initial greeting to start conversation
                                print(f"Sending initial greeting to lead {lead_db_id}...")
                                greeting_result = send_initial_greeting(lead_db_id, new_lead.name)
                                if greeting_result:
                                    print(f"Initial greeting sent successfully to lead {lead_db_id}")
                                else:
                                    print(f"Failed to send initial greeting to lead {lead_db_id}")
                            else:
                                print("Failed to save lead to database")
                        else:
                            print(f"Skipping lead ID {lead_id} - could not fetch details (may be invalid or lack permissions)")

            return {"message": "Lead data stored successfully"}
    except Exception as e:
        print(f"Error in store_lead_data: {e}")
        return {"error": str(e)}


def get_all_leads():
    """
    Fetch all leads from the database.
    """
    session = None
    try:
        engine = create_engine(PG_CONN_STRING)
        Session = sessionmaker(bind=engine)
        session = Session()

        Base = declarative_base()
        lead_model = models.create_leads_model('leads', Base)

        leads = session.query(lead_model).all()

        def serialize_lead(lead):
            return {
                "id": lead.id,
                "name": lead.name,
                "email": lead.email,
                "eligible": lead.eligible,
                "created_at": lead.created_at.isoformat() if lead.created_at else None,
            }

        return {"leads": [serialize_lead(l) for l in leads]}
    except Exception as e:
        print(f"Error fetching leads: {e}")
        return {"error": str(e)}
    finally:
        if session:
            session.close()


def delete_user_data(user_id: str):
    """
    Delete user data associated with the given user_id from the database.
    """
    session = None
    session_id = str(user_id)
    chat_history_table = "messages"

    engine = create_engine(PG_CONN_STRING)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        Base = declarative_base()
        message_model = models.create_message_model(chat_history_table, Base)

        lead_to_delete = session.query(message_model).filter(
            message_model.session_id == session_id
        ).first()
        if lead_to_delete:
            session.delete(lead_to_delete)
            session.commit()
            return {"message": f"User data for {user_id} deleted successfully."}
        else:
            return {"message": f"No user data found for {user_id}."}
    except Exception as e:
        print(f"Error deleting user data: {e}")
        if session:
            session.rollback()
        return {"error": str(e)}
    finally:
        if session:
            session.close()

def get_conversation(lead_id: int):
    """
    Retrieve conversation history for a specific lead from the messages table.
    
    Args:
        lead_id (int): The ID of the lead to retrieve conversation for.
    
    Returns:
        dict: A dictionary containing the conversation messages with timestamps or an error message.
    """
    try:
        # Query the database directly to get messages with timestamps
        session_id = str(lead_id)
        chat_history_table = "messages"
        
        engine = create_engine(PG_CONN_STRING)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            Base = declarative_base()
            message_model = models.create_message_model(chat_history_table, Base)
            
            # Query messages with timestamps, ordered by created_at
            db_messages = session.query(message_model).filter(
                message_model.session_id == session_id
            ).order_by(message_model.created_at.asc()).all()
            
            # Convert database messages to conversation format
            conversation_messages = []
            message_converter = MessageConverterWithDateTime(chat_history_table)
            
            for idx, db_msg in enumerate(db_messages):
                try:
                    # Convert JSON message back to LangChain message object
                    langchain_message = message_converter.from_sql_model(db_msg)
                    
                    # Skip the first message if it's a user message with "Hello"
                    if idx == 0:
                        continue  # Skip this message
                    
                    # Extract message information
                    message_dict = {
                        "type": langchain_message.__class__.__name__,
                        "content": langchain_message.content,
                        "created_at": db_msg.created_at.isoformat() if db_msg.created_at else None,
                    }
                    
                    # Add additional metadata if available
                    if hasattr(langchain_message, 'additional_kwargs'):
                        message_dict["additional_kwargs"] = langchain_message.additional_kwargs
                    
                    conversation_messages.append(message_dict)
                except Exception as e:
                    print(f"Error parsing message {db_msg.id}: {e}")
                    continue
            
        finally:
            session.close()
        
        return {
            "lead_id": lead_id,
            "messages": conversation_messages,
            "message_count": len(conversation_messages)
        }
        
    except Exception as e:
        print(f"Error retrieving conversation for lead {lead_id}: {e}")
        return {"error": str(e)}

