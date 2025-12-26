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
import time
from langchain.chains import LLMChain
import requests
from datetime import datetime
from utils.message_converter import MessageConverterWithDateTime
import config
from database import models
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
import time

# Load environment variables
load_dotenv()

# Twilio Client - Initialize only if credentials are available
try:
    if config.TWILIO_ACCOUNT_SID and config.TWILIO_AUTH_TOKEN:
        client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
    else:
        client = None
        print("Warning: Twilio credentials not configured. WhatsApp messaging will not work.")
except Exception as e:
    client = None
    print(f"Warning: Failed to initialize Twilio client: {e}")


def load_domain_qa_data(file_path: str = "domain-q&a.json"):
    """
    Load domain Q&A data from JSON file and format it for the system prompt.
    
    Args:
        file_path: Path to the domain Q&A JSON file
        
        
    Returns:
        tuple: (formatted string for prompt, dict of user inputs to responses with next_route, eligibility_mapping, eligibility_rules)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            qa_data = json.load(file)
        
        domains_text = "DOMAIN KNOWLEDGE:\nYou have access to the following domain-specific information to help users:\n\n"
        qa_mapping = {}  # Map user inputs to responses with next_route
        eligibility_mapping = {}  # Map user inputs to eligibility status
        eligibility_rules = {}  # Store eligibility rules per domain
        
        for domain_name, domain_info in qa_data.get("domains", {}).items():
            # Format domain name (replace underscores with spaces and capitalize)
            formatted_domain = domain_name.replace('_', ' ').title()
            domains_text += f"{formatted_domain.upper()}:\n"
            
            # Store eligibility rules for this domain
            if "eligibility_rules" in domain_info:
                eligibility_rules[domain_name] = domain_info["eligibility_rules"]
            
            questions = domain_info.get("questions", [])
            for question_data in questions:
                question = question_data.get("question", "")
                answers = question_data.get("answers", {})
                
                domains_text += f"- {question}\n"
                
                for answer_key, answer_info in answers.items():
                    response = answer_info.get("response", "")
                    next_route = answer_info.get("next_route", "/")
                    eligible = answer_info.get("eligible", False)
                    domains_text += f"  - If they say '{answer_key}': Respond with \"{response}\"\n"
                    # Store mapping for quick lookup
                    qa_mapping[answer_key.lower()] = {
                        "response": response,
                        "next_route": next_route
                    }
                    # Store eligibility mapping
                    eligibility_mapping[answer_key.lower()] = eligible
                
                domains_text += "\n"
        
        global_rules = qa_data.get("global_eligibility_rules", {})
        
        return domains_text, qa_mapping, eligibility_mapping, eligibility_rules, global_rules
        
    except FileNotFoundError:
        print(f"Warning: Domain Q&A file {file_path} not found")
        return "DOMAIN KNOWLEDGE:\nNo domain-specific information available.\n\n", {}, {}, {}, {}
    except json.JSONDecodeError as e:
        print(f"Error parsing domain Q&A file: {e}")
        return "DOMAIN KNOWLEDGE:\nError loading domain information.\n\n", {}, {}, {}, {}


# PostgreSQL (PGVector) Connection
PG_CONN_STRING = config.PG_CONN_STRING

# Load domain Q&A data
domain_knowledge, qa_mapping, eligibility_mapping, eligibility_rules, global_eligibility_rules = load_domain_qa_data()


def determine_eligibility(user_input: str, conversation_history: list = None):
    """
    Determine if a lead is eligible based on their answers (user messages only, not agent responses).
    
    Args:
        user_input: The current user input
        conversation_history: List of previous messages in the conversation
    
    Returns:
        bool: True if eligible, False otherwise
    """
    user_input_lower = user_input.lower().strip()
    
    # Check if the current user input matches any eligible answer
    if user_input_lower in eligibility_mapping:
        return eligibility_mapping[user_input_lower]
    
    # Check for partial matches in current user input
    for answer_key, is_eligible in eligibility_mapping.items():
        if answer_key in user_input_lower or user_input_lower in answer_key:
            return is_eligible
    
    # If no match found and we have conversation history, check previous USER messages only
    if conversation_history:
        eligible_answers_found = []
        for msg in conversation_history:
            # Only check HumanMessage (user messages), ignore AI/agent messages
            if hasattr(msg, '__class__') and 'Human' in msg.__class__.__name__:
                if hasattr(msg, 'content'):
                    msg_content = msg.content.lower().strip()
                    # Skip initial greeting messages
                    if msg_content == "hello":
                        continue
                    # Check if any previous user message matches eligible answers
                    for answer_key, is_eligible in eligibility_mapping.items():
                        if answer_key in msg_content or msg_content in answer_key:
                            if is_eligible:
                                eligible_answers_found.append(True)
        
        # If we found at least one eligible answer from user messages, return True
        if eligible_answers_found:
            return True
    
    # Default: not eligible if no matching criteria found
    return global_eligibility_rules.get("default_eligible", False)


def evaluate_final_eligibility(leadgen_id: str):
    """
    Evaluate and update eligibility based on the entire conversation history.
    This function should be called when the conversation ends.
    
    Args:
        leadgen_id: The Facebook leadgen_id
    
    Returns:
        dict: Contains final eligibility status and evaluation details
    """
    try:
        # Get all messages from the conversation
        session_id = str(leadgen_id)
        chat_history_table = "messages"
        message_history = SQLChatMessageHistory(
            session_id=session_id,
            connection_string=PG_CONN_STRING,
            table_name=chat_history_table,
            custom_message_converter=MessageConverterWithDateTime(chat_history_table),
        )
        
        # Get all messages
        all_messages = message_history.messages
        
        # Extract user messages (HumanMessage type), skipping the first "Hello" message
        user_messages = []
        for idx, msg in enumerate(all_messages):
            # Skip the first message if it's "Hello" (initial greeting)
            if idx == 0 and hasattr(msg, 'content'):
                msg_content = msg.content.lower().strip()
                if msg_content == "hello":
                    continue
            
            # Check if it's a HumanMessage (user message)
            if hasattr(msg, '__class__') and 'Human' in msg.__class__.__name__:
                if hasattr(msg, 'content'):
                    user_messages.append(msg.content.lower().strip())
        
        # Evaluate eligibility based on all user answers
        eligible_answers_found = []
        ineligible_answers_found = []
        
        for user_msg in user_messages:
            # Check exact matches
            if user_msg in eligibility_mapping:
                is_eligible = eligibility_mapping[user_msg]
                if is_eligible:
                    eligible_answers_found.append(user_msg)
                else:
                    ineligible_answers_found.append(user_msg)
            else:
                # Check partial matches
                for answer_key, is_eligible in eligibility_mapping.items():
                    if answer_key in user_msg or user_msg in answer_key:
                        if is_eligible:
                            eligible_answers_found.append(user_msg)
                        else:
                            ineligible_answers_found.append(user_msg)
                        break
        
        # Determine final eligibility
        # If we found at least one eligible answer, the lead is eligible
        # Otherwise, check global rules
        min_answers_required = global_eligibility_rules.get("minimum_answers_required", 1)
        
        if len(eligible_answers_found) >= min_answers_required:
            final_eligibility = True
        elif len(eligible_answers_found) > 0:
            final_eligibility = True  # At least one eligible answer found
        else:
            final_eligibility = global_eligibility_rules.get("default_eligible", False)
        
        # Update the lead's eligibility in the database
        update_lead_eligibility(leadgen_id, final_eligibility)
        
        return {
            "leadgen_id": leadgen_id,
            "eligible": final_eligibility,
            "eligible_answers_count": len(eligible_answers_found),
            "ineligible_answers_count": len(ineligible_answers_found),
            "eligible_answers": eligible_answers_found,
            "total_user_messages": len(user_messages)
        }
        
    except Exception as e:
        print(f"Error evaluating final eligibility for lead {leadgen_id}: {e}")
        return {"error": str(e)}


def update_lead_eligibility(leadgen_id: str, is_eligible: bool):
    """
    Update the eligible status of a lead in the database.
    
    Args:
        leadgen_id: The Facebook leadgen_id
        is_eligible: Boolean indicating if the lead is eligible
    """
    session = None
    try:
        engine = create_engine(PG_CONN_STRING)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Create Base and models - both must use the same Base for foreign keys to work
        Base = declarative_base()
        business_model = models.create_business_model('businesses', Base)
        lead_model = models.create_leads_model('leads', Base)
        
        lead = session.query(lead_model).filter(lead_model.leadgen_id == leadgen_id).first()
        if lead:
            lead.eligible = is_eligible
            session.commit()
            print(f"Updated eligibility for lead {leadgen_id}: {is_eligible}")
        else:
            print(f"Lead {leadgen_id} not found for eligibility update")
    except Exception as e:
        print(f"Error updating lead eligibility: {e}")
        if session:
            session.rollback()
    finally:
        if session:
            session.close()


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

def is_conversation_ending(user_input: str, conversation_history: list = None):
    """
    Check if the conversation is ending based on user input and conversation patterns.
    
    Args:
        user_input: The current user input
        conversation_history: List of previous messages
    
    Returns:
        bool: True if conversation appears to be ending
    """
    user_input_lower = user_input.lower().strip()
    
    # Keywords that indicate conversation ending
    ending_keywords = [
        "bye", "goodbye", "see you", "farewell",
        "thanks", "thank you", "thank", "thx",
        "done", "finished", "complete", "all set",
        "that's all", "that's it", "nothing else",
        "no more questions", "no further", "no more"
    ]
    
    # Check if user input contains ending keywords
    for keyword in ending_keywords:
        if keyword in user_input_lower:
            return True
    
    # Check conversation length - if we have enough messages, consider it might be ending
    if conversation_history:
        user_message_count = sum(1 for msg in conversation_history 
                                if hasattr(msg, '__class__') and 'Human' in msg.__class__.__name__)
        # If user has sent 3+ messages, they might be wrapping up
        if user_message_count >= 3:
            # Check if last few messages are short (indicating closing)
            recent_messages = [msg.content.lower().strip() for msg in conversation_history[-3:] 
                              if hasattr(msg, 'content')]
            if recent_messages and all(len(msg) < 20 for msg in recent_messages):
                return True
    
    return False


def run_agent(user_input: str, lead_id, max_number_of_messages: int=5):
    """
    Interact with the agent using LangChain and handle memory and initialization.
    Args:
        user_input (str): The query or command provided by the user.
        lead_id (int or str): Unique identifier for the lead (can be database ID or Facebook leadgen_id).
        max_number_of_messages (int): Maximum number of messages to maintain in memory.

    Returns:
        str: The agent's response based on the user input.
    """
    # Check if user is requesting to delete their data
    if is_deletion_request(user_input):
        print(f"Deletion request detected for lead {lead_id}. Deleting all data...")
        try:
            deletion_result = delete_all_lead_data(str(lead_id))
            if "error" in deletion_result:
                return {
                    "text": "I apologize, but I encountered an error while trying to delete your data. Please contact support if this issue persists.",
                    "next_route": "/",
                    "eligible": False,
                    "conversation_ending": True,
                    "data_deleted": False
                }
            else:
                return {
                    "text": "I've successfully deleted all of your data from our system. Your information, conversation history, and account have been permanently removed. Thank you for using our service, and I wish you all the best.",
                    "next_route": "/",
                    "eligible": False,
                    "conversation_ending": True,
                    "data_deleted": True
                }
        except Exception as e:
            print(f"Error processing deletion request for lead {lead_id}: {e}")
            return {
                "text": "I apologize, but I encountered an error while trying to delete your data. Please contact support if this issue persists.",
                "next_route": "/",
                "eligible": False,
                "conversation_ending": True,
                "data_deleted": False
            }
    
    # Initialize message history with lead_id (convert to string for session_id)
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
        
        # Get conversation history for eligibility determination
        conversation_history = memory.chat_memory.messages if hasattr(memory, 'chat_memory') else []
        
        # Determine eligibility based on user input and conversation history
        is_eligible = determine_eligibility(user_input, conversation_history)
        
        # Update lead eligibility in database (real-time update)
        update_lead_eligibility(str(lead_id), is_eligible)
        
        # Check if conversation is ending
        conversation_ending = is_conversation_ending(user_input, conversation_history)
        
        # If conversation is ending, evaluate final eligibility based on all messages
        if conversation_ending:
            print(f"Conversation ending detected for lead {lead_id}. Evaluating final eligibility...")
            try:
                final_evaluation = evaluate_final_eligibility(str(lead_id))
                print(f"Final eligibility evaluation completed: {final_evaluation}")
            except Exception as e:
                print(f"Error during final eligibility evaluation: {e}")
        
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
        
        # Return both response, next_route, and eligibility status
        return {
            "text": response_text,
            "next_route": next_route,
            "eligible": is_eligible,
            "conversation_ending": conversation_ending
        }
    except Exception as e:
        print(f"Error in run_agent: {e}")
        return {
            "text": "I apologize, but I'm having trouble processing your request right now. Please try again in a moment.",
            "next_route": "/",
            "eligible": False,
            "conversation_ending": False
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


def get_or_create_business(twilio_from_number: str):
    """
    Get or create a business record based on Twilio from number.
    This is a standalone helper function that can be used independently.
    
    Args:
        twilio_from_number: The Twilio phone number (can be with or without whatsapp: prefix)
    
    Returns:
        int: The business ID, or None if there was an error
    """
    session = None
    try:
        # Clean the phone number (remove whatsapp: prefix)
        clean_number = twilio_from_number.replace("whatsapp:", "").strip()
        
        engine = create_engine(PG_CONN_STRING)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        Base = declarative_base()
        business_model = models.create_business_model('businesses', Base)
        
        # Try to find existing business
        business = session.query(business_model).filter(
            business_model.twilio_from_number == clean_number
        ).first()
        
        if business:
            return business.id
        
        # Create new business if not found
        new_business = business_model(
            twilio_from_number=clean_number,
            created_at=datetime.now()
        )
        session.add(new_business)
        session.commit()
        session.refresh(new_business)
        
        print(f"Created new business with Twilio number: {clean_number} (ID: {new_business.id})")
        return new_business.id
        
    except Exception as e:
        print(f"Error getting or creating business: {e}")
        if session:
            session.rollback()
        return None
    finally:
        if session:
            session.close()


def send_initial_greeting(leadgen_id: str, lead_name: str = None):
    """
    Send an initial greeting message to a new lead to start the conversation.
    This initializes the chat session and sends a greeting from the agent.
    
    Args:
        leadgen_id (str): The Facebook leadgen_id from the webhook
        lead_name (str, optional): The name of the lead for personalization
    """
    try:
        # Send a simple greeting message as user input to trigger agent's response
        # The agent will respond naturally with a friendly greeting based on the system prompt
        initial_user_message = "Hello"
        
        # Use run_agent to initialize the conversation and get agent's greeting response
        # Convert leadgen_id to int if it's numeric, otherwise use as string
        try:
            lead_id_for_agent = int(leadgen_id)
        except (ValueError, TypeError):
            # If leadgen_id is not numeric, use it as string directly
            lead_id_for_agent = leadgen_id
        
        result = run_agent(initial_user_message, lead_id_for_agent)
        
        if result and result.get('text'):
            print(f"Initial greeting sent to lead {leadgen_id} (name: {lead_name}): {result.get('text', 'No response')[:100]}...")
        else:
            print(f"Warning: No response received for initial greeting to lead {leadgen_id}")
        
        return result
    except Exception as e:
        print(f"Error sending initial greeting to lead {leadgen_id}: {e}")
        return None


def save_lead_to_db(leadgen_id: str, lead_data=None, twilio_from_number: str = None):
    """
    Extracts lead fields and stores them in your Leads table.
    
    Args:
        leadgen_id: The Facebook leadgen_id from the webhook (or lead_id from webhook payload)
        lead_data: Optional lead data dict with fields: full_name, email, phone, form_id, created_time
                  OR Facebook API format with field_data array
        twilio_from_number: Optional Twilio from number to associate lead with a business
    
    Returns:
        tuple: (new_lead object, leadgen_id string, lead_name, lead_email, lead_phone)
    """
    session = None
    try:
        # Extract fields from lead_data if provided
        name = None
        email = None
        phone = None
        form_id = None
        created_time = None
        
        if lead_data and isinstance(lead_data, dict):
            # Handle direct webhook format: {'full_name': '...', 'email': '...', 'phone': '...', etc.}
            if "full_name" in lead_data or "email" in lead_data:
                name = lead_data.get("full_name")
                email = lead_data.get("email")
                phone = lead_data.get("phone")
                form_id = lead_data.get("form_id")
                created_time_str = lead_data.get("created_time")
                if created_time_str:
                    try:
                        created_time = datetime.fromisoformat(created_time_str.replace('Z', '+00:00'))
                    except Exception as e:
                        print(f"Error parsing created_time: {e}")
                        created_time = datetime.now()
            # Handle Facebook API format with field_data array
            elif "field_data" in lead_data:
                field_data = {item["name"]: item["values"][0] for item in lead_data["field_data"]}
                name = field_data.get("full_name")
                email = field_data.get("email")
                phone = field_data.get("phone")
        
        # Mock phone number for test leads (if phone is missing or contains test/dummy data)
        if not phone or "test" in str(phone).lower() or "dummy" in str(phone).lower() or "<test" in str(phone):
            phone = "+923007675900"  # Default test phone number
            print(f"Using mock phone number for test lead: {phone}")

        # Create engine and session
        engine = create_engine(PG_CONN_STRING)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Create Base and models - both must use the same Base for foreign keys to work
        Base = declarative_base()
        business_model = models.create_business_model('businesses', Base)
        lead_model = models.create_leads_model('leads', Base)
        
        # Get or create business if twilio_from_number is provided
        business_id = None
        if twilio_from_number:
            # Clean the phone number (remove whatsapp: prefix)
            clean_number = twilio_from_number.replace("whatsapp:", "").strip()
            
            # Try to find existing business
            business = session.query(business_model).filter(
                business_model.twilio_from_number == clean_number
            ).first()
            
            if business:
                business_id = business.id
            else:
                # Create new business if not found
                new_business = business_model(
                    twilio_from_number=clean_number,
                    created_at=datetime.now()
                )
                session.add(new_business)
                session.commit()
                session.refresh(new_business)
                business_id = new_business.id
                print(f"Created new business with Twilio number: {clean_number} (ID: {business_id})")
            
            if business_id:
                print(f"Associating lead {leadgen_id} with business ID: {business_id}")
            else:
                print(f"Warning: Could not get/create business for {twilio_from_number}, saving lead without business association")
        
        # Use provided created_time or current time
        created_at = created_time if created_time else datetime.now()
        
        new_lead = lead_model(
            leadgen_id=str(leadgen_id),  # Store Facebook leadgen_id
            business_id=business_id,  # Associate with business if available
            name=name,  
            email=email,  
            phone=phone, 
            form_id=form_id,
            eligible=None,
            created_at=created_at
        )
        session.add(new_lead)
        session.commit()
        
        # Refresh the object to ensure it's fully loaded
        session.refresh(new_lead)
        
        # Extract values before closing session to avoid detached instance errors
        lead_name = new_lead.name
        lead_email = new_lead.email
        lead_phone = new_lead.phone
        
        return new_lead, str(leadgen_id), lead_name, lead_email, lead_phone
    except Exception as e:
        print(f"Error saving lead: {e}")
        if session:
            session.rollback()
        return None, None, None, None, None
    finally:
        if session:
            session.close()

def store_lead_data(data: dict, twilio_from_number: str = None):
    """
    Store lead data in the database and automatically send initial greeting.
    
    Args:
        data: Lead data from webhook
        twilio_from_number: Optional Twilio from number to associate lead with a business
    """
    try:
            # Loop through entries and changes
            for entry in data.get("entry", []):
                for change in entry.get("changes", []):
                    if change.get("field") == "leadgen":
                        lead_id = change["value"]["leadgen_id"]
                        print(f"Fetching lead details for ID: {lead_id}")
                        lead_data = None

                        # If twilio_from_number not provided, try to get from config
                        if not twilio_from_number:
                            twilio_from_number = config.TWILIO_WHATSAPP_FROM
                        
                        new_lead, leadgen_id, lead_name, lead_email, lead_phone = save_lead_to_db(
                            lead_id, lead_data, twilio_from_number
                        )
                        if new_lead and leadgen_id:
                            print(f"Lead saved successfully: {lead_name} ({lead_email}) with leadgen_id: {leadgen_id}, phone: {lead_phone}")
                            
                            # Send WhatsApp template message with first name
                            if lead_phone:
                                print(f"Sending WhatsApp template message to {lead_phone}...")
                                template_result = sendWhatsAppTemplate(lead_phone, lead_name)
                                if template_result.get('error'):
                                    print(f"Failed to send WhatsApp template: {template_result.get('error')}")
                                else:
                                    print(f"WhatsApp template sent successfully. SID: {template_result.get('message_sid')}")
                            else:
                                print(f"Warning: No phone number found for lead {leadgen_id}. Cannot send WhatsApp message.")
                        else:
                            print("Failed to save lead to database")

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

        # Create Base and models - both must use the same Base for foreign keys to work
        Base = declarative_base()
        business_model = models.create_business_model('businesses', Base)
        lead_model = models.create_leads_model('leads', Base)

        leads = session.query(lead_model).all()

        def serialize_lead(lead):
            return {
                "leadgen_id": lead.leadgen_id,
                "business_id": lead.business_id,
                "name": lead.name,
                "email": lead.email,
                "phone": lead.phone,
                "form_id": lead.form_id,
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


def get_leads_eligibility():
    """
    Fetch all lead IDs and their eligibility status.
    Returns a simplified list with only leadgen_id and eligible status.
    """
    session = None
    try:
        engine = create_engine(PG_CONN_STRING)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Create Base and models - both must use the same Base for foreign keys to work
        Base = declarative_base()
        business_model = models.create_business_model('businesses', Base)
        lead_model = models.create_leads_model('leads', Base)

        # Query only leadgen_id and eligible fields for efficiency
        leads = session.query(lead_model.leadgen_id, lead_model.eligible).all()

        # Format results
        leads_eligibility = [
            {
                "leadgen_id": leadgen_id,
                "eligible": eligible
            }
            for leadgen_id, eligible in leads
        ]

        return {
            "leads_eligibility": leads_eligibility,
            "total_leads": len(leads_eligibility),
            "eligible_count": sum(1 for _, eligible in leads if eligible is True),
            "ineligible_count": sum(1 for _, eligible in leads if eligible is False),
            "pending_count": sum(1 for _, eligible in leads if eligible is None)
        }
    except Exception as e:
        print(f"Error fetching leads eligibility: {e}")
        return {"error": str(e)}
    finally:
        if session:
            session.close()


def is_deletion_request(user_input: str):
    """
    Check if the user is requesting to delete their data.
    Uses flexible matching to catch variations like "delete this convo", "delete info", etc.
    
    Args:
        user_input: The current user input
    
    Returns:
        bool: True if user is requesting data deletion
    """
    if not user_input:
        return False
    
    user_input_lower = user_input.lower().strip()
    
    # Primary deletion action words
    deletion_actions = ["delete", "remove", "erase", "clear", "wipe", "forget"]
    
    # Context words that indicate data/information
    data_contexts = [
        "data", "information", "info", "account", "convo", "conversation", 
        "chat", "messages", "history", "record", "details", "everything",
        "all", "my data", "my info", "my information", "this", "that"
    ]
    
    # Check for exact phrase matches first (more specific)
    deletion_keywords = [
        "delete my data", "delete my information", "delete my account",
        "delete data", "delete information", "delete account",
        "remove my data", "remove my information", "remove my account",
        "remove data", "remove information", "remove account",
        "erase my data", "erase my information", "erase my account",
        "erase data", "erase information", "erase account",
        "forget my data", "forget my information", "forget me",
        "gdpr delete", "delete everything", "delete all my data",
        "i want to delete", "i want my data deleted", "delete everything about me",
        "delete all", "remove all", "erase all", "delete everything",
        "clear my data", "clear data", "wipe my data", "wipe data",
        "delete this", "delete that", "remove this", "remove that",
        "delete convo", "delete conversation", "delete chat",
        "remove convo", "remove conversation", "remove chat"
    ]
    
    # Check for exact phrase matches
    for keyword in deletion_keywords:
        if keyword in user_input_lower:
            print(f"   🗑️ Deletion keyword matched: '{keyword}' in '{user_input_lower}'")
            return True
    
    # Flexible matching: check if message contains deletion action + data context
    has_deletion_action = any(action in user_input_lower for action in deletion_actions)
    has_data_context = any(context in user_input_lower for context in data_contexts)
    
    if has_deletion_action and has_data_context:
        print(f"   🗑️ Deletion detected (flexible match): action + context in '{user_input_lower}'")
        return True
    
    # Also check for just "delete" with common phrases
    if "delete" in user_input_lower:
        delete_phrases = ["delete this", "delete that", "delete it", "delete convo", 
                         "delete info", "delete my", "delete the", "delete any"]
        if any(phrase in user_input_lower for phrase in delete_phrases):
            print(f"   🗑️ Deletion detected (delete phrase): '{user_input_lower}'")
            return True
    
    return False


def delete_all_lead_data(leadgen_id: str):
    """
    Delete all data associated with a lead from the database.
    This includes:
    - All messages in the conversation history
    - The lead record itself
    
    Args:
        leadgen_id: The Facebook leadgen_id
    
    Returns:
        dict: Result of the deletion operation
    """
    session = None
    session_id = str(leadgen_id)
    chat_history_table = "messages"
    
    try:
        print(f"   🔍 Starting deletion for leadgen_id: {leadgen_id}")
        engine = create_engine(PG_CONN_STRING)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Create Base and models - all must use the same Base for foreign keys to work
        Base = declarative_base()
        business_model = models.create_business_model('businesses', Base)
        message_model = models.create_message_model(chat_history_table, Base)
        lead_model = models.create_leads_model('leads', Base)
        
        deleted_messages_count = 0
        deleted_lead = False
        
        # Delete all messages for this lead (by session_id and lead_id to be thorough)
        print(f"   🔍 Searching for messages with session_id={session_id} or lead_id={leadgen_id}")
        messages_to_delete = session.query(message_model).filter(
            (message_model.session_id == session_id) | 
            (message_model.lead_id == str(leadgen_id))
        ).all()
        
        print(f"   🔍 Found {len(messages_to_delete)} messages to delete")
        if messages_to_delete:
            for msg in messages_to_delete:
                session.delete(msg)
                deleted_messages_count += 1
            print(f"   ✅ Deleted {deleted_messages_count} messages")
        
        # Delete the lead record itself
        print(f"   🔍 Searching for lead with leadgen_id={leadgen_id}")
        lead_to_delete = session.query(lead_model).filter(
            lead_model.leadgen_id == str(leadgen_id)
        ).first()
        
        if lead_to_delete:
            print(f"   ✅ Found lead record to delete: {lead_to_delete.name} ({lead_to_delete.email})")
            session.delete(lead_to_delete)
            deleted_lead = True
        else:
            print(f"   ⚠️ No lead record found with leadgen_id={leadgen_id}")
        
        print(f"   💾 Committing deletion transaction...")
        session.commit()
        print(f"   ✅ Transaction committed successfully")
        
        result = {
            "message": f"All data for lead {leadgen_id} has been deleted successfully.",
            "deleted_messages": deleted_messages_count,
            "deleted_lead": deleted_lead
        }
        print(f"   ✅ Deletion result: {result}")
        return result
        
    except Exception as e:
        print(f"   ❌ Error deleting all lead data for {leadgen_id}: {e}")
        import traceback
        traceback.print_exc()
        if session:
            session.rollback()
            print(f"   🔄 Transaction rolled back")
        return {"error": str(e)}
    finally:
        if session:
            session.close()
            print(f"   🔒 Database session closed")


def delete_user_data(user_id: str):
    """
    Delete user data associated with the given user_id from the database.
    This is the API endpoint version that calls delete_all_lead_data.
    """
    return delete_all_lead_data(user_id)

def get_conversation(lead_id):
    """
    Retrieve conversation history for a specific lead from the messages table.
    
    Args:
        lead_id (int or str): The ID of the lead to retrieve conversation for (can be database ID or Facebook leadgen_id).
    
    Returns:
        dict: A dictionary containing the conversation messages with timestamps or an error message.
    """
    try:
        # Query the database directly to get messages with timestamps
        # Convert lead_id to string for session_id (works with both int and str)
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


def sendWhatsAppTemplate(to_number, first_name, template_sid=None):
    """
    Send a WhatsApp template message with first name placeholder.
    
    Args:
        to_number: Recipient WhatsApp number (e.g., +923007675900)
        first_name: First name to use in template placeholder {{1}}
        template_sid: Template SID (uses config if not provided)
    
    Returns:
        dict: Message status and details
    """
    if not client:
        print("Error: Twilio client not initialized. Check your credentials.")
        return {"error": "Twilio client not initialized"}
    
    try:
        # Get configuration
        use_sandbox = config.TWILIO_USE_SANDBOX
        
        # Get from number
        if config.TWILIO_WHATSAPP_FROM:
            from_ = config.TWILIO_WHATSAPP_FROM
        else:
            return {"error": "WhatsApp sender not configured"}

        # Get template SID
        template_sid = template_sid or config.TEMPLATE_SID
        if not template_sid:
            return {"error": "Template SID not configured"}

        # Ensure WhatsApp format
        if not to_number.startswith("whatsapp:"):
            to = f"whatsapp:{to_number}"
        else:
            to = to_number
            
        if not from_.startswith("whatsapp:"):
            from_ = f"whatsapp:{from_}"
        
        # Extract first name from full name if needed
        if first_name:
            # Split by space and take first part
            first_name_clean = first_name.split()[0] if first_name else "there"
        else:
            first_name_clean = "there"
        
        # Create content variables JSON for template placeholder {{1}}
        content_variables = json.dumps({"1": first_name_clean})
        
        mode = "SANDBOX" if use_sandbox else "PRODUCTION"
        print(f"[{mode}] Sending WhatsApp template to {to} with name: {first_name_clean}")
        print(f"  Template SID: {template_sid}")

        message = client.messages.create(
            from_=from_,
            to=to,
            content_sid=template_sid,
            content_variables=content_variables
        )
        
        print(f"✓ Template message created successfully. SID: {message.sid}")
        print(f"  Status: {message.status}")
        
        return {
            "status": "sent",
            "message_sid": message.sid,
            "message_status": message.status,
            "mode": mode.lower(),
            "to": to,
            "from": from_
        }
    except Exception as e:
        print(f"✗ Error sending WhatsApp template: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def sendWhatsAppMessage(to_number, message_body, from_number=None):
    """
    Send a regular WhatsApp text message (not template).
    
    Args:
        to_number: Recipient WhatsApp number (e.g., +923007675900)
        message_body: Message text to send
        from_number: Sender WhatsApp number (uses config if not provided)
    
    Returns:
        dict: Message status and details
    """
    if not client:
        print("Error: Twilio client not initialized. Check your credentials.")
        return {"error": "Twilio client not initialized"}
    
    try:
        # Get from number
        if from_number:
            from_ = from_number
        elif config.TWILIO_WHATSAPP_FROM:
            from_ = config.TWILIO_WHATSAPP_FROM
        else:
            return {"error": "WhatsApp sender not configured"}

        # Ensure WhatsApp format
        if not to_number.startswith("whatsapp:"):
            to = f"whatsapp:{to_number}"
        else:
            to = to_number
            
        if not from_.startswith("whatsapp:"):
            from_ = f"whatsapp:{from_}"
        
        print(f"Sending WhatsApp message to {to}")

        message = client.messages.create(
            from_=from_,
            to=to,
            body=message_body
        )
        
        print(f"✓ Message sent successfully. SID: {message.sid}")
        
        return {
            "status": "sent",
            "message_sid": message.sid,
            "message_status": message.status,
            "to": to,
            "from": from_
        }
    except Exception as e:
        print(f"✗ Error sending WhatsApp message: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def get_lead_by_phone(phone_number):
    """
    Get lead information by phone number.
    
    Args:
        phone_number: Phone number to search for (can be with or without whatsapp: prefix)
    
    Returns:
        tuple: (leadgen_id, lead_name) or (None, None) if not found
    """
    session = None
    try:
        # Clean phone number (remove whatsapp: prefix and any formatting)
        clean_phone = phone_number.replace("whatsapp:", "").strip()
        
        engine = create_engine(PG_CONN_STRING)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Create Base and models - both must use the same Base for foreign keys to work
        Base = declarative_base()
        business_model = models.create_business_model('businesses', Base)
        lead_model = models.create_leads_model('leads', Base)
        
        # Search for lead by phone number
        lead = session.query(lead_model).filter(lead_model.phone == clean_phone).first()
        
        if lead:
            return lead.leadgen_id, lead.name
        else:
            return None, None
    except Exception as e:
        print(f"Error finding lead by phone: {e}")
        return None, None
    finally:
        if session:
            session.close()

def handle_deletion_request(from_number, leadgen_id, incoming_message):
    """
    Handle deletion request from user.
    
    Args:
        from_number: User's WhatsApp number
        leadgen_id: Lead identifier
        incoming_message: User's deletion request message
    
    Returns:
        str: TwiML XML response
    """
    response = MessagingResponse()
    print(f"🗑️ DELETION REQUEST detected from {from_number} for lead {leadgen_id}")
    print(f"   Message: '{incoming_message}'")
    
    try:
        deletion_result = delete_all_lead_data(leadgen_id)
        
        if "error" in deletion_result:
            deletion_msg = "I apologize, but I encountered an error while trying to delete your data. Please contact support if this issue persists."
            print(f"   ❌ Error deleting data: {deletion_result.get('error')}")
        else:
            deleted_messages = deletion_result.get('deleted_messages', 0)
            deleted_lead = deletion_result.get('deleted_lead', False)
            deletion_msg = "I've successfully deleted all of your data from our system. Your information, conversation history, and account have been permanently removed. Thank you for using our service, and I wish you all the best."
            print(f"   ✅ Data deleted successfully: {deleted_messages} messages deleted, lead record deleted: {deleted_lead}")
        
        send_result = sendWhatsAppMessage(from_number, deletion_msg)
        if send_result.get('error'):
            response.message(deletion_msg)
        else:
            response.message("")
        
        return str(response)
    except Exception as e:
        print(f"   ❌ Exception during deletion: {e}")
        import traceback
        traceback.print_exc()
        error_msg = "I apologize, but I encountered an error while trying to delete your data. Please contact support if this issue persists."
        sendWhatsAppMessage(from_number, error_msg)
        response.message("")
        return str(response)


def handle_agent_conversation(from_number, leadgen_id, incoming_message):
    """
    Handle conversation with agent.
    
    Args:
        from_number: User's WhatsApp number
        leadgen_id: Lead identifier
        incoming_message: User's message
    
    Returns:
        str: TwiML XML response
    """
    response = MessagingResponse()
    
    try:
        agent_result = run_agent(incoming_message, leadgen_id)
        
        if agent_result and isinstance(agent_result, dict):
            agent_response = agent_result.get('text', 'I apologize, but I had trouble processing that. Could you please rephrase?')
            
            if agent_result.get('data_deleted'):
                print(f"Data deletion completed via agent for lead {leadgen_id}")
            
            send_result = sendWhatsAppMessage(from_number, agent_response)
            
            if send_result.get('error'):
                response.message(agent_response)
            else:
                response.message("")
            
            if agent_result.get('conversation_ending') and not agent_result.get('data_deleted'):
                print(f"Conversation ending for lead {leadgen_id}. Evaluating eligibility...")
                try:
                    evaluate_final_eligibility(leadgen_id)
                except Exception as e:
                    print(f"Error evaluating eligibility: {e}")
        else:
            error_msg = "I apologize, but I'm having trouble processing your request right now. Please try again in a moment."
            sendWhatsAppMessage(from_number, error_msg)
            response.message("")
        
        return str(response)
    except Exception as e:
        print(f"Error processing message with agent: {e}")
        import traceback
        traceback.print_exc()
        error_msg = "Sorry, there was an error processing your message. Please try again."
        sendWhatsAppMessage(from_number, error_msg)
        response.message("")
        return str(response)


def receive_message(data=None):
    """
    Receive and route incoming WhatsApp messages.
    
    Args:
        data: Incoming message data from Twilio webhook with 'Body', 'From', and 'To' fields
    
    Returns:
        str: TwiML XML response
    """
    try:
        response = MessagingResponse()
        
        if not data:
            response.message("Thank you for your message. We will get back to you soon.")
            return str(response)
        
        incoming_message = data.get('Body', '').strip()
        from_number = data.get('From', '')  # User's phone number
        to_number = data.get('To', '')  # Business Twilio number (receiving number)
        
        if not incoming_message:
            response.message("Please send a message.")
            return str(response)
        
        print(f"Received WhatsApp message from {from_number} to {to_number}: {incoming_message}")
        
        # Get or create business based on the receiving Twilio number
        business_id = None
        if to_number:
            business_id = get_or_create_business(to_number)
            if business_id:
                print(f"Message received by business ID: {business_id} (Twilio number: {to_number})")
        
        # Get lead_id from phone number
        leadgen_id, lead_name = get_lead_by_phone(from_number)
        
        if not leadgen_id:
            print(f"⚠️ Warning: No lead found for phone number {from_number}")
            if is_deletion_request(incoming_message):
                response.message("Your data has already been deleted. There is no information stored about you in our system.")
            else:
                response.message("Sorry, we couldn't find your information. Please contact support.")
            return str(response)
        
        print(f"Found lead: {leadgen_id} ({lead_name})")
        
        # Route to appropriate handler
        if is_deletion_request(incoming_message):
            return handle_deletion_request(from_number, leadgen_id, incoming_message)
        else:
            return handle_agent_conversation(from_number, leadgen_id, incoming_message)
        
    except Exception as e:
        print(f"Error processing incoming message: {e}")
        import traceback
        traceback.print_exc()
        response = MessagingResponse()
        response.message("Sorry, there was an error processing your message.")
        return str(response)