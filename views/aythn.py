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