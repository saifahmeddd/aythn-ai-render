from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, text
from sqlalchemy import Column, Integer, Text, DateTime, Boolean, ForeignKey, String
import config


def create_leads_model(table_name, dynamic_base):
    """
    Create a leads model for a given table name.

    Args:
        table_name: The name of the table to use.
        dynamic_base: The base class to use for the model.

    Returns:
        The model class.
    """
    class Lead(dynamic_base):
        __tablename__ = table_name
        leadgen_id = Column(String, primary_key=True)  # Facebook leadgen_id from webhook (primary key)
        name = Column(Text, nullable=True)
        email = Column(Text, nullable=True)
        phone = Column(Text, nullable=True)
        form_id = Column(String, nullable=True)
        eligible = Column(Boolean)
        created_at = Column(DateTime)

    return Lead


def create_message_model(table_name, dynamic_base):
    """
    Create a message model for a given table name.

    Args:
        table_name: The name of the table to use.
        dynamic_base: The base class to use for the model.

    Returns:
        The model class.

    """

    # Model declared inside a function to have a dynamic table name
    class Message(dynamic_base):
        __tablename__ = table_name
        id = Column(Integer, primary_key=True)
        lead_id = Column(String, ForeignKey('leads.leadgen_id'))  # Foreign key to leads.leadgen_id
        # Required by LangChain's SQLChatMessageHistory
        session_id = Column(String, index=True)
        message = Column(Text)
        created_at = Column(DateTime)

    return Message


def initialize_database():
    """
    Initialize database tables if they don't exist.
    Creates both 'leads' and 'messages' tables with proper foreign key constraints.
    """
    try:
        engine = create_engine(config.PG_CONN_STRING)
        Base = declarative_base()

        # Create models with the same Base instance to ensure FK relationships work
        create_leads_model('leads', Base)
        create_message_model('messages', Base)

        # Create all tables with foreign key constraints
        Base.metadata.create_all(engine)

        print("Database tables initialized successfully")

    except Exception as e:
        print(f"Error initializing database: {e}")
        import traceback
        traceback.print_exc()
        return False
