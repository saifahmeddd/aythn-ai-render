from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, text
from sqlalchemy import Column, Text, DateTime, Boolean, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
import uuid
import config


# Create a single Base instance for all models
Base = declarative_base()


def create_business_model(table_name, dynamic_base):
    """
    Create a business model for a given table name.
    Maps to existing 'businesses' table created by NestJS migrations.

    Args:
        table_name: The name of the table to use.
        dynamic_base: The base class to use for the model.

    Returns:
        The model class.
    """
    class Business(dynamic_base):
        __tablename__ = table_name
        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        name = Column(String, nullable=False)
        subdomain = Column(String, nullable=False)
        # Map Python attribute to database column name
        twilio_number = Column(String, nullable=False, name='twilio_number')
        created_at = Column(DateTime, nullable=False)

    return Business


def create_leads_model(table_name, dynamic_base):
    """
    Create a leads model for a given table name.
    Maps to existing 'leads' table created by NestJS migrations.

    Args:
        table_name: The name of the table to use.
        dynamic_base: The base class to use for the model.

    Returns:
        The model class.
    """
    class Lead(dynamic_base):
        __tablename__ = table_name
        # Map Python attribute 'leadgen_id' to database column 'leadgenId' (camelCase)
        leadgen_id = Column(String, primary_key=True, name='leadgenId')
        business_id = Column(UUID(as_uuid=True), ForeignKey('businesses.id'), nullable=True)
        name = Column(String, nullable=True)
        email = Column(String, nullable=True)
        phone = Column(String, nullable=True)
        status = Column(String, nullable=False, default='new')
        # Map Python attribute 'form_id' to database column 'formId' (camelCase)
        form_id = Column(String, nullable=True, name='formId')
        eligible = Column(Boolean, nullable=False, default=False)
        created_at = Column(DateTime, nullable=False)

    return Lead


def create_message_model(table_name, dynamic_base):
    """
    Create a message model for a given table name.
    Maps to existing 'messages' table created by NestJS migrations.

    Args:
        table_name: The name of the table to use.
        dynamic_base: The base class to use for the model.

    Returns:
        The model class.
    """
    class Message(dynamic_base):
        __tablename__ = table_name
        id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
        # Map Python attribute 'lead_id' to database column 'leadgen_id'
        lead_id = Column(String, ForeignKey('leads.leadgenId'), nullable=True, name='leadgen_id')
        session_id = Column(String, nullable=False, index=True)
        message = Column(Text, nullable=False)
        created_at = Column(DateTime, nullable=False)

    return Message


def get_database_engine():
    """
    Create and return a database engine connection.
    Tables are already created by NestJS migrations, so we just need to connect.
    
    Returns:
        SQLAlchemy engine instance
    """
    return create_engine(config.PG_CONN_STRING)


def initialize_database():
    """
    Test database connection. Tables are already created by NestJS migrations.
    This function just verifies the connection is working.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        engine = get_database_engine()
        # Test connection by executing a simple query
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("Database connection successful")
        return True
    except Exception as e:
        print(f"Error connecting to database: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_models():
    """
    Get all model classes mapped to existing database tables.
    
    Returns:
        Tuple of (Business, Lead, Message) model classes
    """
    business_model = create_business_model('businesses', Base)
    lead_model = create_leads_model('leads', Base)
    message_model = create_message_model('messages', Base)
    
    return business_model, lead_model, message_model
