from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)
from langchain_community.chat_message_histories.sql import BaseMessageConverter
from sqlalchemy.ext.declarative import declarative_base
from typing import Any
import json
from datetime import datetime
from sqlalchemy.orm import Session
from database.models import create_message_model, create_leads_model


class MessageConverterWithDateTime(BaseMessageConverter):
    """Custom message converter for SQLChatMessageHistory that store messages with Date Time and Session ID in the DB."""

    def __init__(self, table_name: str, leads_table_name: str = 'leads'):
        # Use a single shared Base so that foreign keys resolve correctly
        shared_base = declarative_base()
        self.leads_model_class = create_leads_model(leads_table_name, shared_base)
        self.model_class = create_message_model(table_name, shared_base)

    def from_sql_model(self, sql_message: Any) -> BaseMessage:
        """
        Convert a SQL model to a BaseMessage object.

        Args:
            sql_message (Any): The SQL model to convert.

        Returns:
            BaseMessage: The converted BaseMessage object.
        """
        return messages_from_dict([json.loads(sql_message.message)])[0]

    def to_sql_model(self, message: BaseMessage, session_id: str) -> Any:
        """
        Converts a given message and session ID into a SQL model object.

        Args:
            message (BaseMessage): The message to be converted.
            session_id (str): The session identifier used by LangChain to partition histories.

        Returns:
            Any: The SQL model object representing the converted message.
        """
        current_time = datetime.now()
        # Our app uses session_id = str(lead_id), derive lead_id for FK consistency
        try:
            derived_lead_id = int(session_id)
        except (TypeError, ValueError):
            derived_lead_id = None
        return self.model_class(
            session_id=session_id,
            lead_id=derived_lead_id,
            message=json.dumps(message_to_dict(message)),
            created_at=current_time,
        )

    def get_sql_model_class(self) -> Any:
        return self.model_class

    def get_leads_model_class(self) -> Any:
        return self.leads_model_class

    # A function that clears or delete the messages for the given date range for the specific lead_id
    def clear_messages(
        self,
        session: Session,
        lead_id: int,
        start_date: datetime,
        end_date: datetime,
    ):
        """
        Clears or deletes the messages for the given date range for the specific lead_id.

        Args:
            session (Session): The SQLAlchemy session to use for database operations.
            lead_id (int): The lead ID to filter the messages.
            start_date (datetime): The start date of the range.
            end_date (datetime): The end date of the range.
        """
        # Creating the query
        query = session.query(self.model_class).filter(
            self.model_class.lead_id == lead_id,
            self.model_class.created_at.between(start_date, end_date),
        )
        # Executing the query
        query.delete(synchronize_session="fetch")
        session.commit()
