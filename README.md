# Aythn AI - Conversational Lead Qualification Agent

Aythn is an intelligent conversational AI agent designed to automatically qualify leads from Facebook Lead Ads. It uses LangChain and OpenAI to engage with leads in natural conversations, determine their eligibility based on domain-specific Q&A, and manage the entire lead qualification process.

## 🚀 Features

- **Automatic Lead Processing**: Receives leads from Facebook Lead Ads via webhooks
- **Conversational AI**: Powered by OpenAI GPT-4o-mini for natural, human-like conversations
- **Intelligent Eligibility Assessment**: Automatically determines lead eligibility based on conversation responses
- **Conversation History**: Maintains full conversation history using PostgreSQL with LangChain's SQLChatMessageHistory
- **Domain-Specific Knowledge**: Configurable Q&A system for different business domains (real estate, education, etc.)
- **Automatic Greetings**: Sends personalized initial greetings to new leads
- **RESTful API**: Complete API for querying, managing leads, and retrieving conversations

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Database Schema](#database-schema)
- [Domain Q&A Configuration](#domain-qa-configuration)
- [Usage Examples](#usage-examples)
- [Webhook Setup](#webhook-setup)
- [Development](#development)

## 🏗️ Architecture

```
┌─────────────────┐
│  Facebook Ads   │
│   Lead Forms    │
└────────┬────────┘
         │ Webhook
         ▼
┌─────────────────┐
│  Flask API      │
│  (main.py)      │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌──────────────┐
│PostgreSQL│ │  OpenAI API  │
│Database │ │  (GPT-4o)    │
└────────┘ └──────────────┘
```

### Key Components

- **Flask Application**: RESTful API server
- **LangChain**: Conversation management and memory
- **OpenAI GPT-4o-mini**: Conversational AI engine
- **PostgreSQL**: Lead and conversation storage
- **Facebook Webhooks**: Lead generation event handling

## 📦 Installation

### Prerequisites

- Python 3.8+
- PostgreSQL database
- OpenAI API key
- Facebook App credentials (for Lead Ads integration)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd aythn-ai
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Set Up Environment Variables

Create a `.env` file in the root directory:

```env
# Application Settings
DEBUG_MODE=True
HOST=0.0.0.0
APPLICATION_PORT=5000
FRONTEND_URL=http://localhost:3000

# Database
PG_CONN_STRING=postgresql://username:password@localhost:5432/aythn_db

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Facebook/Meta API
META_APP_ID=your_meta_app_id
META_ACCESS_TOKEN=your_meta_access_token
META_APP_SECRET=your_meta_app_secret
PAGE_ACCESS_TOKEN=your_page_access_token

# Webhook Configuration
WEBHOOK_VERIFY_TOKEN=your_webhook_verify_token
PUBLIC_BASE_URL=https://your-domain.com
WEBHOOK_CALLBACK_URL=https://your-domain.com/aythn/webhook

# Logging
APP_LOG=app.log
```

### Step 4: Initialize Database

The database tables are automatically created when you run the application. Ensure PostgreSQL is running and the connection string is correct.

### Step 5: Run the Application

```bash
python main.py
```

The application will:
- Initialize the database
- Start the Flask server
- Automatically subscribe to Facebook webhooks (after 5 seconds)

## ⚙️ Configuration

### Domain Q&A Configuration

Edit `domain-q&a.json` to configure your business domain questions and eligibility criteria:

```json
{
  "domains": {
    "real_estate": {
      "questions": [
        {
          "question": "Are you interested in buying or renting a property?",
          "answers": {
            "buy": {
              "response": "Great! What type of property are you looking for?",
              "next_route": "/property-type",
              "eligible": true
            },
            "rent": {
              "response": "Got it! What city are you looking to rent in?",
              "next_route": "/rent-location",
              "eligible": true
            }
          }
        }
      ],
      "eligibility_rules": {
        "description": "For real estate, leads are eligible if...",
        "required_answers": ["buy_or_rent", "budget_range"]
      }
    }
  },
  "global_eligibility_rules": {
    "minimum_answers_required": 1,
    "default_eligible": false
  }
}
```

## 🔌 API Endpoints

### Base URL
```
http://localhost:4000/aythn
```

### 1. Query Agent
Send a message to the conversational agent.

**POST** `/query`

**Request Body:**
```json
{
  "query": "I'm interested in buying a property",
  "lead_id": "123456789012345"
}
```

**Response:**
```json
{
  "response": {
    "text": "Great! What type of property are you looking for?",
    "next_route": "/property-type",
    "eligible": true,
    "conversation_ending": false
  }
}
```

### 2. Webhook (Facebook Lead Ads)
Handles Facebook webhook verification and lead events.

**GET** `/webhook` - Webhook verification
**POST** `/webhook` - Lead generation events

### 3. Get All Leads
Retrieve all leads from the database.

**GET** `/leads`

**Response:**
```json
{
  "leads": [
    {
      "leadgen_id": "123456789012345",
      "name": "John Doe",
      "email": "john@example.com",
      "eligible": true,
      "created_at": "2024-01-15T10:30:00"
    }
  ]
}
```

### 4. Get Conversation
Retrieve conversation history for a specific lead.

**POST** `/conversation`

**Request Body:**
```json
{
  "lead_id": "123456789012345"
}
```

**Response:**
```json
{
  "lead_id": "123456789012345",
  "messages": [
    {
      "type": "AIMessage",
      "content": "Hello! Thanks for reaching out...",
      "created_at": "2024-01-15T10:30:00"
    },
    {
      "type": "HumanMessage",
      "content": "I'm interested in buying",
      "created_at": "2024-01-15T10:31:00"
    }
  ],
  "message_count": 2
}
```

### 5. Finalize Eligibility
Manually trigger final eligibility evaluation (also happens automatically when conversation ends).

**POST** `/conversation/finalize-eligibility`

**Request Body:**
```json
{
  "leadgen_id": "123456789012345"
}
```

**Response:**
```json
{
  "leadgen_id": "123456789012345",
  "eligible": true,
  "eligible_answers_count": 2,
  "ineligible_answers_count": 0,
  "eligible_answers": ["buy", "medium"],
  "total_user_messages": 3
}
```

### 6. Delete User Data
Delete user data (GDPR compliance).

**POST** `/webhook/delete-user-data`

**Request Body:**
```json
{
  "lead_id": "123456789012345"
}
```

## 🗄️ Database Schema

### Leads Table
- `leadgen_id` (String, Primary Key): Facebook leadgen_id from webhook
- `name` (Text, Nullable): Lead's full name
- `email` (Text, Nullable): Lead's email address
- `eligible` (Boolean): Eligibility status
- `created_at` (DateTime): Lead creation timestamp

### Messages Table
- `id` (Integer, Primary Key): Auto-incrementing ID
- `lead_id` (String, Foreign Key): References `leads.leadgen_id`
- `session_id` (String, Indexed): LangChain session identifier
- `message` (Text): JSON-encoded LangChain message
- `created_at` (DateTime): Message timestamp

## 📝 Usage Examples

### Starting a Conversation

When a new lead is created via Facebook webhook, the system automatically:
1. Saves the lead to the database
2. Sends an initial greeting message
3. Initializes the conversation history

### Querying the Agent

```python
import requests

response = requests.post(
    "http://localhost:5000/aythn/query",
    json={
        "query": "I'm looking to buy a house",
        "lead_id": "123456789012345"
    }
)

result = response.json()
print(result["response"]["text"])
print(f"Eligible: {result['response']['eligible']}")
```

### Retrieving Conversation History

```python
response = requests.post(
    "http://localhost:5000/aythn/conversation",
    json={"lead_id": "123456789012345"}
)

conversation = response.json()
for message in conversation["messages"]:
    print(f"{message['type']}: {message['content']}")
```

## 🔗 Webhook Setup

### Facebook Lead Ads Webhook Configuration

1. **Create a Facebook App** in the [Facebook Developers Console](https://developers.facebook.com/)

2. **Set Up Webhook**:
   - Webhook URL: `https://your-domain.com/aythn/webhook`
   - Verify Token: Use the same token from your `.env` file
   - Subscribe to: `leadgen` field

3. **Webhook Verification**:
   - Facebook will send a GET request to verify your webhook
   - The application automatically handles verification

4. **Lead Events**:
   - When a lead is generated, Facebook sends a POST request
   - The application automatically:
     - Saves the lead
     - Sends initial greeting
     - Starts conversation tracking

## 🧪 Development

### Project Structure

```
aythn-ai/
├── main.py                 # Flask application entry point
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── domain-q&a.json       # Domain-specific Q&A configuration
├── database/
│   ├── models.py         # SQLAlchemy database models
│   └── __init__.py
├── routes/
│   ├── aythn.py          # API route definitions
│   └── __init__.py
├── views/
│   ├── aythn.py          # Business logic and agent implementation
│   └── __init__.py
└── utils/
    ├── message_converter.py  # LangChain message converter
    └── __init__.py
```

### Key Features Implementation

- **Conversation Memory**: Uses LangChain's `ConversationBufferWindowMemory` with PostgreSQL backend
- **Eligibility Detection**: Automatically detects conversation-ending signals and evaluates final eligibility
- **Message History**: Custom message converter stores messages with timestamps and session IDs

### Running Tests

```bash
# Add test files and run
python -m pytest tests/
```

### Logging

Application logs are written to `app.log` by default. Log level and format can be configured in `config.py`.

## 🔒 Security Considerations

- Store sensitive credentials in `.env` file (never commit to version control)
- Use HTTPS for webhook endpoints in production
- Implement rate limiting for API endpoints
- Validate and sanitize all user inputs
- Use strong webhook verify tokens
<!-- 
## 📄 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here] -->

## 📧 Support

For issues and questions, please open an issue in the repository.

## 🙏 Acknowledgments

- Built with [Flask](https://flask.palletsprojects.com/)
- Powered by [LangChain](https://www.langchain.com/) and [OpenAI](https://openai.com/)
- Database powered by [PostgreSQL](https://www.postgresql.org/)

