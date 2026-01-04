from flask import Blueprint, request, jsonify, Response
from twilio.twiml.messaging_response import MessagingResponse

from config import MAKE_SECRET
from views import AythnView
import config

AYTHN_BLUEPRINT = Blueprint("aythn", __name__, url_prefix="/aythn")

# Content type for webhook responses
TEXT_PLAIN = "text/plain"
MAKE_SECRET=config.MAKE_SECRET

@AYTHN_BLUEPRINT.route("/new-lead", methods=["POST"])
def receive_lead():
    # simple verification via header 
    if request.headers.get('X-Make-Secret') != MAKE_SECRET:
        return jsonify({"error": "forbidden"}), 403

    payload = request.get_json(force=True)
    print("🔥 New Lead Received: %s", payload)
    
    try:
        # Extract lead_id from payload (could be 'lead_id' or 'leadgen_id')
        lead_id = payload.get('lead_id') or payload.get('leadgen_id')
        if not lead_id:
            return jsonify({"error": "lead_id is required"}), 400
        
        # Save lead data to database
        # Get business_id from payload
        business_id = payload.get('business_id')
        new_lead, leadgen_id, lead_name, lead_email, lead_phone, _ = AythnView.save_lead_to_db(
            lead_id, payload, business_id
        )

        if new_lead and leadgen_id:
            print(f"Lead saved successfully: {lead_name} ({lead_email}) with leadgen_id: {leadgen_id}, phone: {lead_phone}")
            
            # Send WhatsApp template message with first name
            if lead_phone:
                print(f"Sending WhatsApp template message to {lead_phone}...")
                template_result = AythnView.sendWhatsAppTemplate(lead_phone, lead_name)
                if template_result.get('error'):
                    print(f"Failed to send WhatsApp template: {template_result.get('error')}")
                else:
                    print(f"WhatsApp template sent successfully. SID: {template_result.get('message_sid')}")
            else:
                print(f"Warning: No phone number found for lead {leadgen_id}. Cannot send WhatsApp message.")
            
            return jsonify({
                "status": "received",
                "leadgen_id": leadgen_id,
                "message": "Lead saved and WhatsApp template sent" if lead_phone else "Lead saved (no phone number)"
            }), 200
        else:
            return jsonify({"error": "Failed to save lead to database"}), 500
            
    except Exception as e:
        print(f"Error processing lead: {e}")
        return jsonify({"error": str(e)}), 500


@AYTHN_BLUEPRINT.route("/query", methods=["POST"])
def run_agent():
    """
    Endpoint to handle conversational agent requests.
    """
    try:
        # Extract user input from the request
        user_input = request.json.get("query")
        lead_id = request.json.get("lead_id")

        if not user_input:
            return jsonify({"error": "Query is required"}), 400
        if not lead_id:
            return jsonify({"error": "Lead ID is required"}), 400

        # Call the LangChain agent function
        result = AythnView.run_agent(user_input, lead_id)
        return jsonify({"response": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@AYTHN_BLUEPRINT.route("/webhook", methods=["GET", "POST"])
def webhook():
    """
    Endpoint to handle webhook events.
    For GET: Facebook webhook verification
    For POST: Facebook webhook events
    """
    if request.method == "GET":
        mode = request.args.get("hub.mode")
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")

        # Verify the token and respond with the challenge
        if mode == "subscribe" and token == config.WEBHOOK_VERIFY_TOKEN and challenge:
            return Response(challenge, mimetype=TEXT_PLAIN), 200
        else:
            return Response("Verification failed", mimetype=TEXT_PLAIN), 403

    if request.method == "POST":
        data = request.get_json()
        print("Received webhook:", data)
        AythnView.store_lead_data(data)
        return Response("EVENT_RECEIVED", mimetype=TEXT_PLAIN), 200


@AYTHN_BLUEPRINT.route("/leads", methods=["GET"])
def list_leads():
    """
    Return all leads stored in the database.
    """
    try:
        result = AythnView.get_all_leads()
        status = 200 if "error" not in result else 500
        return jsonify(result), status
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@AYTHN_BLUEPRINT.route("/leads/eligibility", methods=["GET"])
def leads_eligibility():
    """
    Return all lead IDs and their eligibility status.
    Returns a simplified list with leadgen_id and eligible status, plus summary statistics.
    """
    try:
        result = AythnView.get_leads_eligibility()
        status = 200 if "error" not in result else 500
        return jsonify(result), status
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@AYTHN_BLUEPRINT.route("/conversation", methods=["POST"])
def lead_conversation():
    """
    Endpoint to retrieve conversation history for a specific lead.
    """    
    try:
        data = request.get_json()
        lead_id = data.get("lead_id")
        conversation = AythnView.get_conversation(lead_id)
        status = 200 if "error" not in conversation else 500
        return jsonify(conversation), status
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@AYTHN_BLUEPRINT.route("/conversation/finalize-eligibility", methods=["POST"])
def finalize_eligibility():
    """
    Endpoint to finalize and update eligibility when conversation ends.
    Evaluates all conversation messages and updates the lead's eligible flag.
    """
    try:
        data = request.get_json()
        leadgen_id = data.get("leadgen_id") or data.get("lead_id")
        
        if not leadgen_id:
            return jsonify({"error": "Leadgen ID is required"}), 400
        
        result = AythnView.evaluate_final_eligibility(leadgen_id)
        status = 200 if "error" not in result else 500
        return jsonify(result), status
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@AYTHN_BLUEPRINT.route("/webhook/delete-user-data", methods=["POST"])
def delete_user_data():
    """
    Endpoint to handle user data deletion requests.
    """
    try:
        data = request.get_json()
        lead_id = data.get("lead_id")

        if not lead_id:
            return jsonify({"error": "User ID is required"}), 400

        result = AythnView.delete_user_data(lead_id)
        status = 200 if "error" not in result else 500
        return jsonify(result), status

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@AYTHN_BLUEPRINT.route("/twilio/whatsapp/receive-message", methods=["POST"])
def receive_whatsapp_message():
    """
    Endpoint to receive messages from WhatsApp via Twilio webhook.
    Returns TwiML response for Twilio.
    """
    try:
        # Twilio sends form data, not JSON
        data = {
            'From': request.form.get('From'),
            'To': request.form.get('To'),
            'Body': request.form.get('Body'),
            'MessageSid': request.form.get('MessageSid'),
            'AccountSid': request.form.get('AccountSid')
        }
        
        print("Received WhatsApp message:", data)
        
        # Process the message using AythnView
        twiml_response = AythnView.receive_message(data)
        
        # Return TwiML response with proper content type
        return Response(twiml_response, mimetype='text/xml'), 200
        
    except Exception as e:
        print(f"Error processing WhatsApp message: {e}")
        # Return error TwiML response
        response = MessagingResponse()
        response.message("Sorry, there was an error processing your message.")
        return Response(str(response), mimetype='text/xml'), 200