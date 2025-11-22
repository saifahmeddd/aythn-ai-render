from flask import Blueprint, request, jsonify, Response
from views import AythnView
import config

AYTHN_BLUEPRINT = Blueprint("aythn", __name__, url_prefix="/aythn")

# Content type for webhook responses
TEXT_PLAIN = "text/plain"


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