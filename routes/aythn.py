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


    """
    Return all leads stored in the database.
    """
    try:
        result = AythnView.get_all_leads()
        status = 200 if "error" not in result else 500
        return jsonify(result), status
    except Exception as e:
        return jsonify({"error": str(e)}), 500