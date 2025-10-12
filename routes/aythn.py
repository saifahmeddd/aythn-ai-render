from flask import Blueprint, request, jsonify
from views import AythnView

AYTHN_BLUEPRINT = Blueprint("aythn", __name__, url_prefix="/aythn")


@AYTHN_BLUEPRINT.route("/query", methods=["POST"])
def run_agent():
    return True
