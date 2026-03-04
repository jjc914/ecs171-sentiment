import os
import sys

from flask import Flask, jsonify, render_template, request

# Import models
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "models", "naive_bayes"))
sys.path.insert(0, os.path.join(_ROOT, "models", "encoder_only"))
sys.path.insert(0, os.path.join(_ROOT, "models", "lora_bert"))

from naive_bayes import infer_nb, load_nb_model
from encoder_only import infer_enc, load_enc_model
from lora_bert import infer_bert, load_bert_model

app = Flask(__name__)
nb_dict = None
enc_dict = None
bert_dict = None

"""
Web routes
"""
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")
    results = {
        "naive_bayes": infer_nb(nb_dict, text),
        "encoder": infer_enc(enc_dict, text),
        "bert": infer_bert(bert_dict, text),
    }
    return jsonify({"results": results})

# Startup
if __name__ == "__main__":
    nb_dict   = load_nb_model()
    enc_dict  = load_enc_model()
    bert_dict = load_bert_model()

    app.run(debug=True)
