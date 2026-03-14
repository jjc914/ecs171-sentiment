import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_DIR = os.path.dirname(os.path.abspath(__file__))

LABELS = {0: "Negative", 1: "Positive"}

def load_bert_model():
    print("Loading LoRA BERT model...")

    path = os.path.join(_DIR, "LoRA_BERT_IMDB_model", "merged_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(path)
    model     = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(device)
    model.eval()

    print("Finished loading LoRA BERT model")

    return {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
    }

def get_bert_attributions(model_dict, text, n_steps=20):
    """
    Integrated Gradients attribution per token.
    Baseline = zero embeddings.  Always computed w.r.t. the Positive class (index 1)
    so that positive weight = pushes toward positive, negative = pushes toward negative.
    """
    tokenizer = model_dict["tokenizer"]
    model     = model_dict["model"]
    device    = model_dict["device"]

    enc = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    input_ids      = enc["input_ids"]
    attention_mask = enc.get("attention_mask")

    # Input and baseline embeddings
    emb_layer    = model.get_input_embeddings()
    input_embeds = emb_layer(input_ids).detach()
    baseline     = torch.zeros_like(input_embeds)

    # Integrated Gradients: accumulate gradients along the interpolation path
    total_grads = torch.zeros_like(input_embeds)
    for step in range(1, n_steps + 1):
        alpha  = step / n_steps
        interp = (baseline + alpha * (input_embeds - baseline)).requires_grad_(True)

        kw = {"inputs_embeds": interp}
        if attention_mask is not None:
            kw["attention_mask"] = attention_mask

        model.zero_grad()
        model(**kw).logits[0, 1].backward()  # always w.r.t. Positive class
        total_grads += interp.grad.detach()

    ig     = (total_grads / n_steps) * (input_embeds - baseline)
    scores = ig.sum(dim=-1).squeeze(0).tolist()

    tokens  = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    special = set(tokenizer.all_special_tokens)

    # Merge WordPiece continuation tokens (##) back into whole words
    merged = []
    for token, score in zip(tokens, scores):
        if token in special:
            continue
        if token.startswith("##") and merged:
            merged[-1]["token"] += token[2:]
            merged[-1]["weight"] += float(score)
        else:
            merged.append({"token": token, "weight": float(score)})
    return merged


def infer_bert(model_dict, text):
    tokenizer = model_dict["tokenizer"]
    model     = model_dict["model"]
    device    = model_dict["device"]

    enc = tokenizer(
        text, truncation=True, padding=True, max_length=512, return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits

    probs = torch.softmax(logits, dim=-1)[0]
    idx   = int(probs.argmax())

    return {
        "label":        LABELS[idx],
        "score":        round(float(probs[idx]), 4),
        "attributions": get_bert_attributions(model_dict, text),
    }
