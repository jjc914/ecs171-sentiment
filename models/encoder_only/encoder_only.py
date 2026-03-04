import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_DIR = os.path.dirname(os.path.abspath(__file__))

LABELS = {0: "Negative", 1: "Positive"}

CONFIG = dict(
    seed             = 42,
    max_len          = 384,     # covers ~90 % of reviews; 512 is possible but slower
    vocab_size       = 30_000,  # top-K words → 99 %+ token coverage on IMDb
    d_model          = 512,     # hidden dim
    n_heads          = 8,       # 512 / 8 = 64 per head
    n_layers         = 6,       # 6 Pre-LN blocks ≈ 22 M trainable params
    dim_ff           = 2048,    # 4 × d_model (standard)
    dropout          = 0.15,    # slightly above 0.1 to fight overfit on 45 K samples
    batch_size       = 48,      # micro-batch that fits T4 with AMP
    grad_accum_steps = 2,       # effective batch = 96
    lr               = 5e-4,    # higher than fine-tune; good for from-scratch
    weight_decay     = 0.01,
    warmup_ratio     = 0.10,    # 10 % of total optimizer steps
    epochs           = 20,
    label_smoothing  = 0.05,
    patience         = 5,       # early stopping on val F1
    pooling          = "mean",  # "mean" = masked mean pooling (better from scratch)
)

OOM_LADDER = [
    dict(batch_size=32, grad_accum_steps=3),
    dict(max_len=256),
    dict(batch_size=16, grad_accum_steps=6),
    dict(n_layers=4),
    dict(d_model=256, dim_ff=1024, n_heads=4),
]

# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5
        self.qkv      = nn.Linear(d_model, 3 * d_model)
        self.out_proj  = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, H, L, d_h)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale      # (B, H, L, L)
        attn = attn.masked_fill(mask[:, None, None, :] == 0, float("-inf"))
        attn = self.attn_drop(attn.softmax(dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_ff, dropout):
        super().__init__()
        self.fc1  = nn.Linear(d_model, dim_ff)
        self.fc2  = nn.Linear(dim_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-LN: LayerNorm before attention / FFN for stable gradients."""
    def __init__(self, d_model, n_heads, dim_ff, dropout):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = FeedForward(d_model, dim_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = x + self.drop(self.attn(self.ln1(x), mask))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class SentimentTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pooling  = cfg["pooling"]
        self.tok_emb  = nn.Embedding(cfg["vocab_size"], cfg["d_model"], padding_idx=0)
        self.pos_emb  = nn.Embedding(cfg["max_len"],    cfg["d_model"])
        self.emb_drop = nn.Dropout(cfg["dropout"])

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg["d_model"], cfg["n_heads"],
                             cfg["dim_ff"],  cfg["dropout"])
            for _ in range(cfg["n_layers"])
        ])
        self.final_ln = nn.LayerNorm(cfg["d_model"])
        self.head = nn.Sequential(
            nn.LayerNorm(cfg["d_model"]),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["d_model"], 2),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, mask, inputs_embeds=None):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        if inputs_embeds is None:
            inputs_embeds = self.tok_emb(input_ids)
        x = self.emb_drop(inputs_embeds + self.pos_emb(pos))

        for blk in self.blocks:
            x = blk(x, mask)
        x = self.final_ln(x)

        if self.pooling == "cls":
            pooled = x[:, 0]
        else:                                       # masked mean pooling
            mask_f = mask.unsqueeze(-1).float()      # (B, L, 1)
            pooled = (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)

        return self.head(pooled)

def load_enc_model():
    print("Loading Encoder-only model...")

    path = os.path.join(_DIR, "best_model_encoder_only.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = dict(CONFIG)
    model = SentimentTransformer(cfg)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    vocab_path = os.path.join(_DIR, "word_tokenizer.json")
    if os.path.exists(vocab_path):
        import json
        with open(vocab_path) as f:
            tok_data = json.load(f)
        vocab = tok_data["token2id"]
    else:
        vocab = {}
        print("Warning: word_tokenizer.json not found — all tokens will map to <unk>")

    print("Finished loading Encoder-only model")

    return {
        "config": cfg,
        "model":  model,
        "device": device,
        "vocab":  vocab,
    }

def _tokenize(vocab, text, max_len):
    """Mirrors WordTokenizer.encode: prepends <cls>, appends <eos>, pads to max_len."""
    cls_id = vocab.get("<cls>", 2)
    eos_id = vocab.get("<eos>", 3)
    unk_id = vocab.get("<unk>", 1)

    words = text.lower().split()
    ids = [cls_id] + [vocab.get(w, unk_id) for w in words] + [eos_id]
    if len(ids) > max_len:
        ids = ids[:max_len - 1] + [eos_id]

    n_real  = len(ids)
    pad_len = max_len - n_real
    mask    = [1] * n_real + [0] * pad_len
    ids     = ids + [0] * pad_len

    # Display tokens: actual words only (strip <cls> at 0 and <eos> at n_real-1)
    display_words = words[:n_real - 2]
    return display_words, ids, mask, n_real


def get_enc_attributions(model_dict, text, n_steps=20):
    """
    Integrated Gradients attribution per token.
    Baseline = zero embeddings.  Always computed w.r.t. the Positive class (index 1)
    so that positive weight = pushes toward positive, negative = pushes toward negative.
    """
    model   = model_dict["model"]
    device  = model_dict["device"]
    cfg     = model_dict["config"]
    vocab   = model_dict["vocab"]
    max_len = cfg["max_len"]

    tokens, ids, mask, n_real = _tokenize(vocab, text, max_len)
    if n_real == 0:
        return []

    input_ids = torch.tensor([ids],  dtype=torch.long, device=device)
    mask_t    = torch.tensor([mask], dtype=torch.long, device=device)

    # Input and baseline embeddings
    emb_layer    = model.tok_emb
    input_embeds = emb_layer(input_ids).detach()
    baseline     = torch.zeros_like(input_embeds)

    # Integrated Gradients: accumulate gradients along the interpolation path
    total_grads = torch.zeros_like(input_embeds)
    for step in range(1, n_steps + 1):
        alpha  = step / n_steps
        interp = (baseline + alpha * (input_embeds - baseline)).requires_grad_(True)

        model.zero_grad()
        model(input_ids, mask_t, inputs_embeds=interp)[0, 1].backward()  # always w.r.t. Positive class
        total_grads += interp.grad.detach()

    ig     = (total_grads / n_steps) * (input_embeds - baseline)
    scores = ig.sum(dim=-1).squeeze(0).tolist()

    # positions: 0=<cls>, 1..n_real-2=words, n_real-1=<eos>
    word_scores = scores[1:n_real - 1]                    # strip <cls> and <eos>

    return [{"token": t, "weight": float(s)} for t, s in zip(tokens, word_scores)]


def infer_enc(model_dict, text):
    model   = model_dict["model"]
    device  = model_dict["device"]
    cfg     = model_dict["config"]
    vocab   = model_dict["vocab"]
    max_len = cfg["max_len"]

    _, ids, mask, _ = _tokenize(vocab, text, max_len)

    input_ids = torch.tensor([ids],  dtype=torch.long, device=device)
    mask_t    = torch.tensor([mask], dtype=torch.long, device=device)

    with torch.inference_mode():
        logits = model(input_ids, mask_t)

    probs = torch.softmax(logits, dim=-1)[0]
    idx   = int(probs.argmax().item())

    return {
        "label":        LABELS[idx],
        "score":        round(float(probs[idx].item()), 4),
        "attributions": get_enc_attributions(model_dict, text),
    }