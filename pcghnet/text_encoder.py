import re
import zlib
from typing import List

import torch
from torch import nn


class HashTextEncoder(nn.Module):
    """Small, dependency-free text encoder for development and ablations."""

    def __init__(self, output_dim=256, vocab_size=8192, embedding_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def _token_ids(self, text):
        tokens = re.findall(r"[\w']+|[^\w\s]", text.lower(), flags=re.UNICODE)
        if not tokens:
            return [1]
        return [2 + zlib.crc32(token.encode("utf-8")) % (self.vocab_size - 2) for token in tokens]

    def forward(self, prompts: List[str]):
        device = self.embedding.weight.device
        token_lists = [self._token_ids(prompt) for prompt in prompts]
        max_length = max(len(tokens) for tokens in token_lists)
        ids = torch.zeros(len(prompts), max_length, dtype=torch.long, device=device)
        mask = torch.zeros(len(prompts), max_length, dtype=torch.float32, device=device)
        for row, tokens in enumerate(token_lists):
            length = len(tokens)
            ids[row, :length] = torch.tensor(tokens, dtype=torch.long, device=device)
            mask[row, :length] = 1.0
        embedded = self.embedding(ids)
        pooled = (embedded * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)
        return self.projection(pooled)


class MiniLMTextEncoder(nn.Module):
    """Frozen MiniLM encoder used by the proposed model during experiments."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", output_dim=256):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "MiniLM requires transformers. Install it with `pip install transformers`."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Sequential(nn.Linear(hidden_size, output_dim), nn.LayerNorm(output_dim))
        self.output_dim = output_dim

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, prompts: List[str]):
        device = next(self.parameters()).device
        tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            hidden = self.encoder(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return self.projection(pooled)


def build_text_encoder(name, output_dim=256, model_name=None):
    if name == "hash":
        return HashTextEncoder(output_dim=output_dim)
    if name == "minilm":
        return MiniLMTextEncoder(
            model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2",
            output_dim=output_dim,
        )
    raise ValueError("Unknown text encoder: {}".format(name))
