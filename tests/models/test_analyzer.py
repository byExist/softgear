"""Tests for Analyzer."""

import torch
from torch import nn

from softgear.models.analyzer import Analyzer, ModelOutput
from softgear.models.chain import Chain
from softgear.models.gear import Gear

B, SEQ, D, HEADS, FFN = 2, 16, 32, 4, 64
VOCAB = 5


def _make_gear() -> Gear:
    return Gear(D, HEADS, FFN, dropout=0.0)


def _make_analyzer() -> Analyzer:
    encoder = nn.Embedding(VOCAB, D)
    decoder = nn.Linear(D, VOCAB)
    return Analyzer(encoder, decoder, Chain(), hidden_dim=D)


def test_forward_returns_model_output():
    model = _make_analyzer()
    x = torch.randint(0, VOCAB, (B, SEQ))
    out = model(x)
    assert isinstance(out, ModelOutput)
    assert out.logits.shape == (B, SEQ, VOCAB)


def test_parameter_count_increases_with_gears():
    model = _make_analyzer()
    base_count = model.parameter_count()
    model.chain.mount(_make_gear())
    assert model.parameter_count() > base_count


def test_forward_works_with_gears():
    model = _make_analyzer()
    model.chain.mount(_make_gear())
    model.chain.mount(_make_gear())
    x = torch.randint(0, VOCAB, (B, SEQ))
    assert model(x).logits.shape == (B, SEQ, VOCAB)
