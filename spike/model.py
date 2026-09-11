"""Bidirectional gated-scan tagger, ported from gpu-time's TimeTagger.

The architecture is deliberately unchanged: a summed sparse-feature embedding,
a depthwise window convolution, two neighbor gathers, forward and backward
affine scans, a mean-pooled global context, and a two-layer head emitting role
logits plus one clause-boundary logit.

Only the feature table and the label set differ from gpu-time. Keeping the
backbone identical is the point — the spike tests whether the representation
transfers, not whether a better architecture exists.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

HIDDEN = 32


def affine_scan(gate: Tensor, candidate: Tensor) -> Tensor:
    """Inclusive scan for state[t] = gate[t] * state[t-1] + candidate[t]."""
    width = gate.shape[1]
    stride = 1
    while stride < width:
        next_gate = gate[:, stride:] * gate[:, :-stride]
        next_candidate = (
            candidate[:, stride:] + gate[:, stride:] * candidate[:, :-stride]
        )
        gate = torch.cat((gate[:, :stride], next_gate), dim=1)
        candidate = torch.cat((candidate[:, :stride], next_candidate), dim=1)
        stride *= 2
    return candidate


def quantize(weight: Tensor, bits: int = 6) -> Tensor:
    maximum = (1 << (bits - 1)) - 1
    scale = (weight.detach().abs().max() / maximum).clamp_min(1e-8)
    quantized = (weight / scale).round().clamp(-maximum, maximum) * scale
    return weight + (quantized - weight).detach()


class QueryTagger(nn.Module):
    def __init__(self, feature_rows: int, role_classes: int) -> None:
        super().__init__()
        self.feature_rows = feature_rows
        self.role_classes = role_classes
        self.embedding = nn.Parameter(torch.empty(feature_rows, HIDDEN))
        self.encoder_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.convolution = nn.Parameter(torch.empty(5, HIDDEN))
        self.neighbor_weights = nn.Parameter(torch.empty(2, HIDDEN))
        self.gate_weight = nn.Parameter(torch.empty(HIDDEN, HIDDEN))
        self.gate_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.candidate_weight = nn.Parameter(torch.empty(HIDDEN, HIDDEN))
        self.candidate_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.combine_weight = nn.Parameter(torch.empty(HIDDEN, HIDDEN * 2))
        self.combine_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.global_weight = nn.Parameter(torch.empty(HIDDEN, HIDDEN))
        self.global_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.head_gate_weight = nn.Parameter(torch.empty(16, HIDDEN * 2))
        self.head_gate_bias = nn.Parameter(torch.zeros(16))
        self.head_hidden_weight = nn.Parameter(torch.empty(64, HIDDEN * 2 + 16))
        self.head_hidden_bias = nn.Parameter(torch.zeros(64))
        self.output_weight = nn.Parameter(torch.empty(role_classes + 1, 64))
        self.output_bias = nn.Parameter(torch.zeros(role_classes + 1))
        self.quantization_bits = 6
        self.qat = False
        # Set to capture every intermediate for trace export. Off by default so
        # training pays nothing. Mirrors gpu-time's TimeTagger.
        self.record_trace = False
        self.trace: dict = {}

        for _name, parameter in self.named_parameters():
            if parameter.ndim >= 2:
                nn.init.xavier_uniform_(parameter)
        nn.init.normal_(self.embedding, std=0.08)
        nn.init.normal_(self.convolution, std=0.15)
        nn.init.normal_(self.neighbor_weights, std=0.1)
        with torch.no_grad():
            # Give lanes short and long memories from the start.
            self.gate_bias.copy_(torch.linspace(0.0, 4.0, HIDDEN))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def weight(self, name: str) -> Tensor:
        value = getattr(self, name)
        return quantize(value, self.quantization_bits) if self.qat else value

    def linear(self, value: Tensor, name: str) -> Tensor:
        return F.linear(
            value, self.weight(f"{name}_weight"), self.weight(f"{name}_bias")
        )

    def forward(
        self, rows: Tensor, valid: Tensor, neighbors: Tensor
    ) -> tuple[Tensor, Tensor]:
        row_mask = (rows != self.feature_rows).unsqueeze(-1)
        embedded = F.embedding(
            rows.clamp_max(self.feature_rows - 1), self.weight("embedding")
        )
        embedded = (embedded * row_mask).sum(dim=2)
        embedded = embedded * valid.unsqueeze(-1)

        channels = embedded.transpose(1, 2)
        kernel = self.weight("convolution").transpose(0, 1).unsqueeze(1)
        encoded = F.conv1d(channels, kernel, padding=2, groups=HIDDEN).transpose(1, 2)
        encoded = encoded + self.weight("encoder_bias")
        for side in range(2):
            indices = neighbors[:, :, side]
            gathered = embedded.gather(
                1, indices.clamp_min(0).unsqueeze(-1).expand(-1, -1, HIDDEN)
            )
            encoded = (
                encoded
                + gathered
                * (indices >= 0).unsqueeze(-1)
                * self.weight("neighbor_weights")[side]
            )
        encoded = torch.tanh(encoded) * valid.unsqueeze(-1)

        gate = torch.sigmoid(self.linear(encoded, "gate"))
        candidate = (1 - gate) * torch.tanh(self.linear(encoded, "candidate"))
        gate = torch.where(valid.unsqueeze(-1), gate, torch.ones_like(gate))
        candidate = candidate * valid.unsqueeze(-1)
        forward = affine_scan(gate, candidate)
        backward = affine_scan(gate.flip(1), candidate.flip(1)).flip(1)
        combined = torch.tanh(
            encoded + self.linear(torch.cat((forward, backward), dim=-1), "combine")
        )
        combined = combined * valid.unsqueeze(-1)

        pooled = combined.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        context = torch.sigmoid(self.linear(pooled, "global")) * pooled
        joined = torch.cat((combined, context.unsqueeze(1).expand_as(combined)), dim=-1)
        head_gate = torch.sigmoid(self.linear(joined, "head_gate"))
        hidden = torch.tanh(
            self.linear(torch.cat((joined, head_gate), dim=-1), "head_hidden")
        )
        if self.record_trace:
            self.trace = {
                name: value.detach()
                for name, value in {
                    "embedded": embedded,
                    "encoded": encoded,
                    "gate": gate,
                    "candidate": candidate,
                    "forward": forward,
                    "backward": backward,
                    "combined": combined,
                    "pooled": pooled,
                    "context": context,
                    "hidden": hidden,
                }.items()
            }
        output = self.linear(hidden, "output")
        return output[..., : self.role_classes], output[..., self.role_classes]
