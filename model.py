"""Backbone-agnostic autoregressive model for hierarchical item identifiers.

The recommender intentionally contains no joint-tokenizer objective. Item SIDs
are prepared before recommendation training and remain fixed. Any
encoder-decoder that exposes T5-style encoder/decoder hidden states can be used
as the sequence backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers.modeling_outputs import ModelOutput


@dataclass
class SIDModelOutput(ModelOutput):
    """Outputs used by the base objective and HGIG's two modules."""

    logits: Optional[torch.FloatTensor] = None
    base_logits: Optional[torch.FloatTensor] = None
    encoder_state: Optional[torch.FloatTensor] = None
    decoder_states: Optional[torch.FloatTensor] = None
    intent_vectors: Optional[torch.FloatTensor] = None
    route_scores: Optional[torch.FloatTensor] = None
    route_gates: Optional[torch.FloatTensor] = None
    route_entropy: Optional[torch.FloatTensor] = None
    route_margins: Optional[torch.FloatTensor] = None
    route_gate_logits: Optional[torch.FloatTensor] = None


class GenerativeSIDModel(nn.Module):
    """Generate a fixed-length SID from a sequence of historical SIDs."""

    supports_last_depth_routing = True

    def __init__(
        self,
        config,
        backbone,
        code_length: int,
        code_number: int,
        sid_centroids: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.code_length = int(code_length)
        self.code_number = int(code_number)
        self.hidden_size = int(backbone.config.d_model)
        self.num_beams = int(config.get("num_beams", 20))
        self.decoder_start_token_id = int(backbone.config.decoder_start_token_id or 0)
        self.token_embeddings = nn.ModuleList(
            [nn.Embedding(self.code_number, self.hidden_size) for _ in range(self.code_length)]
        )
        self.topology_router = None
        self.prototype_index = None
        self.topology_enabled = True
        self.sid_initialization = "gaussian"
        self.reset_parameters(
            sid_centroids=sid_centroids,
            target_std=float(config.get("sid_embedding_init_std", 0.02)),
        )

    @staticmethod
    def _normal_initialization(module: nn.Module) -> None:
        """Use the compact SID trunk's original, deterministic initialization.

        Applying one initializer to the wrapper and T5 trunk avoids silently
        comparing two architectures whose attention/FFN scales differ only
        because they were constructed through different code paths.
        """

        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def reset_parameters(
        self,
        sid_centroids: Optional[torch.Tensor] = None,
        target_std: float = 0.02,
    ) -> None:
        self.apply(self._normal_initialization)
        if sid_centroids is None:
            return
        centroids = torch.as_tensor(sid_centroids, dtype=torch.float32)
        if centroids.ndim != 3:
            raise ValueError("sid_centroids must have [depth, code, dimension] shape")
        if centroids.size(0) > self.code_length or centroids.size(1) != self.code_number:
            raise ValueError("sid_centroids do not match the configured SID vocabulary")
        if centroids.size(2) != self.hidden_size:
            raise ValueError(
                "Centroid initialization requires tokenizer and T5 dimensions to match; "
                f"got {centroids.size(2)} and {self.hidden_size}"
            )
        if float(target_std) <= 0.0:
            raise ValueError("sid_embedding_init_std must be positive")
        with torch.no_grad():
            for depth in range(centroids.size(0)):
                # The T5 trunk is trained from scratch, so there is no lexical
                # embedding mean to match.  Center each RQ level and preserve
                # its geometry at the same RMS scale as the Gaussian trunk.
                values = centroids[depth] - centroids[depth].mean(dim=0, keepdim=True)
                rms = values.square().mean().sqrt().clamp_min(1e-8)
                values = values * (float(target_std) / rms)
                self.token_embeddings[depth].weight.copy_(values)
        self.sid_initialization = "rq_centroid_geometry"

    def configure_topology(self, router=None, prototype_index=None) -> None:
        """Attach or detach Module 1 without changing the common backbone."""
        if router is not None:
            if prototype_index is None:
                raise ValueError("A topology router requires a prototype index")
            # Materialize catalog-sized parameters before the optimizer is
            # constructed.  Creating them lazily in the first forward would
            # leave those tensors outside every optimizer parameter group.
            router.bind_prototype_index(prototype_index)
        self.topology_router = router
        self.prototype_index = prototype_index

    def get_input_embeddings(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Embed a flattened history while preserving SID depth identity."""
        safe_ids = input_ids.clamp_min(0)
        result = torch.zeros(
            *safe_ids.shape,
            self.hidden_size,
            dtype=self.token_embeddings[0].weight.dtype,
            device=input_ids.device,
        )
        for depth, embedding in enumerate(self.token_embeddings):
            result[:, depth :: self.code_length] = embedding(
                safe_ids[:, depth :: self.code_length]
            )
        # Padding positions use the shared start embedding. Attention still
        # masks these positions as keys.
        padding = self.backbone.shared.weight[0].view(1, 1, -1)
        return torch.where(attention_mask.unsqueeze(-1), result, padding)

    def _decoder_embeddings(self, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        columns = []
        for position in range(decoder_input_ids.size(1)):
            token_ids = decoder_input_ids[:, position].clamp_min(0)
            if position == 0:
                columns.append(self.backbone.shared(token_ids))
            else:
                depth = min(position - 1, self.code_length - 1)
                columns.append(self.token_embeddings[depth](token_ids))
        return torch.stack(columns, dim=1)

    def _shift_labels(self, labels: torch.Tensor) -> torch.Tensor:
        start = torch.full(
            (labels.size(0), 1),
            self.decoder_start_token_id,
            dtype=torch.long,
            device=labels.device,
        )
        return torch.cat([start, labels[:, :-1]], dim=1)

    def _depth_logits(self, decoder_states: torch.Tensor) -> torch.Tensor:
        columns = []
        for depth in range(decoder_states.size(1)):
            embedding = self.token_embeddings[min(depth, self.code_length - 1)].weight
            # Keep the tied SID head on the same scale as the reference trunk.
            # An earlier d_model**-0.5 factor flattened these logits while the
            # semantic router stayed unscaled, allowing it to dominate.
            columns.append(torch.matmul(decoder_states[:, depth], embedding.t()))
        return torch.stack(columns, dim=1)

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ):
        """Encode a history once for reuse by constrained beam decoding."""

        inputs_embeds = self.get_input_embeddings(input_ids, attention_mask)
        return self.backbone.get_encoder()(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        encoder_outputs=None,
        routing_depth: Optional[int] = None,
        **_kwargs,
    ) -> SIDModelOutput:
        if attention_mask is None:
            raise ValueError("attention_mask is required")
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings(input_ids, attention_mask)
        if decoder_input_ids is None:
            if labels is None:
                decoder_input_ids = torch.full(
                    (attention_mask.size(0), 1),
                    self.decoder_start_token_id,
                    dtype=torch.long,
                    device=attention_mask.device,
                )
            else:
                decoder_input_ids = self._shift_labels(labels)

        outputs = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_inputs_embeds=self._decoder_embeddings(decoder_input_ids),
            encoder_outputs=encoder_outputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        base_logits = self._depth_logits(outputs.decoder_hidden_states[-1])
        encoder_state = self._mean_pool(outputs.encoder_last_hidden_state, attention_mask)
        decoder_states = outputs.decoder_hidden_states[-1]

        logits = base_logits
        intent_vectors = route_scores = route_gates = route_entropy = route_margins = None
        route_gate_logits = None
        route_this_step = (routing_depth is None or (self.topology_router is not None
                           and routing_depth < self.topology_router.semantic_depth))
        if self.topology_enabled and self.topology_router is not None and self.prototype_index is not None and route_this_step:
            routed = self.topology_router(
                base_logits=base_logits,
                encoder_state=encoder_state,
                decoder_states=decoder_states,
                prefixes=decoder_input_ids[:, 1:],
                prototype_index=self.prototype_index,
                target_tokens=labels[:, : base_logits.size(1)] if labels is not None else None,
                active_depth=routing_depth,
            )
            logits = routed.logits
            intent_vectors = routed.intent_vectors
            route_scores = routed.route_scores
            route_gates = routed.gates
            route_entropy = routed.entropy
            route_margins = routed.margins
            route_gate_logits = routed.gate_logits

        return SIDModelOutput(
            logits=logits,
            base_logits=base_logits,
            encoder_state=encoder_state,
            decoder_states=decoder_states,
            intent_vectors=intent_vectors,
            route_scores=route_scores,
            route_gates=route_gates,
            route_entropy=route_entropy,
            route_margins=route_margins,
            route_gate_logits=route_gate_logits,
        )


Model = GenerativeSIDModel
