"""MAFNet missingness-aware temporal fusion model."""

from __future__ import annotations

import torch
import torch.nn as nn


class MissingnessAwareTemporalEncoder(nn.Module):
    """GRU-D-inspired temporal encoder with transformer refinement."""

    def __init__(
        self,
        n_channels: int,
        hidden_dim: int = 128,
        n_time_steps: int = 24,
        n_heads: int = 4,
        transformer_ff_dim: int = 256,
        dropout: float = 0.15,
        transformer_layers: int = 1,
        use_decay: bool = True,
        use_transformer: bool = True,
    ) -> None:
        super().__init__()
        if n_channels <= 0:
            raise ValueError("n_channels must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.n_channels = int(n_channels)
        self.hidden_dim = int(hidden_dim)
        self.n_time_steps = int(n_time_steps)
        self.use_decay = bool(use_decay)
        self.use_transformer = bool(use_transformer)

        self.input_decay_w = nn.Parameter(torch.zeros(n_channels))
        self.input_decay_b = nn.Parameter(torch.zeros(n_channels))
        self.hidden_decay = nn.Linear(n_channels, hidden_dim)

        self.input_projection = nn.Sequential(
            nn.Linear(4 * n_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_time_steps, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=transformer_layers,
            )
            if self.use_transformer and transformer_layers > 0
            else nn.Identity()
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.output_projection = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.20),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        count: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, T, V]")
        if x.shape != mask.shape or x.shape != delta.shape or x.shape != count.shape:
            raise ValueError("x, mask, delta, and count must have matching shapes")
        batch_size, n_steps, n_channels = x.shape
        if n_channels != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels, got {n_channels}")
        if n_steps > self.n_time_steps:
            raise ValueError(f"expected at most {self.n_time_steps} time steps, got {n_steps}")

        h = x.new_zeros(batch_size, self.hidden_dim)
        x_last = x.new_zeros(batch_size, n_channels)
        hidden_states = []

        for step in range(n_steps):
            x_t = x[:, step, :]
            m_t = mask[:, step, :]
            d_t = delta[:, step, :]
            c_t = count[:, step, :]

            if self.use_decay:
                gamma_x = torch.exp(-torch.relu(self.input_decay_w * d_t + self.input_decay_b))
                x_decay = gamma_x * x_last
                x_hat = m_t * x_t + (1.0 - m_t) * x_decay
                x_last = m_t * x_t + (1.0 - m_t) * x_last

                gamma_h = torch.exp(-torch.relu(self.hidden_decay(d_t)))
                h = gamma_h * h
            else:
                x_hat = x_t

            recurrent_input = torch.cat([x_hat, m_t, d_t, c_t], dim=-1)
            projected = self.input_projection(recurrent_input)
            h = self.gru_cell(projected, h)
            hidden_states.append(h.unsqueeze(1))

        sequence = torch.cat(hidden_states, dim=1)
        sequence = sequence + self.pos_emb[:, :n_steps, :]
        sequence = self.transformer(sequence)

        attention_scores = self.attention(sequence).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        h_att = torch.sum(sequence * attention_weights.unsqueeze(-1), dim=1)
        h_last = sequence[:, -1, :]
        h_max = torch.max(sequence, dim=1).values
        patient_embedding = self.output_projection(torch.cat([h_att, h_last, h_max], dim=-1))

        return {
            "patient_embedding": patient_embedding,
            "sequence_embedding": sequence,
            "attention_weights": attention_weights,
        }


class StaticEncoder(nn.Module):
    """MLP encoder for static demographics and prior-diagnosis features."""

    def __init__(self, n_features: int, hidden_dim: int = 64, dropout: float = 0.10) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_static: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_static)


class AggregateEncoder(nn.Module):
    """MLP encoder for engineered first-6-hour aggregate features."""

    def __init__(
        self,
        n_features: int,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 96,
        dropout_1: float = 0.20,
        dropout_2: float = 0.15,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim_1),
            nn.LayerNorm(hidden_dim_1),
            nn.GELU(),
            nn.Dropout(dropout_1),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.LayerNorm(hidden_dim_2),
            nn.GELU(),
            nn.Dropout(dropout_2),
        )

    def forward(self, x_aggregate: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_aggregate)


class GatedFusion(nn.Module):
    """Patient-specific gated fusion over temporal, static, and aggregate embeddings."""

    def __init__(
        self,
        temporal_dim: int = 128,
        static_dim: int = 64,
        aggregate_dim: int = 96,
        fusion_dim: int = 128,
        use_gate: bool = True,
    ) -> None:
        super().__init__()
        self.use_gate = bool(use_gate)
        self.static_proj = nn.Linear(static_dim, fusion_dim)
        self.aggregate_proj = nn.Linear(aggregate_dim, fusion_dim)
        self.gate = nn.Sequential(
            nn.Linear(3 * fusion_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )
        if temporal_dim != fusion_dim:
            self.temporal_proj = nn.Linear(temporal_dim, fusion_dim)
        else:
            self.temporal_proj = nn.Identity()

    @property
    def output_dim(self) -> int:
        fusion_dim = self.static_proj.out_features
        return 4 * fusion_dim if self.use_gate else 3 * fusion_dim

    def forward(
        self,
        h_temporal: torch.Tensor,
        h_static: torch.Tensor,
        h_aggregate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_temporal = self.temporal_proj(h_temporal)
        z_static = self.static_proj(h_static)
        z_aggregate = self.aggregate_proj(h_aggregate)
        z_cat = torch.cat([z_temporal, z_static, z_aggregate], dim=-1)
        if not self.use_gate:
            gate_weights = z_temporal.new_full((z_temporal.shape[0], 3), 1.0 / 3.0)
            return z_cat, gate_weights
        gate_weights = torch.softmax(self.gate(z_cat), dim=-1)
        z_gate = (
            gate_weights[:, 0:1] * z_temporal
            + gate_weights[:, 1:2] * z_static
            + gate_weights[:, 2:3] * z_aggregate
        )
        return torch.cat([z_gate, z_cat], dim=-1), gate_weights


class MortalityClassifier(nn.Module):
    """Binary mortality logit head."""

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        dropout_1: float = 0.25,
        dropout_2: float = 0.15,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim_1),
            nn.GELU(),
            nn.Dropout(dropout_1),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.GELU(),
            nn.Dropout(dropout_2),
            nn.Linear(hidden_dim_2, 1),
        )

    def forward(self, z_fused: torch.Tensor) -> torch.Tensor:
        return self.classifier(z_fused).squeeze(-1)


class TemporalReconstructionHead(nn.Module):
    """Predict masked standardized temporal values."""

    def __init__(self, hidden_dim: int, n_channels: int) -> None:
        super().__init__()
        self.head = nn.Linear(hidden_dim, n_channels)

    def forward(self, sequence_embedding: torch.Tensor) -> torch.Tensor:
        return self.head(sequence_embedding)


class MeasurementForecastHead(nn.Module):
    """Predict whether each channel is measured in the next bin."""

    def __init__(self, hidden_dim: int, n_channels: int) -> None:
        super().__init__()
        self.head = nn.Linear(hidden_dim, n_channels)

    def forward(self, sequence_embedding: torch.Tensor) -> torch.Tensor:
        return self.head(sequence_embedding[:, :-1, :])


class MAFNet(nn.Module):
    """Missingness-Aware Fusion Network for first-window ICU mortality modeling."""

    def __init__(
        self,
        n_temporal_channels: int,
        n_static_features: int,
        n_aggregate_features: int,
        hidden_dim: int = 128,
        n_time_steps: int = 24,
        n_heads: int = 4,
        transformer_ff_dim: int = 256,
        transformer_layers: int = 1,
        temporal_dropout: float = 0.15,
        use_static_branch: bool = True,
        use_aggregate_branch: bool = True,
        use_decay: bool = True,
        use_transformer: bool = True,
        use_gated_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.use_static_branch = bool(use_static_branch)
        self.use_aggregate_branch = bool(use_aggregate_branch)
        self.temporal_encoder = MissingnessAwareTemporalEncoder(
            n_channels=n_temporal_channels,
            hidden_dim=hidden_dim,
            n_time_steps=n_time_steps,
            n_heads=n_heads,
            transformer_ff_dim=transformer_ff_dim,
            dropout=temporal_dropout,
            transformer_layers=transformer_layers,
            use_decay=use_decay,
            use_transformer=use_transformer,
        )
        self.static_encoder = StaticEncoder(n_static_features)
        self.aggregate_encoder = AggregateEncoder(n_aggregate_features)
        self.fusion = GatedFusion(
            temporal_dim=hidden_dim,
            static_dim=64,
            aggregate_dim=96,
            fusion_dim=hidden_dim,
            use_gate=use_gated_fusion,
        )
        self.classifier = MortalityClassifier(input_dim=self.fusion.output_dim)
        self.reconstruction_head = TemporalReconstructionHead(hidden_dim, n_temporal_channels)
        self.measurement_forecast_head = MeasurementForecastHead(hidden_dim, n_temporal_channels)

    def forward(
        self,
        x_temporal: torch.Tensor,
        mask_temporal: torch.Tensor,
        delta_temporal: torch.Tensor,
        count_temporal: torch.Tensor,
        x_static: torch.Tensor,
        x_aggregate: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        temporal_out = self.temporal_encoder(
            x_temporal,
            mask_temporal,
            delta_temporal,
            count_temporal,
        )
        h_temporal = temporal_out["patient_embedding"]
        h_sequence = temporal_out["sequence_embedding"]
        batch_size = h_temporal.shape[0]
        h_static = (
            self.static_encoder(x_static)
            if self.use_static_branch
            else h_temporal.new_zeros(batch_size, 64)
        )
        h_aggregate = (
            self.aggregate_encoder(x_aggregate)
            if self.use_aggregate_branch
            else h_temporal.new_zeros(batch_size, 96)
        )
        z_fused, gate_weights = self.fusion(h_temporal, h_static, h_aggregate)
        mortality_logit = self.classifier(z_fused)

        return {
            "mortality_logit": mortality_logit,
            "x_recon": self.reconstruction_head(h_sequence),
            "mask_next_logit": self.measurement_forecast_head(h_sequence),
            "gate_weights": gate_weights,
            "temporal_attention": temporal_out["attention_weights"],
        }

    def predict_proba(
        self,
        x_temporal: torch.Tensor,
        mask_temporal: torch.Tensor,
        delta_temporal: torch.Tensor,
        count_temporal: torch.Tensor,
        x_static: torch.Tensor,
        x_aggregate: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.forward(
            x_temporal,
            mask_temporal,
            delta_temporal,
            count_temporal,
            x_static,
            x_aggregate,
        )
        return torch.sigmoid(outputs["mortality_logit"])


__all__ = [
    "AggregateEncoder",
    "GatedFusion",
    "MAFNet",
    "MeasurementForecastHead",
    "MissingnessAwareTemporalEncoder",
    "MortalityClassifier",
    "StaticEncoder",
    "TemporalReconstructionHead",
]
