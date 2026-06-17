"""Small training callbacks for neural ICU models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """Track patience for a validation metric."""

    mode: str = "max"
    patience: int = 20
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.best_score: float | None = None
        self.num_bad_epochs = 0

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = float(score)
            self.num_bad_epochs = 0
            return False

        improved = (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        )
        if improved:
            self.best_score = float(score)
            self.num_bad_epochs = 0
            return False
        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience


__all__ = ["EarlyStopping"]
