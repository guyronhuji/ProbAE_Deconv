from __future__ import annotations


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 0.0) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.num_bad_epochs = 0

    def update(self, value: float) -> bool:
        if value < (self.best - self.min_delta):
            self.best = value
            self.num_bad_epochs = 0
            return False
        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience
