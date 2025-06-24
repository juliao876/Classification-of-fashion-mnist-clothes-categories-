import numpy as np

class EarlyStopping:
    def __init__(self, patience: int = 5, delta: float = 1e-4):
        self.patience  = patience
        self.delta     = delta
        self.best_loss = np.inf
        self.counter   = 0

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
