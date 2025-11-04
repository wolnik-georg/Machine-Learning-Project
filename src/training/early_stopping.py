class EarlyStopping:
    """
    Early stopping utility to stop training when a monitored metric has stopped improving.
    Args:
        patience (int): How many epochs to wait after last time the monitored metric improved.
        min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
        mode (str): 'min' for loss, 'max' for accuracy, etc.
    """
    def __init__(self, patience=5, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False

        if (self.mode == 'min' and current_score < self.best_score - self.min_delta) or \
           (self.mode == 'max' and current_score > self.best_score + self.min_delta):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
