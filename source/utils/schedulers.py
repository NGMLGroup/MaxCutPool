class CosineScheduler():
    def __init__(self, g_max=1, g_min=0, epochs=100):
        self.g_max = g_max
        self.g_min = g_min
        self.epochs = epochs

    def __call__(self, epoch):
        gamma = self.g_min + 0.5 * (self.g_max - self.g_min) * (1 + torch.cos(epoch * 3.1415 / self.epochs))
        return gamma

class LinearScheduler():
    def __init__(self, g_max=1, g_min=0, epochs=100):
        self.g_max = g_max
        self.g_min = g_min
        self.epochs = epochs

    def __call__(self, epoch):
        gamma = self.g_max - (self.g_max - self.g_min) * epoch / self.epochs
        return gamma


class CoefficientScheduler(pl.Callback):
    def __init__(self, epochs, g_max=1, g_min=0):
        super().__init__()
        self.gamma_scheduler = CosineScheduler(g_max, g_min, epochs)

    def on_epoch_end(self, trainer, model):
        if model.pooling in ['maxcutpool']:
            gamma = self.gamma_scheduler(trainer.current_epoch)  # Calculate coefficients
            model.pool.gamma = gamma  # Update coefficients in model
            model.log('gamma', gamma) # Log coefficients