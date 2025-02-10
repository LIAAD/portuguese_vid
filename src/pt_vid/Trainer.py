from pt_vid.trainer.Strategy import Strategy

class Trainer:
    def __init__(self, training_strategy:Strategy):
        self.training_strategy = training_strategy
    
    def train(self):
        return self.training_strategy.train()