from abc import ABC, abstractmethod

class ModelBase(ABC):

    @abstractmethod
    def __init__(self, device):
        self.device = device
        # Add additional attributes for inference here

    @abstractmethod
    def predict(self, input_path) -> float:
        pass
        # Implement inference logic here
    