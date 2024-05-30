from quackseq.spectrometer.spectrometer import Spectrometer

from .limenqr_model import LimeNQRModel
from .limenqr_controller import LimeNQRController


class LimeNQR(Spectrometer):
    def __init__(self):
        self.model = LimeNQRModel()
        self.controller = LimeNQRController(self)

    def run_sequence(self, sequence):
        result = self.controller.run_sequence(sequence)
        return result

    def set_averages(self, value: int):
        self.model.averages = value

    def set_frequency(self, value: float):
        self.model.target_frequency = value

    @property
    def settings(self):
        return self.model.settings
