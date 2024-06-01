"""LimeNQR implementation for quackseq."""

from quackseq.spectrometer.spectrometer import Spectrometer
from quackseq.pulsesequence import QuackSequence
from quackseq.measurement import Measurement
from quackseq.spectrometer.spectrometer_model import QuackSettings

from .limenqr_model import LimeNQRModel
from .limenqr_controller import LimeNQRController


class LimeNQR(Spectrometer):
    """The LimeNQR spectrometer implementation for quackseq.

    Attributes:
        model (LimeNQRModel) : The model of the LimeNQR spectrometer - contains the settings
        controller (LimeNQRController) : The controller of the LimeNQR spectrometer - contains the logic on how a sequence is interpreted and run
    """

    def __init__(self):
        """Initializes the LimeNQR spectrometer."""
        self.model = LimeNQRModel()
        self.controller = LimeNQRController(self)

    def run_sequence(self, sequence: QuackSequence) -> Measurement:
        """Runs a sequence on the LimeNQR spectrometer.

        Args:
            sequence (QuackSequence) : The sequence to run

        Returns:
            Measurement : The measurement data
        """
        result = self.controller.run_sequence(sequence)
        return result

    def set_averages(self, value: int) -> None:
        """Sets the number of averages.

        Args:
            value (int) : The number of averages
        """
        self.model.averages = value

    def set_frequency(self, value: float) -> None:
        """Sets the frequency of the spectrometer.

        Args:
            value (float) : The frequency in Hz to set
        """
        self.model.target_frequency = value

    @property
    def settings(self) -> QuackSettings:
        """The settings of the LimeNQR spectrometer. Settings are defined in the LimeNQRModel."""
        return self.model.settings
