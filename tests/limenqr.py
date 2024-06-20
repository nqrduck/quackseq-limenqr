"""Test for the LimeNQR quackseq implementation."""

import unittest
import logging
import matplotlib.pyplot as plt
import numpy as np
from quackseq.pulsesequence import QuackSequence
from quackseq.event import Event
from quackseq.functions import SincFunction
from quackseq_limenqr.limenqr import LimeNQR

logging.basicConfig(level=logging.DEBUG)

# Mute matplotlib logs
logging.getLogger("matplotlib").setLevel(logging.WARNING)


class TestQuackSequence(unittest.TestCase):
    """Test the LimeNQR quackseq implementation."""
    def test_loopback(self):
        """Tests a loopback sequence."""
        # Loopback sequence
        seq = QuackSequence("test - simulation run sequence")

        loopback = Event("tx", "20u", seq)
        seq.add_event(loopback)
        seq.set_tx_amplitude(loopback, 100)
        seq.set_tx_phase(loopback, 0)

        rect = SincFunction()
        seq.set_tx_shape(loopback, rect)

        seq.set_rx(loopback, True)

        TR = Event("TR", "1m", seq)
        seq.add_event(TR)

        print(seq.to_json())

        lime = LimeNQR()
        lime.set_averages(1000)
        lime.set_frequency(100e6)
        lime.settings.channel = "0"
        lime.settings.tx_gain = 30

        result = lime.run_sequence(seq)

        plt.plot(result.tdx[-1], result.tdy[-1].imag)
        plt.plot(result.tdx[-1], result.tdy[-1].real)
        plt.plot(result.tdx[-1], np.abs(result.tdy[-1]))
        plt.show()


if __name__ == "__main__":
    unittest.main()
