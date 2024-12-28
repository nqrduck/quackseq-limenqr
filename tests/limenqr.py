"""Test for the LimeNQR quackseq implementation."""

import unittest
import logging
import matplotlib.pyplot as plt
import numpy as np
from quackseq.pulsesequence import QuackSequence
from quackseq.event import Event
from quackseq.functions import SincFunction, RectFunction
from quackseq_limenqr.limenqr import LimeNQR

logging.basicConfig(level=logging.DEBUG)

# Mute matplotlib logs
logging.getLogger("matplotlib").setLevel(logging.WARNING)


# Loopback sequence

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

        sinc = SincFunction()
        seq.set_tx_shape(loopback, sinc)

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

        plt.plot(result.tdx, result.tdy.imag)
        plt.plot(result.tdx, result.tdy.real)
        plt.plot(result.tdx, np.abs(result.tdy))
        plt.show()

    def test_loopback_phase_cycling(self):
        """Tests a loopback sequence with phase cycling."""
        seq = QuackSequence("test - simulation run sequence")

        loopback = Event("tx", "3u", seq)
        seq.add_event(loopback)
        seq.set_tx_amplitude(loopback, 100)
        seq.set_tx_phase(loopback, 0)

        seq.set_tx_n_phase_cycles(loopback, 2)

        rect = RectFunction()
        seq.set_tx_shape(loopback, rect)

        #readout_scheme = [180, 0]
        #seq.set_rx_phase(loopback, readout_scheme)

        seq.set_rx(loopback, True)


        TR = Event("TR", "1m", seq)
        seq.add_event(TR)

        print(seq.to_json())

        lime = LimeNQR()
        lime.set_averages(100)
        lime.set_frequency(100e6)
        lime.settings.channel = "0"
        lime.settings.tx_gain = 30

        result = lime.run_sequence(seq)

        plt.plot(result.tdx, result.tdy[:, 0].real)
        plt.plot(result.tdx, result.tdy[:, 1].real)
        plt.plot(result.tdx, result.tdy[:, 2])
        plt.show()



if __name__ == "__main__":
    unittest.main()
