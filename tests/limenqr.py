import unittest
import logging
import matplotlib.pyplot as plt
from quackseq.pulsesequence import QuackSequence
from quackseq.event import Event
from quackseq.functions import RectFunction, SincFunction
from quackseq_limenqr.limenqr import LimeNQR

logging.basicConfig(level=logging.DEBUG)

class TestQuackSequence(unittest.TestCase):

    def test_event_creation(self):
        seq = QuackSequence("test - simulation run sequence")

        loopback = Event("tx", "15u", seq)
        seq.add_event(loopback)
        seq.set_tx_amplitude(loopback, 1)
        seq.set_tx_phase(loopback, 0)

        rect = SincFunction()
        seq.set_tx_shape(loopback, rect)

        seq.set_rx(loopback, True)

        TR = Event("TR", "1m", seq)
        seq.add_event(TR)

        lime = LimeNQR()
        lime.set_averages(1000)
        lime.set_frequency(100e6)
        lime.settings.channel = "0"
        lime.settings.tx_gain = 30

        result = lime.run_sequence(seq)

        #plt.plot(result.tdx, abs(result.tdy))
        #plt.show()

if __name__ == "__main__":
    unittest.main()