"""Example of a simple SE sequence.

This sequence is a simple spin echo sequence with a 90 degree pulse followed by a 180 degree pulse.
"""

import logging
from quackseq.sequences.SE import create_SE
from quackseq_limenqr.limenqr import LimeNQR
from matplotlib import pyplot as plt

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    # Create a SE sequence
    seq = create_SE()

    events =seq.get_event_names()

    # Add a tr event
    seq.add_blank_event("tr", 1e-3)

    # Create a LimeNQR object
    lime = LimeNQR()

    # Set the number of averages
    lime.set_averages(1000)

    # Set a longer acquisition time
    lime.settings.acquisition_time = 830e-6

    # Set the frequency
    lime.set_frequency(83.56e6)

    # Run the sequence
    result = lime.run_sequence(seq)

    # Plot the results
    plt.plot(result.tdx, result.tdy[:, -1].imag)
    plt.plot(result.tdx, result.tdy[:, -1].real)
    plt.plot(result.tdx, abs(result.tdy[:, -1]))
    plt.show()