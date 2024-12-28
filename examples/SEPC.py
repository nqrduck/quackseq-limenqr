"""Spin Echo  with Phase Cycling (SEPC) with the limenqr spectrometer.

The intended sample is the default BiPh3 NQR sample.
"""

import logging

from quackseq.sequences.SEPC import create_SEPC
from quackseq_limenqr.limenqr import LimeNQR
from matplotlib import pyplot as plt

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    lime = LimeNQR()
    lime.set_averages(100)

    SEPC = create_SEPC()

    SEPC.add_blank_event("tr", 1e-3)

    lime.set_frequency(83.56e6)

    result = lime.run_sequence(SEPC)

    # Plot  the results
    plt.plot(result.tdx, result.tdy[:, -1].imag)
    plt.plot(result.tdx, result.tdy[:, -1].real)
    plt.plot(result.tdx, abs(result.tdy[:, -1]))
    plt.show()

