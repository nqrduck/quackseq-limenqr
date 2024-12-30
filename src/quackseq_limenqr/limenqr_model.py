"""Model for the Lime NQR spectrometer."""

import logging
from quackseq.spectrometer.spectrometer_model import SpectrometerModel
from quackseq.spectrometer.spectrometer_settings import (
    FloatSetting,
    IntSetting,
    BooleanSetting,
    SelectionSetting,
    StringSetting,
)

logger = logging.getLogger(__name__)


class LimeNQRModel(SpectrometerModel):
    """Model for the Lime NQR spectrometer."""

    # Setting constants for the names of the spectrometer settings
    CHANNEL = "TX/RX Channel"
    TX_MATCHING = "TX Matching"
    RX_MATCHING = "RX Matching"
    SAMPLING_FREQUENCY = "Sampling Frequency (Hz)"
    RX_DWELL_TIME = "RX Dwell Time (s)"
    IF_FREQUENCY = "IF Frequency (Hz)"
    ACQUISITION_TIME = "Acquisition time (s)"
    GATE_ENABLE = "Enable"
    GATE_PADDING_LEFT = "Gate padding left"
    GATE_PADDING_RIGHT = "Gate padding right"
    GATE_SHIFT = "Gate shift"
    RX_GAIN = "RX Gain"
    TX_GAIN = "TX Gain"
    RX_LPF_BW = "RX LPF BW (Hz)"
    TX_LPF_BW = "TX LPF BW (Hz)"
    TX_I_DC_CORRECTION = "TX I DC correction"
    TX_Q_DC_CORRECTION = "TX Q DC correction"
    TX_I_GAIN_CORRECTION = "TX I Gain correction"
    TX_Q_GAIN_CORRECTION = "TX Q Gain correction"
    TX_PHASE_ADJUSTMENT = "TX phase adjustment"
    RX_I_DC_CORRECTION = "RX I DC correction"
    RX_Q_DC_CORRECTION = "RX Q DC correction"
    RX_I_GAIN_CORRECTION = "RX I Gain correction"
    RX_Q_GAIN_CORRECTION = "RX Q Gain correction"
    RX_PHASE_ADJUSTMENT = "RX phase adjustment"
    RX_OFFSET = "RX offset"
    FFT_SHIFT = "FFT shift"

    # Constants for the Categories of the settings
    ACQUISITION = "Acquisition"
    GATE_SETTINGS = "Gate Settings"
    RX_TX_SETTINGS = "RX/TX Settings"
    CALIBRATION = "Calibration"
    SIGNAL_PROCESSING = "Signal Processing"

    # Pulse parameter constants
    TX = "TX"
    RX = "RX"

    # Settings that are not changed by the user
    OFFSET_FIRST_PULSE = 300

    def __init__(self) -> None:
        """Initializes the Lime NQR model."""
        super().__init__()
        # Acquisition settings
        channel_options = ["0", "1"]
        channel_setting = SelectionSetting(
            self.CHANNEL, self.ACQUISITION, channel_options, "0", "TX/RX Channel"
        )
        self.add_setting("channel", channel_setting)

        tx_matching_options = ["0", "1", "2", "3", "4"]
        tx_matching_setting = SelectionSetting(
            self.TX_MATCHING, self.ACQUISITION, tx_matching_options, "0", "TX Matching"
        )
        self.add_setting("tx_matching", tx_matching_setting)

        rx_matching_options = ["0", "1", "2", "3", "4"]
        rx_matching_setting = SelectionSetting(
            self.RX_MATCHING, self.ACQUISITION, rx_matching_options, "0", "RX Matching"
        )
        self.add_setting("rx_matching", rx_matching_setting)

        sampling_frequency_options = [
            "30.72e6",
            "15.36e6",
            "7.68e6",
        ]
        sampling_frequency_setting = SelectionSetting(
            self.SAMPLING_FREQUENCY,
            self.ACQUISITION,
            sampling_frequency_options,
            "30.72e6",
            "The rate at which the spectrometer samples the input signal.",
        )
        self.add_setting("sampling_frequency", sampling_frequency_setting)

        rx_dwell_time_setting = StringSetting(
            self.RX_DWELL_TIME,
            self.ACQUISITION,
            "32n",
            "The time between samples in the receive path.",
        )
        self.add_setting("rx_dwell_time", rx_dwell_time_setting)

        if_frequency_setting = FloatSetting(
            self.IF_FREQUENCY,
            self.ACQUISITION,
            5e6,
            "The intermediate frequency to which the input signal is down converted during analog-to-digital conversion.",
            min_value=0,
        )
        self.add_setting("if_frequency", if_frequency_setting)
        self.if_frequency = 5e6

        acquisition_time_setting = StringSetting(
            self.ACQUISITION_TIME,
            self.ACQUISITION,
            82e-6,
            "Acquisition time - this is from the beginning of the pulse sequence",
        )
        self.add_setting("acquisition_time", acquisition_time_setting)

        # Gate Settings
        gate_enable_setting = BooleanSetting(
            self.GATE_ENABLE,
            self.GATE_SETTINGS,
            True,
            "Setting that controls whether gate is on during transmitting.",
        )
        self.add_setting("gate_enable", gate_enable_setting)

        gate_padding_left_setting = IntSetting(
            self.GATE_PADDING_LEFT,
            self.GATE_SETTINGS,
            10,
            "The number of samples by which to extend the gate window to the left.",
            min_value=0,
        )
        self.add_setting("gate_padding_left", gate_padding_left_setting)

        gate_padding_right_setting = IntSetting(
            self.GATE_PADDING_RIGHT,
            self.GATE_SETTINGS,
            10,
            "The number of samples by which to extend the gate window to the right.",
            min_value=0,
        )
        self.add_setting("gate_padding_right", gate_padding_right_setting)

        gate_shift_setting = IntSetting(
            self.GATE_SHIFT,
            self.GATE_SETTINGS,
            53,
            "The delay, in number of samples, by which the gate window is shifted.",
            min_value=0,
        )
        self.add_setting("gate_shift", gate_shift_setting)

        # RX/TX settings
        rx_gain_setting = IntSetting(
            self.RX_GAIN,
            self.RX_TX_SETTINGS,
            55,
            "The gain level of the receiver’s amplifier.",
            min_value=0,
            max_value=55,
            slider=True,
        )
        self.add_setting("rx_gain", rx_gain_setting)

        tx_gain_setting = IntSetting(
            self.TX_GAIN,
            self.RX_TX_SETTINGS,
            30,
            "The gain level of the transmitter’s amplifier.",
            min_value=0,
            max_value=55,
            slider=True,
        )
        self.add_setting("tx_gain", tx_gain_setting)

        rx_lpf_bw_setting = FloatSetting(
            self.RX_LPF_BW,
            self.RX_TX_SETTINGS,
            30.72e6 / 2,
            "The bandwidth of the receiver’s low-pass filter which attenuates frequencies below a certain threshold.",
        )
        self.add_setting("rx_lpf_bw", rx_lpf_bw_setting)

        tx_lpf_bw_setting = FloatSetting(
            self.TX_LPF_BW,
            self.RX_TX_SETTINGS,
            130.0e6,
            "The bandwidth of the transmitter’s low-pass filter which limits the frequency range of the transmitted signal.",
        )
        self.add_setting("tx_lpf_bw", tx_lpf_bw_setting)

        # Calibration settings
        tx_i_dc_correction_setting = IntSetting(
            self.TX_I_DC_CORRECTION,
            self.CALIBRATION,
            -45,
            "Adjusts the direct current offset errors in the in-phase (I) component of the transmit (TX) path.",
            min_value=-128,
            max_value=127,
            slider=True,
        )
        self.add_setting("tx_i_dc_correction", tx_i_dc_correction_setting)

        tx_q_dc_correction_setting = IntSetting(
            self.TX_Q_DC_CORRECTION,
            self.CALIBRATION,
            0,
            "Adjusts the direct current offset errors in the quadrature (Q) component of the transmit (TX) path.",
            min_value=-128,
            max_value=127,
            slider=True,
        )
        self.add_setting("tx_q_dc_correction", tx_q_dc_correction_setting)

        tx_i_gain_correction_setting = IntSetting(
            self.TX_I_GAIN_CORRECTION,
            self.CALIBRATION,
            2047,
            "Modifies the gain settings for the I channel of the TX path, adjusting for imbalances.",
            min_value=0,
            max_value=2047,
            slider=True,
        )
        self.add_setting("tx_i_gain_correction", tx_i_gain_correction_setting)

        tx_q_gain_correction_setting = IntSetting(
            self.TX_Q_GAIN_CORRECTION,
            self.CALIBRATION,
            2039,
            "Modifies the gain settings for the Q channel of the TX path, adjusting for imbalances.",
            min_value=0,
            max_value=2047,
            slider=True,
        )
        self.add_setting("tx_q_gain_correction", tx_q_gain_correction_setting)

        tx_phase_adjustment_setting = IntSetting(
            self.TX_PHASE_ADJUSTMENT,
            self.CALIBRATION,
            3,
            "Corrects the Phase of I Q signals in the TX path.",
            min_value=-2048,
            max_value=2047,
            slider=True,
        )
        self.add_setting("tx_phase_adjustment", tx_phase_adjustment_setting)

        rx_i_dc_correction_setting = IntSetting(
            self.RX_I_DC_CORRECTION,
            self.CALIBRATION,
            0,
            "Adjusts the direct current offset errors in the in-phase (I) component of the receive (RX) path.",
            min_value=-63,
            max_value=63,
            slider=True,
        )
        self.add_setting("rx_i_dc_correction", rx_i_dc_correction_setting)

        rx_q_dc_correction_setting = IntSetting(
            self.RX_Q_DC_CORRECTION,
            self.CALIBRATION,
            0,
            "Adjusts the direct current offset errors in the quadrature (Q) component of the receive (RX) path.",
            min_value=-63,
            max_value=63,
            slider=True,
        )
        self.add_setting("rx_q_dc_correction", rx_q_dc_correction_setting)

        rx_i_gain_correction_setting = IntSetting(
            self.RX_I_GAIN_CORRECTION,
            self.CALIBRATION,
            2047,
            "Modifies the gain settings for the I channel of the RX path, adjusting for imbalances.",
            min_value=0,
            max_value=2047,
            slider=True,
        )
        self.add_setting("rx_i_gain_correction", rx_i_gain_correction_setting)

        rx_q_gain_correction_setting = IntSetting(
            self.RX_Q_GAIN_CORRECTION,
            self.CALIBRATION,
            2047,
            "Modifies the gain settings for the Q channel of the RX path, adjusting for imbalances.",
            min_value=0,
            max_value=2047,
            slider=True,
        )
        self.add_setting("rx_q_gain_correction", rx_q_gain_correction_setting)

        rx_phase_adjustment_setting = IntSetting(
            self.RX_PHASE_ADJUSTMENT,
            self.CALIBRATION,
            0,
            "Corrects the Phase of I Q signals in the RX path.",
            min_value=-2048,
            max_value=2047,
            slider=True,
        )
        self.add_setting("rx_phase_adjustment", rx_phase_adjustment_setting)

        # Signal Processing settings
        rx_offset_setting = StringSetting(
            self.RX_OFFSET,
            self.SIGNAL_PROCESSING,
            2.4e-6,
            "The offset of the RX event, this changes all the time",
        )
        self.add_setting("rx_offset", rx_offset_setting)

        fft_shift_setting = BooleanSetting(
            self.FFT_SHIFT, self.SIGNAL_PROCESSING, False, "FFT shift"
        )
        self.add_setting("fft_shift", fft_shift_setting)

        self.averages = 1
        self.target_frequency = 100e6

    @property
    def target_frequency(self) -> float:
        """The target frequency of the spectrometer."""
        return self._target_frequency

    @target_frequency.setter
    def target_frequency(self, value):
        self._target_frequency = value

    @property
    def averages(self) -> int:
        """The number of averages to be taken."""
        return self._averages

    @averages.setter
    def averages(self, value):
        self._averages = value

    @property
    def if_frequency(self) -> float:
        """The intermediate frequency to which the input signal is down converted during analog-to-digital conversion."""
        return self._if_frequency

    @if_frequency.setter
    def if_frequency(self, value):
        self._if_frequency = value
