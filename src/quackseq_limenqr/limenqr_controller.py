"""Controller module for the Lime NQR spectrometer."""

import logging
from datetime import datetime
import tempfile
from pathlib import Path
import numpy as np
from scipy.signal import resample

from limedriver.binding import PyLimeConfig
from limedriver.hdf_reader import HDF

from nqrduck.helpers.unitconverter import UnitConverter
from quackseq.spectrometer.spectrometer_controller import SpectrometerController
from quackseq.measurement import Measurement
from quackseq.pulseparameters import TXPulse, RXReadout
from quackseq.pulsesequence import QuackSequence


logger = logging.getLogger(__name__)


class LimeNQRController(SpectrometerController):
    """Controller class for the Lime NQR spectrometer."""

    def __init__(self, limenqr):
        """Initializes the SimulatorController."""
        super().__init__()
        self.limenqr = limenqr

    def run_sequence(self, sequence: QuackSequence):
        """Starts the measurement procedure."""
        lime = self.initialize_lime(sequence)
        if lime is None:
            # Emit error message
            logger.error("Error initializing Lime driver")
            return -1
        elif lime.Npulses == 0:
            # Emit error message
            logger.error("No pulses in the pulse sequence")
            return -1

        self.setup_lime_parameters(lime, sequence)
        self.setup_temporary_storage(lime)

        if not self.perform_measurement(lime):
            self.emit_status_message("Measurement failed")
            self.emit_measurement_error(
                "Error with measurement data. Did you set an RX event?"
            )
            return -1

        measurement_data = self.process_measurement_results(lime, sequence)

        # Resample the RX data to the dwell time settings
        dwell_time = self.limenqr.model.settings.rx_dwell_time

        dwell_time = UnitConverter.to_float(dwell_time) * 1e6
        logger.debug("Dwell time: %s", dwell_time)
        logger.debug(f"Last tdx value: {measurement_data.tdx[-1]}")

        if dwell_time:
            n_data_points = int(measurement_data.tdx[-1] / dwell_time)
            logger.debug("Resampling to %s data points", n_data_points)
            tdx = np.linspace(
                0, measurement_data.tdx[-1], n_data_points, endpoint=False
            )
            tdy = resample(measurement_data.tdy, n_data_points)
            name = measurement_data.name
            measurement_data = Measurement(
                name,
                tdx,
                tdy,
                self.limenqr.model.target_frequency,
                IF_frequency=self.limenqr.model.if_frequency,
            )

        if measurement_data:
            return measurement_data

    def initialize_lime(self, sequence: QuackSequence) -> PyLimeConfig:
        """Initializes the limr object that is used to communicate with the pulseN driver.

        Returns:
            PyLimeConfig: The PyLimeConfig object that is used to communicate with the pulseN driver
        """
        try:
            n_pulses = self.get_number_of_pulses(sequence)
            lime = PyLimeConfig(n_pulses)
            return lime
        except ImportError as e:
            logger.error("Error while importing limr: %s", e)
        except Exception as e:
            logger.error("Error while initializing Lime driver: %s", e)
            import traceback

            traceback.print_exc()

        return None

    def setup_lime_parameters(
        self, lime: PyLimeConfig, sequence: QuackSequence
    ) -> None:
        """Sets the parameters of the lime config according to the settings set in the spectrometer module.

        Args:
            lime (PyLimeConfig): The PyLimeConfig object that is used to communicate with the pulseN driver
            sequence (QuackSequence): The pulse sequence to run
        """
        # lime.noi = -1
        lime.override_init = -1
        #
        # lime.nrp = 1
        lime.repetitions = 1
        lime = self.update_settings(lime)
        lime = self.translate_pulse_sequence(lime, sequence)
        lime.averages = self.limenqr.model.averages
        self.log_lime_parameters(lime)

    def setup_temporary_storage(self, lime: PyLimeConfig) -> None:
        """Sets up the temporary storage for the measurement data.

        Args:
            lime (PyLimeConfig): The PyLimeConfig object that is used to communicate with the pulseN driver
        """
        temp_dir = tempfile.TemporaryDirectory()
        logger.debug("Created temporary directory at: %s", temp_dir.name)
        lime.save_path = str(Path(temp_dir.name)) + "/"  # Temporary storage path
        lime.file_pattern = "temp"  # Temporary filename prefix or related config

    def perform_measurement(self, lime: PyLimeConfig) -> bool:
        """Executes the measurement procedure.

        Args:
            lime (PyLimeConfig): The PyLimeConfig object that is used to communicate with the pulseN driver

        Returns:
            bool: True if the measurement was successful, False otherwise
        """
        logger.debug("Running the measurement procedure")
        try:
            lime.run()
            return True
        except Exception as e:
            logger.error("Failed to execute the measurement: %s", e)
            return False

    def process_measurement_results(
        self, lime: PyLimeConfig, sequence: QuackSequence
    ) -> Measurement:
        """Processes the measurement results and returns a Measurement object.

        Args:
            lime (PyLimeConfig): The PyLimeConfig object that is used to communicate with the pulseN driver
            sequence (QuackSequence): The pulse sequence to run

        Returns:
            Measurement: The measurement data
        """
        rx_begin, rx_stop = self.translate_rx_event(lime, sequence)
        if rx_begin is None or rx_stop is None:
            # Instead print the whole acquisition range
            rx_begin = 0
            rx_stop = lime.rectime_secs * 1e6

        logger.debug("RX event begins at: %sµs and ends at: %sµs", rx_begin, rx_stop)
        return self.calculate_measurement_data(lime, rx_begin, rx_stop)

    def calculate_measurement_data(
        self, lime: PyLimeConfig, rx_begin: float, rx_stop: float
    ) -> Measurement:
        """Calculates the measurement data from the limr object.

        Args:
            lime (PyLimeConfig): The PyLimeConfig object that is used to communicate with the pulseN driver
            rx_begin (float): The start time of the RX event in µs
            rx_stop (float): The stop time of the RX event in µs

        Returns:
            Measurement: The measurement data
        """
        try:
            path = lime.get_path()
            hdf = HDF(path)
            evidx = self.find_evaluation_range_indices(hdf, rx_begin, rx_stop)
            tdx, tdy = self.extract_measurement_data(lime, hdf, evidx)
            fft_shift = self.get_fft_shift()
            # Measurement name date + module + target frequency + averages + sequence name
            name = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - LimeNQR - {self.limenqr.model.target_frequency / 1e6} MHz - {self.limenqr.model.averages} averages"
            logger.debug(f"Measurement name: {name}")
            return Measurement(
                name,
                tdx,
                tdy,
                self.limenqr.model.target_frequency,
                frequency_shift=fft_shift,
                IF_frequency=self.limenqr.model.if_frequency,
            )
        except Exception as e:
            logger.error("Error processing measurement result: %s", e)
            return None

    def find_evaluation_range_indices(
        self, hdf: HDF, rx_begin: float, rx_stop: float
    ) -> list:
        """Finds the indices of the evaluation range in the measurement data.

        Args:
            hdf (HDF): The HDF object that is used to read the measurement data
            rx_begin (float): The start time of the RX event in µs
            rx_stop (float): The stop time of the RX event in µs

        Returns:
            list: The indices of the evaluation range in the measurement data
        """
        return np.where((hdf.tdx > rx_begin) & (hdf.tdx < rx_stop))[0]

    def extract_measurement_data(
        self, lime: PyLimeConfig, hdf: HDF, indices: list
    ) -> tuple:
        """Extracts the measurement data from the PyLimeConfig object.

        Args:
            lime (PyLimeConfig): The PyLimeConfig object that is used to communicate with the pulseN driver
            hdf (HDF): The HDF object that is used to read the measurement data
            indices (list): The indices of the evaluation range in the measurement data

        Returns:
            tuple: A tuple containing the time vector and the measurement data
        """
        tdx = hdf.tdx[indices] - hdf.tdx[indices][0]
        tdy = hdf.tdy[indices] / lime.averages
        # flatten the  tdy array
        tdy = tdy.flatten()
        return tdx, tdy

    def get_fft_shift(self) -> int:
        """Rreturns the FFT shift value from the settings.

        Returns:
            int: The FFT shift value
        """
        fft_shift_enabled = self.limenqr.model.settings.fft_shift

        return self.limenqr.model.if_frequency if fft_shift_enabled else 0

    def log_lime_parameters(self, lime: PyLimeConfig) -> None:
        """Logs the parameters of the limr object.

        Args:
            lime (PyLimeConfig): The PyLimeConfig object that is used to communicate with the pulseN driver
        """
        # for key, value in lime.__dict__.items():
        # logger.debug("Lime parameter %s has value %s", key, value)
        logger.debug("Lime parameter %s has value %s", "srate", lime.srate)

    def update_settings(self, lime: PyLimeConfig) -> PyLimeConfig:
        """Sets the parameters of the limr object according to the settings set in the spectrometer module.

        Args:
            lime (PyLimeConfig): The PyLimeConfig object that is used to communicate with the pulseN driver

        Returns:
            lime: The updated limr object
        """
        c3_tim = [0, 0, 0, 0]

        lime.srate = float(self.limenqr.model.settings.sampling_frequency)
        lime.channel = int(self.limenqr.model.settings.channel)
        lime.TX_matching = int(self.limenqr.model.settings.tx_matching)
        lime.RX_matching = int(self.limenqr.model.settings.rx_matching)
        lime.frq = self.limenqr.model.target_frequency - self.limenqr.model.if_frequency
        lime.rectime_secs = float(self.limenqr.model.settings.acquisition_time)

        c3_tim[0] = self.limenqr.model.settings.gate_enable
        c3_tim[1] = self.limenqr.model.settings.gate_padding_left
        c3_tim[2] = self.limenqr.model.settings.gate_shift
        c3_tim[3] = self.limenqr.model.settings.gate_padding_right

        lime.TX_gain = self.limenqr.model.settings.tx_gain
        lime.RX_gain = self.limenqr.model.settings.rx_gain
        lime.RX_LPF = self.limenqr.model.settings.rx_lpf_bw
        lime.TX_LPF = self.limenqr.model.settings.tx_lpf_bw
        lime.TX_IcorrDC = self.limenqr.model.settings.tx_i_dc_correction
        lime.TX_QcorrDC = self.limenqr.model.settings.tx_q_dc_correction
        lime.TX_IcorrGain = self.limenqr.model.settings.tx_i_gain_correction
        lime.TX_QcorrGain = self.limenqr.model.settings.tx_q_gain_correction
        lime.TX_IQcorrPhase = self.limenqr.model.settings.tx_phase_adjustment
        lime.RX_IcorrGain = self.limenqr.model.settings.rx_i_gain_correction
        lime.RX_QcorrGain = self.limenqr.model.settings.rx_q_gain_correction
        lime.RX_IQcorrPhase = self.limenqr.model.settings.rx_phase_adjustment

        # This needs to be  done in one go, otherwise not all values will be updated
        lime.c3_tim = c3_tim
        return lime

    def translate_pulse_sequence(
        self, lime: PyLimeConfig, sequence: QuackSequence
    ) -> PyLimeConfig:
        """Translates the pulse sequence to the limr object.

        Args:
            lime (PyLimeConfig): The PyLimeConfig object that is used to communicate with the pulseN driver
            sequence (QuackSequence): The pulse sequence to run
        """
        events = sequence.events

        first_pulse = True

        for event in events:
            self.log_event_details(event)
            for parameter in event.parameters.values():
                self.log_parameter_details(parameter)

                if self.is_translatable_tx_parameter(parameter, sequence):
                    pulse_shape, pulse_amplitude = self.prepare_pulse_amplitude(
                        event, parameter
                    )
                    pulse_amplitude, modulated_phase = self.modulate_pulse_amplitude(
                        pulse_amplitude, event, lime
                    )

                    if first_pulse:  # If the pulse frequency list is empty
                        pfr, pdr, pam, pof, pph = self.initialize_pulse_lists(
                            lime, pulse_amplitude, pulse_shape, modulated_phase
                        )
                        first_pulse = False
                    else:
                        pfr_ext, pdr_ext, pam_ext, pph_ext = self.extend_pulse_lists(
                            pulse_amplitude, pulse_shape, modulated_phase
                        )
                        pof_ext = self.calculate_and_set_offsets(
                            lime, pulse_shape, events, event, pulse_amplitude
                        )

                        pfr.extend(pfr_ext)
                        pdr.extend(pdr_ext)
                        pam.extend(pam_ext)
                        pof.extend(pof_ext)
                        pph.extend(pph_ext)

        lime.p_frq = pfr
        lime.p_dur = pdr
        lime.p_amp = pam
        lime.p_offs = pof
        lime.p_pha = pph
        # Set repetition time event as last event's duration and update number of pulses
        lime.reptime_secs = float(event.duration)
        lime.Npulses = len(lime.p_frq)
        return lime

    def get_number_of_pulses(self, sequence: QuackSequence) -> int:
        """Calculates the number of pulses in the pulse sequence before the LimeDriverBinding is initialized.

        This makes sure it"s initialized with the correct size of the pulse lists.

        Returns:
            int: The number of pulses in the pulse sequence
        """
        events = sequence.events
        num_pulses = 0
        for event in events:
            for parameter in event.parameters.values():
                if self.is_translatable_tx_parameter(parameter, sequence):
                    _, pulse_amplitude = self.prepare_pulse_amplitude(event, parameter)
                    num_pulses += len(pulse_amplitude)
                    logger.debug("Number of pulses: %s", num_pulses)

        return num_pulses

    def log_event_details(self, event) -> None:
        """Logs the details of an event."""
        logger.debug("Event %s has parameters: %s", event.name, event.parameters)

    def log_parameter_details(self, parameter) -> None:
        """Logs the details of a parameter."""
        logger.debug("Parameter %s has options: %s", parameter.name, parameter.options)

    def is_translatable_tx_parameter(self, parameter, sequence: QuackSequence) -> bool:
        """Checks if a parameter a pulse with a transmit pulse shape (amplitude nonzero).

        Args:
            parameter (Parameter): The parameter to check
            sequence (QuackSequence): The pulse sequence to run
        """
        return (
            parameter.name == sequence.TX_PULSE
            and parameter.get_option_by_name(TXPulse.RELATIVE_AMPLITUDE).value > 0
        )

    def prepare_pulse_amplitude(self, event, parameter):
        """Prepares the pulse amplitude for the limr object.

        Args:
            event (Event): The event that contains the parameter
            parameter (Parameter): The parameter that contains the pulse shape and amplitude

        Returns:
        tuple: A tuple containing the pulse shape and the pulse amplitude
        """
        pulse_shape = parameter.get_option_by_name(TXPulse.TX_PULSE_SHAPE).value
        pulse_amplitude = abs(pulse_shape.get_pulse_amplitude(event.duration)) * (
            parameter.get_option_by_name(TXPulse.RELATIVE_AMPLITUDE).value / 100
        )
        pulse_amplitude = np.clip(pulse_amplitude, -0.99, 0.99)

        return pulse_shape, pulse_amplitude

    def modulate_pulse_amplitude(
        self, pulse_amplitude: float, event, lime: PyLimeConfig
    ) -> tuple:
        """Modulates the pulse amplitude for the limr object. We need to do this to have the pulse at IF frequency instead  of LO frequency.

        Args:
            pulse_amplitude (float): The pulse amplitude
            event (Event): The event that contains the parameter
            lime (PyLimeConfig) : The PyLimeConfig object that is used to communicate with the pulseN driver

        Returns:
            tuple: A tuple containing the modulated pulse amplitude and the modulated phase
        """
        # num_samples = int(float(event.duration) * lime.sra)
        num_samples = int(float(event.duration) * lime.srate)
        tdx = np.linspace(0, float(event.duration), num_samples, endpoint=False)
        shift_signal = np.exp(1j * 2 * np.pi * self.limenqr.model.if_frequency * tdx)

        # The pulse amplitude needs to be resampled to the number of samples
        logger.debug("Resampling pulse amplitude to %s samples", num_samples)
        pulse_amplitude = resample(pulse_amplitude, num_samples)

        pulse_complex = pulse_amplitude * shift_signal
        modulated_amplitude = np.abs(pulse_complex)
        modulated_phase = self.unwrap_phase(np.angle(pulse_complex))
        return modulated_amplitude, modulated_phase

    def unwrap_phase(self, phase: float) -> float:
        """This method unwraps the phase of the pulse.

        Args:
            phase (float): The phase of the pulse
        """
        return (np.unwrap(phase) + 2 * np.pi) % (2 * np.pi)

    def initialize_pulse_lists(
        self,
        lime: PyLimeConfig,
        pulse_amplitude: np.array,
        pulse_shape,
        modulated_phase: np.array,
    ) -> tuple:
        """This method initializes the pulse lists of the limr object.

        Args:
            lime (PyLimeConfig): The PyLimeConfig object that is used to communicate with the pulseN driver
            pulse_amplitude (np.array): The pulse amplitude
            pulse_shape (Function): The pulse shape
            modulated_phase (np.array): The modulated phase
        """
        pfr = [float(self.limenqr.model.if_frequency)] * len(pulse_amplitude)
        # We set the first  len(pulse_amplitude) of the p_dur
        pdr = [float(pulse_shape.resolution)] * len(pulse_amplitude)
        pam = list(pulse_amplitude)
        pof = [self.limenqr.model.OFFSET_FIRST_PULSE] + [
            int(pulse_shape.resolution * lime.srate)
        ] * (len(pulse_amplitude) - 1)
        pph = list(modulated_phase)

        return pfr, pdr, pam, pof, pph

    def extend_pulse_lists(self, pulse_amplitude, pulse_shape, modulated_phase):
        """This method extends the pulse lists of the limr object.

        Args:
            pulse_amplitude (float): The pulse amplitude
            pulse_shape (PulseShape): The pulse shape
            modulated_phase (float): The modulated phase

        Returns:
            tuple: A tuple containing the extended pulse lists (frequency, duration, amplitude, phase)
        """
        pfr = [float(self.limenqr.model.if_frequency)] * len(pulse_amplitude)
        pdr = [float(pulse_shape.resolution)] * len(pulse_amplitude)
        pam = list(pulse_amplitude)
        pph = list(modulated_phase)

        return pfr, pdr, pam, pph

    def calculate_and_set_offsets(
        self, lime: PyLimeConfig, pulse_shape, events, current_event, pulse_amplitude
    ) -> list:
        """This method calculates and sets the offsets for the limr object.

        Args:
            lime (PyLimeConfig): The PyLimeConfig object that is used to communicate with the pulseN driver
            pulse_shape (Function): The pulse shape
            events (list): The pulse sequence events
            current_event (Event): The current event
            pulse_amplitude (float): The pulse amplitude

        Returns:
            list: The offsets for the limr object
        """
        blank_durations = self.get_blank_durations_before_event(events, current_event)

        # Calculate the total time that has passed before the current event
        total_blank_duration = sum(blank_durations)
        # Calculate the offset for the current pulse
        # The first pulse offset is already set, so calculate subsequent ones
        offset_for_current_pulse = int(np.ceil(total_blank_duration * lime.srate))

        # Offset for the current pulse should be added only once
        pof = [(offset_for_current_pulse)]

        # Set the offset for the remaining samples of the current pulse (excluding the first sample)
        # We subtract 1 because we have already set the offset for the current pulse's first sample
        offset_per_sample = int(float(pulse_shape.resolution) * lime.srate)
        pof.extend([offset_per_sample] * (len(pulse_amplitude) - 1))
        return pof

    # This method could be refactored in a potential pulse sequence module
    def get_blank_durations_before_event(self, events, current_event) -> list:
        """This method returns the blank durations before the current event.

        Args:
            events (list): The pulse sequence events
            current_event (Event): The current event

        Returns:
            list: The blank durations before the current event
        """
        blank_durations = []
        previous_events_without_tx_pulse = self.get_previous_events_without_tx_pulse(
            events, current_event
        )
        for event in previous_events_without_tx_pulse:
            blank_durations.append(float(event.duration))
        return blank_durations

    # This method could be refactored in a potential pulse sequence module
    def get_previous_events_without_tx_pulse(self, events : list, current_event : "Event") -> list:
        """This method returns the previous events without a transmit pulse.

        Args:
            events (list): The pulse sequence events
            current_event (Event): The current event

        Returns:
            list: The previous events without a transmit pulse
        """
        index = events.index(current_event)
        previous_events = events[:index]
        result = []
        for event in reversed(previous_events):
            translatable = any(
                self.is_translatable_tx_parameter(param)
                for param in event.parameters.values()
            )
            if not translatable:
                result.append(event)
            else:
                break
        return reversed(
            result
        )  # Reversed to maintain the original order if needed elsewhere

    def translate_rx_event(self, lime: PyLimeConfig, sequence: QuackSequence) -> tuple:
        """This method translates the RX event of the pulse sequence to the limr object.

        Args:
            lime (PyLimeConfig): The PyLimeConfig object that is used to communicate with the pulseN driver
            sequence (QuackSequence): The pulse sequence to run

        Returns:
            tuple: A tuple containing the start and stop time of the RX event in µs
        """
        CORRECTION_FACTOR = self.limenqr.model.settings.rx_offset

        events = sequence.events

        rx_event = self.find_rx_event(events)
        if not rx_event:
            return None, None

        previous_events_duration = self.calculate_previous_events_duration(
            events, rx_event
        )
        rx_duration = float(rx_event.duration)

        offset = self.calculate_offset(lime)

        rx_begin = (
            float(previous_events_duration) + float(offset) + float(CORRECTION_FACTOR)
        )
        rx_stop = rx_begin + rx_duration
        return rx_begin * 1e6, rx_stop * 1e6

    def find_rx_event(self, events):
        """This method finds the RX event in the pulse sequence.

        Args:
            events (list): The pulse sequence events

        Returns:
            Event: The RX event
        """
        for event in events:
            parameter = event.parameters.get(RXReadout.RX)
            if parameter and parameter.get_option_by_name(RXReadout.RX).value:
                self.log_event_details(event)
                self.log_parameter_details(parameter)
                return event
        return None

    def calculate_previous_events_duration(self, events, rx_event):
        """This method calculates the duration of the previous events.

        Args:
            events (list): The pulse sequence events
            rx_event (Event): The RX event

        Returns:
            float: The duration of the previous events
        """
        previous_events = events[: events.index(rx_event)]
        return sum(event.duration for event in previous_events)

    def calculate_offset(self, lime: PyLimeConfig) -> float:
        """This method calculates the offset for the RX event.

        Args:
            lime (limr): The limr object that is used to communicate with the pulseN driver

        Returns:
            float: The offset for the RX event
        """
        return self.limenqr.model.OFFSET_FIRST_PULSE * (1 / lime.srate)
