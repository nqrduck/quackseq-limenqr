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
from quackseq.measurement import Measurement, MeasurementError
from quackseq.pulseparameters import TXPulse, RXReadout
from quackseq.pulsesequence import QuackSequence


logger = logging.getLogger(__name__)


class LimeNQRController(SpectrometerController):
    """Controller class for the Lime NQR spectrometer."""

    def __init__(self, limenqr):
        """Initializes the SimulatorController."""
        super().__init__()
        self.limenqr = limenqr

    def run_sequence(self, sequence: QuackSequence) -> Measurement:
        """Starts the measurement procedure."""
        lime = self.initialize_lime(sequence)
        if lime is None:
            # Emit error message
            logger.error("Error initializing Lime driver")
            return self.return_measurement_error("Error initializing Lime driver")

        elif lime.Npulses == 0:
            # Emit error message
            logger.error("No pulses in the pulse sequence")
            return self.return_measurement_error("No pulses in the pulse sequence")

        self.setup_lime_parameters(lime, sequence)
        self.setup_temporary_storage(lime)

        if not self.perform_measurement(lime):
            return self.return_measurement_error(
                "Error performing measurement. Did you set an RX event?"
            )

        n_phase_cycles = sequence.get_n_phase_cycles()
        logger.debug("Expecting %s phase cycles", n_phase_cycles)

        # We get a list of measurements, one for each phase cycle
        measurement_data = self.process_measurement_results(lime, sequence)

        if isinstance(measurement_data, MeasurementError):
            logger.debug("Measurement error: %s", measurement_data.error_message)
            return measurement_data

        # Check if the measurement data is valid - the +1 comes from the summing of the phase cycles
        if n_phase_cycles > 1 and measurement_data.tdy.shape[1] != n_phase_cycles + 1:
            logger.debug(
                f"Expected {n_phase_cycles + 1} phase cycles, got {measurement_data.tdy.shape[1]}"
            )
            return self.return_measurement_error(
                "Error processing measurement data - the number of phase cycles does not match the expected number"
            )
        
        # We resample the data to the dwell time settings
        measurement_data = self.resample_data(measurement_data)

        logger.debug(f"Measurement data shape: {measurement_data.tdy.shape}")

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
        rx_begin, rx_stop, readout_scheme = self.translate_rx_event(lime, sequence)
        if rx_begin is None or rx_stop is None:
            # Instead print the whole acquisition range
            rx_begin = 0
            rx_stop = lime.rectime_secs * 1e6
            readout_scheme = sequence.get_n_phase_cycles() * [0]

        logger.debug("RX event begins at: %sµs and ends at: %sµs", rx_begin, rx_stop)
        return self.calculate_measurement_data(lime, rx_begin, rx_stop, readout_scheme)

    def return_measurement_error(self, message: str) -> Measurement:
        """In case of an error we return a measurement that has no data and the error message as the name.

        Args:
            message (str): The error message

        Returns:
            Measurement: The error message as the name
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error = MeasurementError(timestamp, message)
        return error

    def calculate_measurement_data(
        self, lime: PyLimeConfig, rx_begin: float, rx_stop: float, readout_scheme: list
    ) -> Measurement:
        """Calculates the measurement data from the limr object.

        Args:
            lime (PyLimeConfig): The PyLimeConfig object that is used to communicate with the pulseN driver
            rx_begin (float): The start time of the RX event in µs
            rx_stop (float): The stop time of the RX event in µs
            readout_scheme (list): The readout for phase cycling

        Returns:
            Measurement: The measurement data
        """
        try:
            path = lime.get_path()
            hdf = HDF(path)
            logger.debug(
                f"HDF file has size: hdf.tdx.shape = {hdf.tdx.shape} and hdf.tdy.shape = {hdf.tdy.shape}"
            )
            evidx = self.find_evaluation_range_indices(hdf, rx_begin, rx_stop)
            tdx, tdy = self.extract_measurement_data(lime, hdf, evidx)
            fft_shift = self.get_fft_shift()
            # Measurement name date + module + target frequency + averages + sequence name
            name = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - LimeNQR - {self.limenqr.model.target_frequency / 1e6} MHz - {self.limenqr.model.averages} averages"
            logger.debug(f"Measurement name: {name}")

            measurement = Measurement(
                name,
                tdx,
                tdy,
                self.limenqr.model.target_frequency,
                frequency_shift=fft_shift,
                IF_frequency=self.limenqr.model.if_frequency,
            )

            logger.debug(f"Readout scheme: {readout_scheme}")

            for i, readout_phase in enumerate(readout_scheme):
                logger.debug(f"Performing phase shift with phase: {readout_phase}")
                measurement.phase_shift(readout_phase, index=i)

            # Last measurement datapoint just sums up the phase cycles along the phase cycling dimension if the readout scheme is longer than one
            if len(readout_scheme) > 1:
                sum_tdy = np.sum(tdy, axis=1, keepdims=True)
                logger.debug(f"measurement tdy shape: {measurement.tdy.shape}")
                logger.debug(f"Summed tdy shape: {sum_tdy.shape}")
                measurement.add_dataset(sum_tdy)

            logger.debug(f"Measurement data tdy shape: {measurement.tdy.shape}")

            return measurement

        except Exception as e:
            return MeasurementError(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"Did you connect the LimeSDR? Error: {e}")

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
        # tdx is the same for all phase cycles
        tdx = hdf.tdx[indices] - hdf.tdx[indices][0]
        # tdy has the shape (points, n_phase_cycles)
        tdy = hdf.tdy[indices] / lime.averages

        logger.debug("tdy shape: %s", tdy.shape)
        return tdx, tdy

    def get_fft_shift(self) -> int:
        """Returns the FFT shift value from the settings.

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

        This  is the main method that translates the pulse sequence to the limr object. It iterates over the pulse sequence events and parameters and prepares the pulse lists for the limr object.

        Description of variables:
        pfr: Pulse frequency list
        pdr: Pulse duration list
        pam: Pulse amplitude list
        pof: Pulse offset list
        pph: Pulse phase list

        Phase cycling  variables:
        p_phacyc_N: Number of phase cycles
        p_phacyc_lev: The phase cycle level. This indicates  how the loops are nested. If a pulse event is on the same level they need to have the same number of phase cycles.

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
                    pulse_shape, pulse_amplitude, pulse_phase = (
                        self.prepare_pulse_amplitude(event, parameter)
                    )
                    pulse_amplitude, modulated_phase = self.modulate_pulse_amplitude(
                        pulse_amplitude, pulse_phase, event, lime
                    )

                    if first_pulse:  # If the pulse frequency list is empty
                        pfr, pdr, pam, pof, pph = self.initialize_pulse_lists(
                            lime, pulse_amplitude, pulse_shape, modulated_phase
                        )
                        first_pulse = False

                        # Get the phase cycling parameters
                        p_phacyc_N = parameter.get_option_by_name(
                            parameter.N_PHASE_CYCLES
                        )
                        p_phacyc_lev = parameter.get_option_by_name(
                            parameter.PHASE_CYCLE_GROUP
                        )

                        # Make the length of the phase cycling parameters equal to the length of the pulse amplitude
                        p_phacyc_N = [p_phacyc_N.value] * len(pulse_amplitude)
                        p_phacyc_lev = [p_phacyc_lev.value] * len(pulse_amplitude)

                    else:
                        pfr_ext, pdr_ext, pam_ext, pph_ext = self.extend_pulse_lists(
                            pulse_amplitude, pulse_shape, modulated_phase
                        )
                        pof_ext = self.calculate_and_set_offsets(
                            lime, pulse_shape, events, event, pulse_amplitude, sequence
                        )

                        pfr.extend(pfr_ext)
                        pdr.extend(pdr_ext)
                        pam.extend(pam_ext)
                        pof.extend(pof_ext)
                        pph.extend(pph_ext)

                        # Get the phase cycling parameters
                        p_phacyc_N_ext = parameter.get_option_by_name(
                            parameter.N_PHASE_CYCLES
                        )

                        p_phacyc_lev_ext = parameter.get_option_by_name(
                            parameter.PHASE_CYCLE_GROUP
                        )

                        # Extend the phase cycling parameters by the length of the pulse amplitude
                        p_phacyc_N_ext = [p_phacyc_N_ext.value] * len(pulse_amplitude)
                        p_phacyc_lev_ext = [p_phacyc_lev_ext.value] * len(
                            pulse_amplitude
                        )

                        p_phacyc_N.extend(p_phacyc_N_ext)
                        p_phacyc_lev.extend(p_phacyc_lev_ext)

        lime.p_frq = pfr
        lime.p_dur = pdr
        lime.p_amp = pam
        lime.p_offs = pof
        lime.p_pha = pph

        # Phase cycling
        lime.p_phacyc_N = p_phacyc_N
        lime.p_phacyc_lev = p_phacyc_lev

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
                    _, pulse_amplitude, _ = self.prepare_pulse_amplitude(
                        event, parameter
                    )
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
            tuple: A tuple containing the pulse shape and the pulse amplitude and phase
        """
        pulse_shape = parameter.get_option_by_name(TXPulse.TX_PULSE_SHAPE).value
        pulse_amplitude = abs(pulse_shape.get_pulse_amplitude(event.duration)) * (
            parameter.get_option_by_name(TXPulse.RELATIVE_AMPLITUDE).value / 100
        )
        pulse_amplitude = np.clip(pulse_amplitude, -0.99, 0.99)

        pulse_phase = parameter.get_option_by_name(TXPulse.TX_PHASE).value

        return pulse_shape, pulse_amplitude, pulse_phase

    def modulate_pulse_amplitude(
        self, pulse_amplitude: float, pulse_phase: float, event, lime: PyLimeConfig
    ) -> tuple:
        """Modulates the pulse amplitude for the limr object. We need to do this to have the pulse at IF frequency instead  of LO frequency.

        Args:
            pulse_amplitude (float): The pulse amplitude
            pulse_phase (float): The pulse phase
            event (Event): The event that contains the parameter
            lime (PyLimeConfig) : The PyLimeConfig object that is used to communicate with the pulseN driver

        Returns:
            tuple: A tuple containing the modulated pulse amplitude and the modulated phase
        """
        # num_samples = int(float(event.duration) * lime.sra)
        num_samples = int(float(event.duration) * lime.srate)
        tdx = np.linspace(0, float(event.duration), num_samples, endpoint=False)
        shift_signal = np.exp(
            1j * 2 * np.pi * self.limenqr.model.if_frequency * tdx + pulse_phase
        )

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
        self,
        lime: PyLimeConfig,
        pulse_shape,
        events,
        current_event,
        pulse_amplitude,
        sequence,
    ) -> list:
        """This method calculates and sets the offsets for the limr object.

        Args:
            lime (PyLimeConfig): The PyLimeConfig object that is used to communicate with the pulseN driver
            pulse_shape (Function): The pulse shape
            events (list): The pulse sequence events
            current_event (Event): The current event
            pulse_amplitude (float): The pulse amplitude
            sequence (QuackSequence): The pulse sequence to run

        Returns:
            list: The offsets for the limr object
        """
        blank_durations = self.get_blank_durations_before_event(
            events, current_event, sequence
        )

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
    def get_blank_durations_before_event(self, events, current_event, sequence) -> list:
        """This method returns the blank durations before the current event.

        Args:
            events (list): The pulse sequence events
            current_event (Event): The current event
            sequence (QuackSequence): The pulse sequence to run

        Returns:
            list: The blank durations before the current event
        """
        blank_durations = []
        previous_events_without_tx_pulse = self.get_previous_events_without_tx_pulse(
            events, current_event, sequence
        )
        for event in previous_events_without_tx_pulse:
            blank_durations.append(float(event.duration))
        return blank_durations

    # This method could be refactored in a potential pulse sequence module
    def get_previous_events_without_tx_pulse(
        self, events: list, current_event: "Event", sequence: QuackSequence
    ) -> list:
        """This method returns the previous events without a transmit pulse.

        Args:
            events (list): The pulse sequence events
            current_event (Event): The current event
            sequence (QuackSequence): The pulse sequence to run

        Returns:
            list: The previous events without a transmit pulse
        """
        index = events.index(current_event)
        previous_events = events[:index]
        result = []
        for event in reversed(previous_events):
            translatable = any(
                self.is_translatable_tx_parameter(param, sequence)
                for param in event.parameters.values()
            )
            if not translatable:
                result.append(event)
            else:
                break
        return reversed(
            result
        )  # Reversed to maintain the original order if needed elsewhere

    # Overrides from SpectrometerController  because we need the offset
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

        rx_event = self.find_rx_event(sequence, events)
        logger.debug("Found RX event: %s", rx_event)
        if not rx_event:
            logger.error("No RX event found in the pulse sequence")
            return None, None, None

        previous_events_duration = self.calculate_previous_events_duration(
            events, rx_event
        )
        rx_duration = float(rx_event.duration)

        offset = self.calculate_offset(lime)

        rx_begin = (
            float(previous_events_duration) + float(offset) + float(CORRECTION_FACTOR)
        )
        rx_stop = rx_begin + rx_duration

        # Phase
        readout_scheme = (
            rx_event.parameters[sequence.RX_READOUT]
            .get_option_by_name(RXReadout.READOUT_SCHEME)
            .get_value()[0]
        )

        return rx_begin * 1e6, rx_stop * 1e6, readout_scheme

    def find_rx_event(self, sequence, events):
        """This method finds the RX event in the pulse sequence.

        Args:
            sequence (QuackSequence): The pulse sequence to run
            events (list): The pulse sequence events

        Returns:
            Event: The RX event
        """
        for event in events:
            for parameter in event.parameters.values():
                logger.debug(f"Event: {event.name}")
                logger.debug(f"Parameter: {parameter}")
                if (
                    parameter.name == sequence.RX_READOUT
                    and parameter.get_option_by_name(RXReadout.RX).value
                ):
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

    def resample_data(self, measurement_data: Measurement) -> Measurement:
        """Resamples the data with the specified dwell time.

        Args:
            measurement_data (Measurement): The measurement data to resample

        Returns:
            Measurement: The resampled measurement data
        """
        # Resample the RX data to the dwell time settings
        dwell_time = self.limenqr.model.settings.rx_dwell_time

        dwell_time = UnitConverter.to_float(dwell_time) * 1e6
        logger.debug("Dwell time: %s", dwell_time)

        original_dwell_time = measurement_data.tdx[1] - measurement_data.tdx[0]
        logger.debug("Original dwell time: %s", original_dwell_time)

        new_sampling_frequency = 1.0 / dwell_time  # Dwell time in seconds
        n_data_points = int(measurement_data.tdx[-1] * new_sampling_frequency)
        tdx = np.arange(0, n_data_points) * dwell_time

        first_cycle = True

        # We iterate over the second dimension of tdy
        for tdy_cycle in measurement_data.tdy.T:
            logger.debug("Resampling to %s data points", n_data_points)
            tdy = resample(tdy_cycle, n_data_points)
            # Add empty dimension to make it compatible with the original data
            tdy = np.expand_dims(tdy, axis=1)
            logger.debug("Resampled data shape: %s", tdy.shape)
            logger.debug("Original  data shape: %s", measurement_data.tdy.shape)
            if first_cycle:
                measurement_data_resampled = Measurement(
                    measurement_data.name,
                    tdx,
                    tdy,
                    measurement_data.target_frequency,
                    frequency_shift=measurement_data.frequency_shift,
                    IF_frequency=measurement_data.IF_frequency,
                )
                first_cycle = False
            else:
                measurement_data_resampled.add_dataset(tdy)

        logger.debug("Resampled data shape: %s", measurement_data_resampled.tdy.shape)
        return measurement_data_resampled
