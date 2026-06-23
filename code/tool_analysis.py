import numpy as np

from matplotlib.figure import Figure

import madernpytools.log as mlog
import madernpytools.tools.frequency_response as mfrf
import madernpytools.backbone as mbb
import time
import pickle


class Cylinder(object):

    def __init__(self, w_n, d_cut, D):
        """
        :param w_n  : [Hz] Natural frequency
        :param d_cut: [mm] Distance between cutting lines
        :D          : [mm] Theoretical cylinder_properties diameter
        """
        self.w_n  = w_n
        self.d_cut = d_cut
        self.D = D

    @property
    def circumference(self):
        """ Circumference of cilinder [mm]
        """

        return np.pi*self.D

    @property
    def natural_period(self):
        """natural period of cylinder_properties [s]"""
        return 1.0/self.w_n

    def dist_per_cycle(self, speed):
        """

        :param speed: board speed [m/min]
        :return:
        """
        v = speed/60                      # [m/s] line speed
        return v*self.natural_period      # [m] distance per cycle

    def oscilations_per_rotation(self, speed):
        """ Compute number of oscilations per tool rotation
        :param Speed: [m/min] line speed
        """
        v = speed/60                      # [m/s] line speed
        d_per_osc = v*self.natural_period # [m] distance per cycle
        circ = self.circumference*1e-3    # [m] distance per rotation
        return circ/d_per_osc


class PairExcitation(object):

    def __init__(self, distance, cylinder, in_phase=True, damping_coefficient=0.01):
        """
        :param distance: [mm] Distance between the cutting lines
        :param cylinder: instance of cylinder_properties object
        :param in_phase: Indicates if the cutting lines are in phase (on the same side of cylinder_properties) or out-of-phase

        """

        self._distance = distance
        self._cylinder = cylinder
        self._in_phase = in_phase

        self._zeta = damping_coefficient

    def oscilations_at_speed(self, speed):
        """
        : param distance: [mm] Distance on which to compute number of oscilations
        :param Speed    : [m/min] line speed
        """

        v = speed/60                      # [m/s] line speed
        d_per_osc = v*self._cylinder.natural_period # [m] distance per cycle
        return (self._distance*1e-3)/d_per_osc

    def get_phase_shift(self, speed):
        """ Compute the phase shift that occurs between two cutting lines at given speed.
        :param d: [mm] Distance between two impacts (cutting lines)
        :param speed: [m/min] Speed at which tool rotates
        :returns phase_shift: Phase shift
        :returns N: Number of oscilations

        """
        n = self.oscilations_at_speed(speed)
        N = np.floor(n)  # Number of oscilations
        r = n-N          # Remainder with respect to N

        # Compute phase:
        ind_after  = np.where( (0<r) & (r<0.5))   # out-of-phase after  N
        ind_before = np.where( (0.5<r) & (r<1.0)) # out-of-phase before N+1
        r[ind_before] = -(1-r[ind_before])

        phase_shift = np.pi*2*r

        return phase_shift, n

    def get_excitation_level(self, speeds):
        """
        Returns excitation level of cutting line at specific speed
        """
        # Compute response:
        phase, n = self.get_phase_shift(speeds)
        if self._in_phase:
            tmp = np.cos(phase)*np.exp(-self._zeta*n)
        else:
            # Hits on other side of cylinder_properties
            tmp = -np.cos(phase)*np.exp(-self._zeta*n)

        return tmp


def get_excitation_frequency(velocity, d_excitation):
    """ Compute the excitation frequency of consequtive impacts

    :param velocity: Velocity of moving object [m/s]
    :param d_excitation: Distance between impacts [m]
    :return: Excitation frequency [Hz]
    """
    return velocity/d_excitation


class SignalSmoother(object):

    def __init__(self, window_length=1, window='hanning'):
        """
        Smooth the data using a window with requested size.

        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the beginning and end part of the output signal.


        :param window_length: The dimension of the smothing window; should be an odd integer
        :param window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'. Flat yields
                       moving average smoothing
        """

        if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        self._window_length = window_length
        self._window = window

    @property
    def window_length(self):
        return self._window_length

    def smooth(self, data):
        """Smooth data"""

        if data.size < self._window_length:
            raise ValueError("Input vector needs to be bigger than window size.")

        if self._window_length < 3:
            return data

        s = np.r_[data[self._window_length - 1:0:-1], data, data[-2:-self._window_length - 1:-1]]

        if self._window == 'flat':  # moving average
            w = np.ones(self._window_length, 'd')
        else:
            w = eval('np.' + self._window + '(self._window_length)')

        return np.convolve(w / w.sum(), s, mode='valid')[:-(self._window_length - 1)]


class EncoderConverter(object):

    def __init__(self, sampling_rate, d_per_rev, filter_window, window_length):
        """

        :param sampling_rate: Sampling rate [Hz]
        :param d_per_rev: circumference length [m] corresponding to encoder signal
        :param filter_window: Window-type
        :param window_length: Number of samples to use in filter
        """

        self._d_per_rev = d_per_rev
        self._sampling_rate = sampling_rate
        self._filter = SignalSmoother(window_length=window_length,
                                      window=filter_window)

    def signal2speed(self, data):
        """

        :param data: encoder signal [rad/s], N-d ndarray
        :return: Speed signal [m/s]
        """

        # Normalize the encoder signal such that it's lowest value is zero:
        data -= data.min()

        # Compute gradient:
        grad = np.gradient(data/(2 * np.pi))  # 1/n (rev fraction)

        # We assume only positive speed, if the mean speed is negative, it implies the encoder was turning backwards:
        #if grad.max() > 0:
        #    grad = -grad

        # Remove discontinuities at 2*pi switches
        err_ind = np.where(np.abs(grad) > np.abs(grad).max()/2)[0]
        for i in err_ind:
            s1 = grad[i - 1]
            grad[i] = s1

        # Smooth the signal:
        grad_smooth = self._filter.smooth(grad)

        # Speed signal
        speed = grad_smooth*self._sampling_rate # 1/s, revs per second
        return speed * self._d_per_rev # [m/s]


class GapComputer(object):

    def __init__(self, initial_gap, signal_scale=100, base_names=None):
        """ Computes gap values from sensor data. The

        :param initial_gap:  Initial gap values (keys: ['OS', 'DS'], mu)
        :param signal_scale: Sensor scale [mu/V]
        :param base_names:
        """

        self._base_names = mbb.ArgumentVerifier(list, ['OS', 'DS']).verify(base_names)
        self._init_gap = mbb.ArgumentVerifier(dict, {'OS': 0.0, 'DS': 0.0}).verify(initial_gap)
        self._signal_scale = mbb.ArgumentVerifier(float, 100.0).verify(float(signal_scale))

    def signal2gap(self, signals, calibration_values):
        """ Converts signals to gap data

        :param signals: A dictionary containing signals using the following keys:
                ['OS_upper', 'OS_lower',
                 'DS_upper', 'DS_lower',
                 'OS_mid', 'OS_top'
                 'DS_mid', 'DS_top']
        :param calibration_values: Dictionary of calibration values for each signal.
        :return: Dictionary of gap estimates [mu] (keys: ['OS', 'DS', 'OS_mid', 'DS_mid']
        """

        calib_values = calibration_values
        scale = self._signal_scale
        init_gabs = self._init_gap
        sides = self._base_names

        gaps = {}

        # Compute gap on sides:
        for side in sides:
            up_key = '{0}_{1}'.format(side, 'upper')
            low_key = '{0}_{1}'.format(side, 'lower')

            up_val = scale * (signals[up_key] - float(calib_values['calib_{0}'.format(up_key)]))
            low_val = scale * (signals[low_key] - float(calib_values['calib_{0}'.format(low_key)]))

            gaps[side] = init_gabs[side] + up_val + low_val

        # Compute bending:
        distance_vals = {}
        for key in ['OS_top', 'DS_top', 'OS_mid', 'DS_mid']:
            distance_vals[key] = scale * (signals[key] - float(calib_values['calib_{0}'.format(key)]))

        top_avg = 0.5 * (distance_vals['OS_top'] + distance_vals['DS_top'])
        bend_os = distance_vals['OS_mid'] - top_avg
        bend_ds = distance_vals['DS_mid'] - top_avg

        # Estimate gaps in tool
        gap_avg = 0.5 * (gaps['OS'] + gaps['DS'])
        gaps['OS_mid'] = -2 * bend_os + gap_avg
        gaps['DS_mid'] = -2 * bend_ds + gap_avg
        gaps['time_signal'] = signals['time']

        return gaps


class SpeedAnalysisJob(mbb.BackgroundJob):

    def __init__(self, data, frequency_range, spectral_configuration, speed_id=0, job_id=None, f_step=None, verbose=False):
        """
        Background job to perform frequency analysis on specified data.

        How to use:
        1) Create object by specifying data, frequency range, spectral configuration and an optional speed id
        2) Give job to background worker
        3) Run background worker

        :param speed_id:
        :param frequency_range:
        :param spectral_configuration:
        """
        mbb.BackgroundJob.__init__(self)

        self._frequency_range = frequency_range
        self._fstep = np.gradient(self._frequency_range).mean() if f_step is None else f_step
        self._analyzer = mfrf.SpectralAnalyzer(spectral_configuration)
        self._mag_mesh = np.zeros(len(self._frequency_range))
        self.speed_id = speed_id
        self._data = data
        self._job_id = job_id
        self.verbose=verbose

    @property
    def data(self):
        """Data to analyze """
        return self._data

    @data.setter
    def data(self, value):
        """Data to analyze
        :value : n-d array
        """
        self._data = value

    @property
    def magnitude_data(self):
        """

        :return: n-dimensional array of magnitue values
        """
        return self._mag_mesh

    def _work_sequence(self):
        # This should be done in parallel:




        res = self._analyzer.analyze(self._data[:, None])

        # Gather magnitudes in selected frequency ranges:
        for j, f_sel in enumerate(self._frequency_range):
            f_diff = abs(res.frequencies - f_sel)
            f_ind = np.where(f_diff < self._fstep)[0] # Selected frequencies:

            if len(f_ind) > 0:
                w = 1 - (f_diff[f_ind] / f_diff[
                    f_ind].max())  # Get weight inversely according to distance (closest highest weight)
                w /= w.sum()  # Normalize such that weights sum to 1
                self._mag_mesh[j] = (abs(res.Sff[f_ind])*w).sum()
            elif self.verbose:
                print('No result for frequency {0} (step: {1}'.format(f_sel, self._fstep))

    def get_result(self):
        return mbb.JobResult(self.state, self)


class ToolSignature(object):

    def __init__(self, speed_range, freq_range, n_threads=2, verbose=False, speed_step=None, f_step=None, **kwargs):
        """ Object represents the tool dynamics over the defined speed and frequency ranges

        :param speed_range: array of speeds for which the tool signature holds information
        :param freq_range: array of frequencies for which the tool signature holds information
        :param n_threads: number of threads to use for computations (default:2)
        :param verbose: set to true to let object dump computation information to terminal
        :param kwargs:
        """

        # Get gradients
        self._speed_range = speed_range
        self._freq_range = freq_range
        self._f_step = f_step
        self._speed_step = np.gradient(speed_range).mean() if speed_step is None else speed_step
        self._freq_step = np.gradient(freq_range).mean()
        self.verbose = verbose

        # Define mesh:
        self._fmesh, self._smesh = np.meshgrid(freq_range, speed_range)
        self._mag_mesh = np.zeros(self._fmesh.shape)

        # Range sample sizes
        self._n_samples = np.zeros(len(speed_range))

        # Define spectral analysers
        conf = mfrf.SpectralAnalyzerConfiguration()
        conf.i_input = 0
        conf.i_output = None
        conf.window_fraction = kwargs.get('window_fraction', 0.1)
        conf.overlap_fraction = kwargs.get('overlap_fraction', 0.2)
        conf.sampling_rate = kwargs.get('sampling_rate', 10240)
        conf.scaling='spectrum'
        self._spectral_conf = conf

        # Async handling of data:
        self._computation_que = mbb.JobQue(n_threads=n_threads)
        self._sub_finished_speed_analysis = mbb.SimpleEventSubscriber(h_callback=self._cb_finished_speed_analysis)

    def save(self, filename):
        """ Saves the tool signature to specified filename. Method is based on PYthon Pickle

        :param filename: filename to use for storage
        """

        data = {'mag_mesh': self._mag_mesh,
                'speed_range': self._speed_range,
                'freq_range': self._freq_range,
                }

        with open('{0}.ts'.format(filename), 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        """ Method tries to load tool signature from given filename

        :param filename: filename to load.
        :return:
        """

        with open('{0}.ts'.format(filename), 'rb') as f:
            obj = pickle.load(f)
            speed_range = obj['speed_range']
            freq_range = obj['freq_range']
            mag_mesh = obj['mag_mesh']

            new_sig = ToolSignature(speed_range, freq_range)
            new_sig._mag_mesh = mag_mesh

        return new_sig

    @property
    def magnitude_mesh(self):
        """ Returns magnitude mesh

        :return: n_freqs x n_speeds array
        """
        return self._mag_mesh

    @property
    def frequency_mesh(self):
        """ Returns frequency mesh

        :return: n_freqs x n_speeds array
        """
        return self._fmesh

    @property
    def speed_mesh(self):
        """ Returns speed mesh

        :return: n_freqs x n_speeds array
        """
        return self._smesh

    @property
    def frequency_range(self):
        """ Returns signature frequency range

        :return:
        """
        return self._freq_range

    @property
    def speed_range(self):
        """ Returns signature speed range

        :return:
        """
        return self._speed_range

    def _cb_finished_speed_analysis(self, publisher: mbb.BackgroundWorker):
        """

        :param publisher: Background worker that handled a speed analysis
        :return:
        """

        # Collect result:
        speed_analyzer = publisher.get_data().data

        # Data con
        i = speed_analyzer.speed_id  # Index in mesh grid
        n_new_data = len(speed_analyzer.data)  # Number of data points used to compute (new) result
        mag_data = speed_analyzer.magnitude_data

        if self._n_samples[i] == 0:
            # First sample:
            self._mag_mesh[i, :] = mag_data
        else:
            raise RuntimeWarning('Tried to overwrite already existing samples for speed {0}[m/min]'.format(
                self._speed_range[i]))

        # Update number of samples
        self._n_samples[i] += n_new_data

    def update(self, data, speeds, blocking=False):
        """Update tool signature

        :param data: n-dimensional array of sensor data of which to estimate the tool signature (e.g. acceleration data)
        :param speeds: n-dimensional array of line-speed information
        :param blocking: block thread until computation has finished (default: false)
        :return:
        """

        # This should be performed in background:
        for i, s in enumerate(self._speed_range):
            # Get speeds:
            ind = np.where(abs(speeds - s) < self._speed_step)[0]
            if self.verbose:
                print('{} m/min: nperseg: {}'.format(s, self._spectral_conf.get_csd_args(n=len(ind))['nperseg']))

            if self._spectral_conf.get_csd_args(n=len(ind))['nperseg'] < 0:
                if self.verbose:
                    print('Not enough samples found for {0} [m/min]'.format(s))
            # Assign job to idle worker:
            elif len(ind) > 0:
                newjob = SpeedAnalysisJob(data[ind], self._freq_range, self._spectral_conf,
                                          speed_id=i, f_step=self._f_step, verbose=self.verbose)
                newjob.connect(self._sub_finished_speed_analysis)
                self._computation_que.append(newjob)
            elif self.verbose:
                print('No results found for {0} [m/min]'.format(s))

        # Wait until finished:
        if blocking:
            while self._computation_que.is_running():
                cnt = self._computation_que.active_jobs() + len(self._computation_que)
                if cnt > 0:
                    if self.verbose:
                        print('Waiting for {0} jobs'.format(cnt))
                        time.sleep(1)
                    else:
                        time.sleep(.1)


class FileLoadJob(mbb.BackgroundJob, mbb.JobResult):

    def __init__(self, filename, index):
        """ Background job to load log files

        :param filename: Filename to load
        :param index: job index
        """
        mbb.BackgroundJob.__init__(self)
        mbb.JobResult.__init__(self, None, None)

        self._filename = filename
        self._index = index
        self._data = {}
        self._info = None
        self._reader = mlog.CSVLogReader()

    def _work_sequence(self):
        """
        Load data and store data and info in job-type
        :return:
        """
        info, data = self._reader.read(self._filename)
        self._data = dict(zip(info.signal_header, data.T))
        self._info = info

    @property
    def data(self):
        """ Data loaded by job

        :return:
        """
        return self._data

    @property
    def index(self):
        """Job index

        :return:
        """
        return self._index

    @property
    def info(self):
        """LogInfo of loaded data file

        :return:
        """
        return self._info

    def get_result(self):
        return self


class AbstractLineSpeedComputer(object):
    """
    Abstract class which represent a generalized line-speed computer. It has a single function which accepts
    a raw_data dictionary and returns a speed signal

    """

    def compute_speeds(self, raw_data):
        """

        :param raw_data: Raw data dictionary of measurement data
        :return: line speed signal
        """
        raise NotImplementedError()


class DummyLineSpeedComputer(AbstractLineSpeedComputer):

    def compute_speeds(self, raw_data):
        """ Dummy speed computer returns a zero mean noisy speed signal

        :param raw_data: Raw data dictionary of measurement data
        :return: zero line speed signal
        """
        # Returns zero speed
        keys = list(raw_data.keys())
        return np.random.randn(len(raw_data[keys[0]]))


class EncoderBasedLineSpeedComputer(AbstractLineSpeedComputer):

    def __init__(self, sampling_rate, d_per_rev, signal_key='line_speed'):
        self._sampling_rate = sampling_rate
        self._d_per_rev = d_per_rev
        self._signal_key = signal_key

    def compute_speeds(self, raw_data):
        """

        :param raw_data: Raw data dictionary of measurement data
        :return: zero line speed signal
        """
        encoder_converter = EncoderConverter(sampling_rate=self._sampling_rate,
                                             d_per_rev=self._d_per_rev,
                                             filter_window='flat',
                                             window_length=1000)

        return encoder_converter.signal2speed(raw_data[self._signal_key])


class LinearLineSpeedSignal(AbstractLineSpeedComputer):

    def __init__(self, signal_key='line_speed', scale=1.0, sampling_rate=None):
        """Line speed computer for linearly scaling of signals

        :param scale: linear scaling of the signal
        :param signal_key: key which corresponds to the speed signal
        :param sampling_rate: Data sampling rate used for the low pass filtering of the signal
        :param low_pass_frequency: Frequency at which the lowpass filter operates
        :return:
        """
        self._scale = scale
        self._signal_key = signal_key
        self._sampling_rate = mbb.ArgumentVerifier(int, 1).verify(sampling_rate)
        self._lp_filter = mfrf.LowPassFilter(low_pass_frequency=200, fs=self._sampling_rate, order=2)

    @property
    def low_pass_filter(self):
        return self._lp_filter

    def compute_speeds(self, raw_data):
        """

        :param raw_data: Raw data dictionary of measurement data
        :return: zero line speed signal
        """
        f_data = self._lp_filter.filter(raw_data[self._signal_key] * self._scale)
        return f_data


class ExperimentalResults(object):

    def __init__(self, speed_computer=None, verbose=False):
        """Object to hold measurement results of inline measurements

        :param speed_computer: object required for the computation of line-speed based on the raw measurement singal
        :param verbose:
        """

        # Internal data structures:
        self._raw_data = {}         # Data dictionary
        self._loader_data = {}      # Temporary dictionary for loading data async
        self._info = None
        self._file_range = None
        self._filename_pattern = None
        self._loading_active = False
        self.verbose = verbose

        self._speed_computer = mbb.ArgumentVerifier(AbstractLineSpeedComputer, DummyLineSpeedComputer()
                                                    ).verify(speed_computer)

        # Que for data loading:
        self._load_que = mbb.JobQue(n_threads=2)

        # Subscriber for data loader
        self._sub_load_finished = mbb.SimpleEventSubscriber(h_callback=self._cb_data_loaded)
        self.event_finished_loading = mbb.EventPublisher(ExperimentalResults)

    def load(self, filename_pattern, file_range, blocking=False):
        self._filename_pattern = filename_pattern
        self._file_range = file_range

        self._load_data()

        if blocking:
            while self.loading_active:
                time.sleep(0.1)

    @property
    def loading_active(self):
        """ Flag indicating loading is active

        :return:
        """
        return self._loading_active

    def __getitem__(self, key):
        return self._raw_data[key]

    def keys(self):
        return self._raw_data.keys()

    def _load_data(self):
        """
        Load data according to filename_pattern and file range
        :return:
        """

        self._loader_data = {}      # Temporary dictionary for loading data async

        # Load data
        self._loading_active = True
        for i in self._file_range:
            load_job = FileLoadJob(self._filename_pattern.format(i), index=i)
            load_job.connect(self._sub_load_finished)
            self._load_que.append(load_job)

    def _cb_data_loaded(self, publisher: mbb.BackgroundWorker):
        """
        Callback to handle the data loaded by a FileLoaderJob

        :param publisher:
        :return:
        """
        # Add data:
        file_loader = publisher.get_data()
        self._info = file_loader.info
        self._loader_data[file_loader.index] = file_loader.data

        # Notify user about progress:
        if self.verbose and (np.mod(len(self._loader_data), 5) == 0):
            print('Loaded {0}/{1} files'.format(len(self._loader_data), len(self._file_range)))

        # Check if we are finished processing:
        if not self._load_que.is_running():
            self._merge_loader_data()

            if self.verbose:
                print('Smoothing encoder signal...')
            self._raw_data['line_speed'] = self._speed_computer.compute_speeds(self._raw_data)

            if self.verbose:
                print('Finished loading files')
            self._loading_active = False
            self.event_finished_loading.raise_event(self)

    def _merge_loader_data(self):
        """
        Merge the data gathered in loader_data into raw_data
        :return:
        """
        if self.verbose:
            print('Merging data...')
        for index in sorted(self._loader_data):
            sub_data = self._loader_data[index]
            for key, item in sub_data.items():
                if key in self._raw_data:
                    self._raw_data[key] = np.hstack( (self._raw_data[key], item))
                else:
                    self._raw_data[key] = item

    @property
    def info(self):
        return self._info


class AbstractPlotItem(object):

    def __init__(self, ax=None, **kwargs):
        # Create axis if not given:
        if ax is None:
            fig = Figure(figsize=(4, 3))
            ax = fig.add_subplot(111)
        self._ax = ax

    @property
    def ax(self):
        return self._ax


class SignatureSpeedCrossSection(AbstractPlotItem):

    def __init__(self, toolsignature, s_max=350, s_min=0, ax=None, **kwargs):
        AbstractPlotItem.__init__(self, ax)

        # Define data structure:
        self._smin = s_min
        self._smax = s_max
        self._tool_signature = toolsignature
        self._line, = self.ax.plot([], [], **kwargs)

        # Setup axis:
        self.ax.set_ylabel('Magnitude')
        self.ax.set_xlabel('Frequency [Hz]')

        self.update()

    def update(self):
        ts = self._tool_signature
        mag = ts.magnitude_mesh

        ind_sel = np.where(np.logical_and(self.s_min < ts.speed_range,
                                          ts.speed_range < self.s_max))[0]
        mag_sel = ts.magnitude_mesh[ind_sel, :].sum(axis=0)

        nz_sel = np.where(mag_sel > 0)[0] # Non-zero selection
        self._line.set_data(ts.frequency_range[nz_sel], np.log10(mag_sel[nz_sel]))

        self.ax.relim()
        self.ax.autoscale_view(True, True, True)

    @property
    def s_min(self):
        return self._smin

    @s_min.setter
    def s_min(self, value):
        self._smin = value
        self.update()

    @property
    def s_max(self):
        return self._smax

    @s_max.setter
    def s_max(self, value):
        self._smax = value
        self.update()

    @property
    def tool_signature(self):
        return self._tool_signature

    @tool_signature.setter
    def tool_signature(self, value):
        self._tool_signature = value
        self.update()


class SignatureFrequencyCrossSection(AbstractPlotItem):

    def __init__(self, tool_signature: ToolSignature, f_max=350, f_min=0, ax=None, **kwargs):
        AbstractPlotItem.__init__(self, ax)

        # Define data structure:
        self._fmin = f_min
        self._fmax = f_max
        self._tool_signature = tool_signature
        self._line, = self.ax.plot([], [], **kwargs)

        # Setup axis:
        self.ax.set_ylabel('Magnitude [dB]')
        self.ax.set_xlabel('Speed [m/min]')

        self.update()

    def update(self):
        ts = self._tool_signature
        mag = ts.magnitude_mesh

        ind_sel = np.where(np.logical_and(self.f_min < ts.frequency_range,
                                          ts.frequency_range < self.f_max))[0]
        mag_sel = ts.magnitude_mesh[:, ind_sel].sum(axis=1)
        nz_sel = np.where(mag_sel > 0)[0]

        self._line.set_data(ts.speed_range[nz_sel], mag_sel[nz_sel])
        self.ax.relim()
        self.ax.autoscale_view(True, True, True)

    @property
    def f_min(self):
        return self._fmin

    @f_min.setter
    def f_min(self, value):
        self._fmin = value
        self.update()

    @property
    def f_max(self):
        return self._fmax

    @f_max.setter
    def f_max(self, value):
        self._fmax = value
        self.update()

    @property
    def tool_signature(self):
        return self._tool_signature

    @tool_signature.setter
    def tool_signature(self, value):
        self._tool_signature = value
        self.update()


class ToolSignaturePlot(AbstractPlotItem):

    def __init__(self, tool_signature: ToolSignature, ax=None):
        """Visualization of the tool signature"""
        AbstractPlotItem.__init__(self, ax)
        self._mesh_scale = np.log10

        self._tool_signature = tool_signature
        self._mesh = None

        # Setup axis
        self.ax.set_xlabel('Speed [m/min]')
        self.ax.set_ylabel('Frequency [Hz]')

        self.update()

    @property
    def tool_signature(self):
        return self._tool_signature

    @tool_signature.setter
    def tool_signature(self, value):
        self._tool_signature = value
        self.update()

    def update(self, **kwargs):
        """Update plot based on current data
        """

        if self._mesh is not None:
            self._mesh.remove()

        mag = self.tool_signature.magnitude_mesh
        freq = self.tool_signature.frequency_mesh
        self._mesh = self.ax.pcolor(self.tool_signature.speed_mesh,
                                    self.tool_signature.frequency_mesh,
                                    self.f_mesh_scale(mag, freq),
                                    #self._mesh_scale(mag/(freq**2)),
                                    cmap='jet', **kwargs
                                    )

    def f_mesh_scale(self, mag, freq):
        return np.log10(mag)

    def add_natural_frequencies(self, frequencies, **kwargs):
        """Add Natural Frequencies to plot"""

        linewidth = kwargs.get('linewidth', 1)
        linestyle = kwargs.get('linestyle', '--')
        color = kwargs.get('color', 'gray')
        alpha = kwargs.get('alpha', 0.7)

        xlim = self._ax.get_xlim()
        for i, w_n in enumerate(frequencies):
            self.ax.plot(xlim, [w_n, w_n], '-', color=color, linewidth=linewidth, alpha=alpha)
            self.ax.text(xlim[-1], w_n, r'$\omega_{0}$'.format(i + 1), va='bottom', ha='right', rotation=0,
                         color='black')
        self.ax.set_xlim(xlim)

    def add_excitation_frequency(self, dist, **kwargs):
        """

        :param dist: distance between two impacts [m]
        :param kwargs:
        :return:
        """

        linewidth=kwargs.get('linewidth', 1)
        linestyle=kwargs.get('linestyle', '--')
        color=kwargs.get('color', 'gray')

        # Plot excitation frequencies:
        xlim = np.array(self._ax.get_xlim())
        w_ex = get_excitation_frequency(velocity=xlim / 60, d_excitation=dist)
        self.ax.plot(xlim, w_ex, linestyle, color=color, linewidth=linewidth)
        self.ax.set_xlim(xlim)












