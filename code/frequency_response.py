import numpy as np
import scipy as sp
import scipy.signal # Todo: Quick fix to continue working, this module raises an error on loading, in numpy 1.18
import re, traitlets, os
import xml.etree.cElementTree as ET
from madernpytools.backbone import ArgumentVerifier, AbstractDataProcessor, EventPublisher
import madernpytools.backbone as mbb
import threading as thrd
import madernpytools.plot as mplot
import madernpytools.log as mlog

import matplotlib.cm as cm
from matplotlib.figure import Figure

# Constants:
cols = cm.Set1(np.linspace(0, 1, 10))[:, :3]


class AbstractMode(object):

    @property
    def natural_frequency(self):
        """Natural frequency [Hz]"""
        raise NotImplementedError()

    @property
    def poles(self):
        """Complex conjugate pole pair corresponding to this mode"""
        raise NotImplementedError()

    @property
    def damping(self):
        """Damping"""
        raise NotImplementedError()

    @property
    def damping_ratio(self):
        raise NotImplementedError()

    @property
    def shape(self):
        """Mode shape"""
        raise NotImplementedError()


class Mode(AbstractMode):

    def __init__(self, natural_frequency,
                 damping_ratio=1.0
                 ):
        """Mode properties
        :param natural_frequency: Natural frequency of mode [Hz]
        :type natural_frequency: float
        :param damping_ratio: Damping ratio of the mode [-]
        :type damping_ratio: float
        """

        self._f_n = natural_frequency
        self._damping_ratio = damping_ratio
        self._mode_scale = 1.0
        self._node_gains = None
        self._node_locations = None
        self._z0 = 0.0

    @property
    def natural_frequency(self):
        """Natural frequency [Hz]"""
        return self._f_n

    @property
    def gain(self):
        return self._node_gains

    @property
    def poles(self):
        """Complex conjugat pole pair corresponding to this mode"""
        return ((-self.damping_ratio + 1j * np.sqrt(1-self._damping_ratio**2))*self._f_n * np.pi * 2,
                (-self.damping_ratio - 1j * np.sqrt(1-self._damping_ratio**2))*self._f_n * np.pi * 2)

    @property
    def damping(self):
        """Damping
        """
        return self._damping_ratio*self._f_n*np.pi*2

    @property
    def damping_ratio(self):
        return self._damping_ratio

    @property
    def shape(self):
        """Mode shape"""
        return (self._node_gains*self._mode_scale).imag

    @property
    def node_locations(self):
        if self._node_locations is None and isinstance(self._node_gains, np.ndarray):
            # Nodes locations not defined, but mode shape is available, return equally spaced nodes
            return np.arange(len(self._node_gains))
        elif self._node_gains is not None:
            # Nodes locations defined, return it:
            return self._node_locations
        else:
            return None

    @node_locations.setter
    def node_locations(self, value):
        if isinstance(self._node_gains, np.ndarray):
            # Check value length:
            if len(value) == len(self._node_gains):
                self._node_locations = value
            else:
                raise RuntimeError('Number of node locations does not match the number of nodes')
        else:
            self._node_locations = value

    @property
    def node_gains(self):
        return self._node_gains

    def get_zpg(self, node_id=None):
        """Get ZeroPoleGain system of this mode
        :param node_id: index of node.
        :type node_id: int
        """

        gain = 1.0
        if isinstance(node_id, int) and (node_id < len(self._node_gains)):
            gain = self._node_gains[node_id]
        zeros = [self._z0]
        return sp.signal.ZerosPolesGain(zeros, self.poles, gain)

    def estimate_shape(self, spectral_responses, z0=0.0):
        """Estimate the mode shape from spectral data
        :param spectral_responses: SpectralList of spectral samples taken at different nodes.
        :type: spectral_responses: SpectralList
        :param z0: Transferfunction zero to use in the estimation process
        """

        # Set z0
        self._z0 = z0

        # Get datae@f_n
        f_n = self.natural_frequency
        f_ind = np.argsort(np.abs(spectral_responses.mean.frequencies-f_n))[0]
        H_fn = np.array([d.H[f_ind] for d in spectral_responses])

        # Compute response of second order system, with only z0:
        _, resp_wn = self.get_zpg().freqresp(f_n*np.pi*2)

        # Estimate node-specific gains:
        self._node_gains= np.zeros(len(H_fn), dtype=np.complex128)
        err = np.zeros(len(H_fn), dtype=np.complex128)
        for i, H_fn_k in enumerate(H_fn):
            # Compute required node zero, and update
            self._node_gains[i] = H_fn_k/resp_wn[0]

            # Compute error, for new node zero:
            _, resp = self.get_zpg(node_id=i).freqresp(f_n*np.pi*2)

            err[i] = resp[0]-H_fn_k

        # Compute scaling factor required to transform zeros into mode shape:
        scales = H_fn.imag/self._node_gains.imag  # Scaling required to match Nk the measured shape H_fn
        self._mode_scale = scales.mean()

        return self.shape, err


class ModeList(object):
    """List of mode objects"""

    def __init__(self):
        self._list = {}

    def append(self, item):
        """Add mode to list"""
        if type(item) is Mode:
            self._list['{0:.0f}'.format(item.natural_frequency)] = item

    def __getitem__(self, key):
        """Get mode by frequency"""
        return self._list['{0:.0f}'.format(key)]

    def get_frequencies(self):
        """Get list of frequencies in the list"""
        return sorted([m.natural_frequency for key, m in self._list.items()])


class AbstractSpectralSample(object):

    def keys(self):
        raise NotImplementedError()

    def get(self, key):
        raise NotImplementedError()

    @property
    def Sxx(self):
        raise NotImplementedError()

    @property
    def Sxf(self):
        raise NotImplementedError()

    @property
    def Sff(self):
        raise NotImplementedError()

    @property
    def Cxf(self):
        raise NotImplementedError()

    @property
    def H(self):
        raise NotImplementedError()

    @property
    def frequencies(self):
        raise NotImplementedError()

    @property
    def label(self):
        raise NotImplementedError()


class SpectralSample(AbstractSpectralSample, AbstractDataProcessor):

    def __init__(self, freq=None, Sxx=None, Sxf=None, Sff=None, Cxf=None, H=None, label=''):
        """

        :param freq: array of frequency values corresponding to the specified spectra [Hz]
        :param Sxx: Spectral spectrum of output values
        :param Sxf: Cross spectral spectrum of output and input values
        :param Sff: Spectral spectrum of output values
        :param Cxf: Coherence of output and ipnut values
        :param H:   Ratio between output and input spectra (transfer function)
        :param label: Label
        """
        AbstractDataProcessor.__init__(self)
        self._dict = {'freq': freq,
                      'Sxx': Sxx,
                      'Sxf': Sxf,
                      'Sff': Sff,
                      'Cxf': Cxf,
                      'H': H
                      }
        self._label = ArgumentVerifier(str, "").verify(str(label))

    def get(self, key):
        return self._dict[key]

    def keys(self, available_spectra_only=True):
        """Returns a list of (available) spectral keys.
        :param available_spectra_only: If true, returns only the keys of available spectra
         """
        if available_spectra_only:
            return [key for key in self._dict.keys() if self._dict[key] is not None]
        else:
            return self._dict.keys()

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        if key in self._dict.keys():
            self._dict[key] = value
        else:
            raise RuntimeError("Not a valid Spectral key. Options are {0}".format(self._dict.keys()))

    @property
    def label(self):
        """

        :return: Label of the spectral sample
        """
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def Sxx(self):
        return self._dict['Sxx']

    @property
    def Sxf(self):
        return self._dict['Sxf']

    @property
    def Sff(self):
        return self._dict['Sff']

    @property
    def Cxf(self):
        return self._dict['Cxf']

    @property
    def H(self):
        return self._dict['H']

    @property
    def frequencies(self):
        return self._dict['freq']

    def compute_from_data(self, data, i_input=None, i_output=None, f_range=None, scaling='spectrum', **kwargs):
        """Compute the Frequency response function of the provided data
        :param data :  N x d nd-array (N values, d dimensions)
        :param i_input :  Index of input signal (e.g. Impulse hammer)
        :param i_output :  Index of output signal (e.g. Accelerometer)
        :param f_range : Frequency range to consider, defaults to [0, 2500] [Hz]
        :param scaling: Scaling of spectral data ('spectrum' or 'density')
        :param **kwargs :  Optional filter settings (see documentation of scipy.signal.csd(..) )
        """
        config = SpectralAnalyzerConfiguration(i_input=i_input, i_output=i_output, f_range=f_range, scaling=scaling,
                                               csd_args=kwargs)
        spectral_analyzer = SpectralAnalyzer(config)
        new_sample = spectral_analyzer.analyze(data)

        # Update internal state:
        for key in new_sample.keys():
            self[key] = new_sample[key]

        # Return new state:
        return self

    def get_cutoff_frequency(self, amplitude_reduction=20):
        """Get the cut-off frequency for this data set"""
        Sff = 20 * np.log10(np.abs(self['Sff']))
        w_co = self['freq'][np.where(Sff <= (np.max(Sff) - amplitude_reduction))[0][0]]

        return w_co


class SpectralAnalyzerConfiguration(mbb.IXML):

    def __init__(self, f_range=None, i_input=0, i_output=1, scaling='spectrum', sampling_rate=None,
                 filter_window=None, window_fraction=None, overlap_fraction=None, csd_args=None):
        """

        :param window_fraction: Size of the filter window as fraction of total number of samples
        :param overlap_fraction: Overlap as fraction of the window size
        :param filter_window: Window settings (default: ('exponential', None, 1e6))
        :param i_input: Data index of input values. default=2
        :param i_output: Data index of output values. default=1
        :param frequency_range: Frequency range to display. default = [0,5000]
        """

        # Filter settings
        self.window_fraction = ArgumentVerifier(float, 0.5).verify(window_fraction)
        self.overlap_fraction = ArgumentVerifier(float, 0.5).verify(overlap_fraction)
        self.filter_window = ArgumentVerifier((tuple, str), 'hann').verify(filter_window)
        self.i_input = ArgumentVerifier(int, None).verify(i_input)
        self.i_output = ArgumentVerifier(int, None).verify(i_output)
        self.f_range = ArgumentVerifier(list, [0, 5000]).verify(f_range)
        self.scaling = ArgumentVerifier(str, 'spectrum').verify(scaling)
        self.sampling_rate = ArgumentVerifier(int, -1).verify(sampling_rate)
        self._csd_args = ArgumentVerifier(dict, {}).verify(csd_args)

    def get_csd_args(self, n):
        """
        :param n: Number of samples on which absolute window size is computed
        :return:
        """

        csd_args = {'window': self.filter_window,
                    'nperseg': int(n*self.window_fraction),
                    'noverlap': int(n*self.window_fraction*self.overlap_fraction),
                    'fs': self.sampling_rate
                }

        # Add other csd arguments:
        for key, item in self._csd_args.items():
            csd_args[key] = item

        return csd_args

    def to_xml(self):
        root = ET.Element('spectral_settings')
        for attr in ['window_fraction', 'overlap_fraction', 'i_input', 'i_output', 'f_range',
                     'filter_window', 'scaling', 'sampling_rate']:
            xml_item = ET.SubElement(root, attr)
            xml_item.text = str(getattr(self, attr))

        return root

    @staticmethod
    def from_xml(xml_tree: ET.ElementTree):
        """ Extracts Display settings from xml element
        :param xml_tree: xml tree containing the experiment configuration elements
        :return: DisplaySettings
        """

        # Get window:
        if '(' in xml_tree.find('filter_window').text:
            # assume tuple
            xml_window = xml_tree.find('filter_window').text
            window = eval(xml_window)
        else:
            # Assume string
            window = str(xml_tree.find('filter_window').text)

        conf =SpectralAnalyzerConfiguration(
                               window_fraction=eval(xml_tree.find('window_fraction').text),
                               overlap_fraction=eval(xml_tree.find('overlap_fraction').text),
                               filter_window=window,
                               i_input=eval(xml_tree.find('i_input').text),
                               i_output=eval(xml_tree.find('i_output').text),
                               f_range=eval(xml_tree.find('f_range').text),
                               scaling=xml_tree.find('scaling').text,
                               sampling_rate=eval(xml_tree.find('sampling_rate').text)
                                             )
        return conf


class SpectralAnalyzer(AbstractDataProcessor):

    def __init__(self, configuration: SpectralAnalyzerConfiguration):
        AbstractDataProcessor.__init__(self)

        self._event_new_data = EventPublisher(data_type=SpectralSample)
        self._last_analysis = SpectralSample()
        self._config = ArgumentVerifier(SpectralAnalyzerConfiguration, None).verify(configuration)

    @property
    def configuration(self) -> SpectralAnalyzerConfiguration:
        return self._config

    @configuration.setter
    def configuration(self, value: SpectralAnalyzerConfiguration):
        self._config = value

    @property
    def event_new_data(self):
        return self._event_new_data

    def cb_new_data(self, publisher):
        data = publisher.get_data()
        resp = self.analyze(data)
        self._event_new_data.raise_event(resp)

    def analyze(self, data):

        # Verify arguments:
        data = ArgumentVerifier(np.ndarray, data).verify(data)

        # Get settings:
        f_range = self._config.f_range
        i_input = self._config.i_input
        i_output = self._config.i_output
        scaling = self._config.scaling
        csd_args = self._config.get_csd_args(data.shape[0])

        freq = None
        if i_input is not None:
            freq, Sff = sp.signal.csd(data[:, i_input], data[:, i_input], scaling=scaling, **csd_args)
        else:
            Sff = None

        if i_output is not None:
            freq, Sxx = sp.signal.csd(data[:, i_output], data[:, i_output], scaling=scaling, **csd_args)
        else:
            Sxx = None

        if (i_input is not None) and (i_output is not None):
            freq, Sxf = sp.signal.csd(data[:, i_output], data[:, i_input], scaling=scaling, **csd_args)
            _, Cxf = sp.signal.coherence(data[:, i_output], data[:, i_input], **csd_args)
            H = Sxf / Sff
        else:
            Cxf = None
            Sxf = None
            H = None

        # Index:
        ind = np.where((f_range[0] <= freq) & (freq <= f_range[1]))[0]

        # Assign result:
        self._last_analysis = SpectralSample(
                                            freq=freq[ind] if freq is not None else None,
                                            Sff=Sff[ind] if Sff is not None else None,
                                            Sxf=Sxf[ind] if Sxf is not None else None,
                                            Sxx=Sxx[ind] if Sxx is not None else None,
                                            Cxf=Cxf[ind] if Cxf is not None else None,
                                            H=H[ind] if H is not None else None
                                            )
        return self._last_analysis


class SpectralSampleList(AbstractSpectralSample):

    def __init__(self, label=''):
        self._samples = []
        self._std_spectra = None
        self._mean_spectra = None
        self._statistics_outdated = False
        self._label = label

    def __getitem__(self, key):
        """ If type of key is int, key-th item in list, otherwise returns spectral item according to key (i.e. 'H' , 'Sxx' etc)
        :param key: Index or
        :return:
        """
        if isinstance(key, int):
            return self._samples[key]
        elif isinstance(key, str):
            return self.mean[key]
        else:
            raise KeyError("Invalid key")

    def __setitem__(self, index, item):
        if type(item) is SpectralSample:
            self._samples[index] = item
            self._statistics_outdated = True
        else:
            raise TypeError("Item is not of type {0}".format(SpectralSample.__name__))

    def __len__(self):
        return len(self._samples)

    def append(self, item: AbstractSpectralSample):
        """ Append Spectral Sample

        :param item: Spectral sample to add
        :type item: AbstractSpectralSample
        :return: None
        """

        if isinstance(item, AbstractSpectralSample):
            self._samples.append(item)
            self._statistics_outdated = True
        else:
            raise TypeError("Item is not of type {0}".format(SpectralSample.__name__))

    def __iter__(self):
        return iter(self._samples)

    def pop(self, index):
        if isinstance(index, int):
            popval = self._samples.pop(index)
        elif isinstance(index, str):
            # Find by label:
            labels = self.labels()
            if index in labels:
                popval = self.pop(labels.index(index))
            else:
                raise KeyError('Key not found')
        else:
            raise ValueError('Invalid value, provide index (int) or label (str).')

        self._update_statistics()
        return popval

    def remove(self, spectral_sample):
        self._samples.remove(spectral_sample)
        self._update_statistics()

    def __delitem(self, key):
        self.pop(key)

    def _update_statistics(self):
        """Compute statistics on current samples"""
        # [MZ] this implementation can be more efficient by not recomputing on each call

        # Gather data
        data = {}
        for s in self._samples:
            for key in s.keys():
                if key in data.keys():
                    data[key].append(s[key])
                else:
                    data[key] = [s[key]]

        # Compute means:
        for key, item in data.items():
            data[key] = np.hstack([item])

        self._mean_spectra = SpectralSample( **dict([(key, item.mean(axis=0)) for key, item in data.items()]) )
        self._std_spectra = SpectralSample( **dict([(key, item.std(axis=0)) for key, item in data.items()]) )

        self._statistics_outdated = False

    def get(self, key):
        return self.mean[key]

    def keys(self):
        return self.mean.keys()

    def labels(self):
        return [s.label for s in self._samples]

    @property
    def mean(self):
        """The sample mean of loaded power spectral densities"""
        if self._statistics_outdated:
            self._update_statistics()
        return self._mean_spectra

    @property
    def std(self):
        if self._statistics_outdated:
            self._update_statistics()
        return self._std_spectra

    @property
    def Sxx(self):
        """Mean output spectrum of samples in list"""
        return self.mean.Sxx

    @property
    def Sxf(self):
        """Mean cross spectral density of samples in list"""
        return self.mean.Sxf

    @property
    def Sff(self):
        """Mean input spectrum samples in list"""
        return self.mean.Sff

    @property
    def Cxf(self):
        """Mean coherence of samples in list"""
        return self.mean.Cxf

    @property
    def H(self):
        """Mean transfer function of samples in list"""
        return self.mean.H

    @property
    def frequencies(self):
        """Get mean frequencies of samples in list"""
        return self.mean.frequencies

    @property
    def label(self):
        return self._label

    def get_mean(self, key):
        """Returns mean signal"""
        if key in self._mean_spectra.keys():
            return self.mean[key]
        else:
            return None

    def get_std(self, key):
        """Returns standard deviation signal"""
        if key in self._std_spectra.keys():
            return self.std[key]
        else:
            return None


class ModalAnalysis:

    def __init__(self, filename=None, label='', **kwargs):

        self._samples = SpectralSampleList(label=label)
        self._data = []
        self._filenames = []

        if filename is not None:
            self.load_from_files(filename, **kwargs)

    def load_from_files(self, filename, i_input, i_output, sampling_rate=None,
                        window_fraction=0.05, overlap_fraction=0.5, **kwargs):

        # Load info:
        fn_split = re.split('/', filename)
        fdir = ''.join(['{0}/'.format(item) for item in fn_split[:-1]])
        fn_pattern = fn_split[-1]
        fdir = './' if fdir == '' else fdir

        files = [f for f in os.listdir('{0}'.format(fdir)) if re.match(fn_pattern, f)]
        self._filenames = files

        # Get sampling frequency:
        if sampling_rate is None:
            log_info, data = mlog.CSVLogReader().read('{0}/{1}'.format(fdir, files[0]))
            self.sampling_rate = float(log_info['sampling_rate'])
            check_samplingrate = True
        else:
            check_samplingrate = False

        for f in files:
            # Read log:
            _, fname = os.path.split(f)
            label, _ = os.path.splitext(fname)
            log_info, data = mlog.CSVLogReader().read('{0}{1}'.format(fdir, f))

            # Verify sampling rate:
            if check_samplingrate:
                tmp_rate = float(log_info['sampling_rate'])
                if tmp_rate != self.sampling_rate:
                    raise RuntimeWarning('Sampling rates differ among files. {0} vs. {1}'.format(tmp_rate, sampling_rate))

            # Add frf to dict:
            self._data.append(data)

            # Window settings:
            nperseg = int(data.shape[0]*window_fraction)
            noverlap = int(nperseg*overlap_fraction)

            # Compute FRF
            sample = SpectralSample(label=label).compute_from_data(data=data, i_input=i_input, i_output=i_output,
                                                                   fs=self.sampling_rate, nperseg=nperseg,
                                                                   noverlap=noverlap, **kwargs)
            self._samples.append(sample)


    @property
    def spectral_data(self):
        return self._samples

    @property
    def label(self):
        return self._samples.label

    def plot_signal(self, i_signal, ax=None, plot_args=None):
        """Plot raw data of provided signal"""

        plot_args = {} if plot_args is None else plot_args

        if ax is None:
            fig = Figure()
            ax = fig.gca()
            ax.set_xlabel('# Data point')

        for i, d in enumerate(self.rawdata):
            ax.plot(d[:, i_signal], label=i, **plot_args)

    def plot_FRF(self, **kwargs):
        """Plot the frequency response of the modal data"""
        return mplot.SpectralPlot(spectral_data=self._samples, **kwargs)

    @property
    def rawdata(self):
        return self._data

    @property
    def input_spectrum(self):
        return self._samples.mean['Sff']

    @property
    def output_spectrum(self):
        return self._samples.mean['Sxx']

    @property
    def transfer_function(self):
        return self._samples.mean['H']

    @property
    def coherence(self):
        return self._samples.mean['Cxf']

    @property
    def frequencies(self):
        return self._samples.mean['freq']


def compute_modes(file_info, signal_settings=None, window_settings=None,
                  widths=range(5, 10), fdir='./data', freqs=None):
    """ Compute modal shapes for the provide files
    :param file_info: list of tuples containing a position value, and a filename pattern used to select the data files
    :param signal_settings: dictionary containing input and output indices
    :param window_settings: dictionary containing the window size and window overlap fractions
    :param widths: A range used for computing eigen frequencies
    :param fdir :file directory of the data files
    :returns mode_list, base_ma
    """

    signal_settings = ArgumentVerifier(dict, None).verify(signal_settings)
    window_settings = ArgumentVerifier(dict, None).verify(window_settings)

    # [mz] Old, can be removed?
    base_ma = ModalAnalysis(filename='{0}/{1}'.format(fdir, file_info[0][0]),
                            **signal_settings, **window_settings)

    # Collect mode information:
    response_data = SpectralSampleList()
    for i_pos, (pattern, _) in enumerate(file_info):
        ma = ModalAnalysis('{0}/{1}'.format(fdir, pattern),
                           **signal_settings, **window_settings)
        response_data.append(ma.spectral_data.mean)

    # Get poles:
    sel_poles = SimplePoleFinder().find(response_data.mean)

    # Prepare structure
    mode_list = ModeList()
    locations = np.array([pos for (file, pos) in file_info])
    for pole in sel_poles:
        # Compute mode frequency and damping:
        f_n = pole.imag/(np.pi*2)
        zeta = pole.real/pole.imag

        # Create mode object & and estimate mode:
        m = Mode(natural_frequency=f_n, damping_ratio=zeta)
        m.node_locations = locations
        m.estimate_shape(response_data)
        mode_list.append(m)

    return mode_list, base_ma


""" Parameter Estimation


"""


class Omegaj(object):
    """ Basis for polynomial
    """

    def get_at_order(self, omega, order):
        """Returns the j-th order of transfer basis"""
        raise NotImplementedError()


class DiscreteOmegaj(Omegaj):

    def __init__(self, ts):
        """Basis function for discrete polynomial with time-step ts
        :param ts: Time step [s]
        """
        self._ts = ts

    def get_at_order(self, omega, order):
        """
        :param omega: frequency [rad/s]
        :param order: order of exponent
        """
        return np.exp(-1j * omega * self._ts * order)


class ContinuousOmegaj(Omegaj):

    def __init__(self):
        """Basis function for continuous polynomial
        """
        pass

    def get_at_order(self, omega, order):
        """
        :param omega: frequency [rad/s]
        :param order: order of exponent
        """
        return (1j * omega)**order


class LeastSquaresSolver(object):

    def __init__(self, A, b):
        raise NotImplementedError()

    def solve(self):
        raise NotImplementedError()


class HomogeneousLSQSolver(LeastSquaresSolver):

    def __init__(self, A, b=None):
        """Solver for homogeneous least-squares problems of the form Ax=0"""
        self._A = A

        self._S = None
        self._U = None
        self.VH = None

    def solve(self):
        """Solve homogeneous least squares """
        (U, S, VH) = sp.linalg.svd(self._A)

        self.S = S
        self.U = U
        self.VH = VH

        # Sort values:
        ind = np.argsort(S)

        # Note that symp.linalg.svd returns V transposed,
        # so we take the row corresponding to the smallest eigen value
        return VH[ind[0],:]


class CommonDenominatorModel(object):
    """ In modal analysis we try to estimate a parameterized transfer function from experimental data. Experiemntal data
    consist of a measured excitation and responses on different input and/or output positions.
    Although one could estimate the parameters of a the mechanical transfer function 1/(Ms^2 + Cs + K) for each separately,
    it turns out to be more convenient to assume a common denominator model, in which the all response functions are assumed
    to share a (common) denominator and differ only in their numerator. i.e. measurement k has the following transfer
    function:

    H_k = N_k(s)/d(s)

    where
    N_k(s) = b_k,j*s^j + b_(k,j-1)*s^(j-1) ...
    d(s) = a_j*s^j + a_(j-1)*s^(j-1) ...

    are the numerators represented as polynomials.

    This class represents the common denominator model in terms of the polynomial coefficients

    """

    def __init__(self, A, B):
        """Common denominator model for Modal analysis"""
        self._A = A
        self._B = B
        self._omega_j = ContinuousOmegaj()

    @property
    def order(self):
        """

        :return: The order of the model
        """
        return len(self._A)

    @property
    def omega_j(self):
        """

        :return: The polynomial variable function Omega_j
        """
        return self._omega_j

    @omega_j.setter
    def omega_j(self, value):

        self._omega_j = value

    @property
    def B(self):
        """
        :return: The numerator coefficients of the common denominator model
        """
        return self._B

    @property
    def A(self):
        """

        :return: The common denominator coefficients
        """
        return self._A

    def frequency_response(self, frequencies, node_id=None):
        """
        :param frequencies: list of frequencies for which to evaluate the transfer function [rad/s]
        :param node_id: [optional] specific id for which to evaluate evaluate the transfer function
        """

        if type(node_id) is int:
            return self._freq_response_at_node(frequencies, node_id)
        elif type(node_id) is list:
            # Evaluate for
            H = []
            for n_id in self._B.shape[0]:
                H.append(self.frequency_response(frequencies, n_id))

        return H

    def _freq_response_at_node(self, frequencies, node_id):
        """Evaluate the frequency response of the common denominator model at given frequencies"""
        tf = sp.signal.TransferFunction(self.B[node_id], self.A)
        _, H = tf.freqresp(frequencies)
        return H


class DiscreteCommonDenominatorModel(CommonDenominatorModel):

    def __init__(self, A, B, dt):
        CommonDenominatorModel.__init__(self, A, B)
        self._dt = dt
        self.omega_j = DiscreteOmegaj(dt)

    @property
    def dt(self):
        """

        :return: The time step of the model
        """
        return self._dt

    def _freq_response_at_node(self, frequencies, node_id):
        """
        Get discrete transferfunction
        frequencies [rad/s]
        """
        num = 0
        for j, b in enumerate(self._B[node_id,:]):
            num += self._omega_j.get_at_order(frequencies, j) * b

        den = 0
        for j, a in enumerate(self._A):
            den += self._omega_j.get_at_order(frequencies, j) * a

        H = num / den

        return H


class CDMEstimator(object):

    def __init__(self, order):
        """

        :param order: Order of the model to estimate
        """
        self._order = order
        self._model = CommonDenominatorModel(None, None)

    @property
    def model(self):
        """

        :return The most recently estimated common denominator model
        """
        return self._model

    @property
    def order(self):
        """

        :return: The order of the model to estimate
        """
        return self._order

    def _update_model(self, A, B):
        raise NotImplementedError()

    def estimate(self, spectral_data_list, W=None, lamb=0):
        """

        :param spectral_data_list: A list of spectral samples
        :type spectral_data_list: SpectralSampleList
        :param W: [optional] vector of weights to apply to each frequency
        :param lamb: [optional] regularization factor used by the solver(implemented as Tikhonov regularization)
        :return: the estimated model
        :rtype: CommonDenominatorModel
        """

        # Construct Xk and Yk:
        Xvalues = []
        Yvalues = []

        for spectral_sample in spectral_data_list:
            Xvalues.append(self.construct_Xk(spectral_sample, W, self._order))
            Yvalues.append(self.construct_Yk(spectral_sample, W, self._order))

        J = sp.linalg.block_diag(*Xvalues)
        Yvalues = np.vstack(Yvalues)
        J = np.hstack([J, Yvalues])

        # Solve (damped) LSQR
        solver = HomogeneousLSQSolver(J.conj().T.dot(J).real + lamb*np.eye(J.shape[1]))
        x = solver.solve()

        # reshape solution:
        x = x.reshape(-1, self._order+1)

        # Reverse order of beta and alpha such that we have the highest function order first:
        # [b0*s^0, b1*s^1  ... bn*s^n] --> [bn*s^n, .... b1*s^1, b0*s^0]
        # This makes it consistent with the polynomial functions commonly used in tranferfunctions
        B = x[:-1,::-1]
        A = x[-1,::-1]

        details = {'J': J,
                   'x': x,
                   'solver': solver
                  }

        self._update_model(A, B)

        return self.model

    def get_tfs(self):
        raise NotImplementedError()

    def construct_Xk(self, spectral_sample, W=None, order=5):
        """
        :param spectral_sample: Spectral sample
        :type spectral_sample: SpectralSample
        """

        frequencies = spectral_sample['freq']*2*np.pi # Convert from [Hz] to [rad/s]
        omega_j = self._model.omega_j

        # Get weights:
        if W is None:
            W = np.ones(len(frequencies))/len(frequencies)

        # Compute elements:
        X = []
        for i, freq in enumerate(frequencies):
            X.append(W[i]*omega_j.get_at_order(omega=freq, order=np.arange(order+1)))
        X = np.vstack(X)

        return X

    def construct_Yk(self, spectral_sample, W=None, order=5):
        """
        :param spectral_sample: Spectral sample

        """

        frequencies = spectral_sample['freq']*np.pi*2 # Convert from [Hz] to [rad/s]
        H = spectral_sample['H']
        omega_j = self._model.omega_j

        # Get weights:
        if W is None:
            W = np.ones(len(frequencies))/len(frequencies)

        # Compute elements:
        Y = []
        for i, freq in enumerate(frequencies):
            try:
                Y.append(-W[i]*H[i]*omega_j.get_at_order(omega=freq, order=np.arange(order+1)))
            except:
                Y.append(-W[i]*H[i]*omega_j.get_at_order(omega=freq, order=np.arange(order+1)))
        Y = np.vstack(Y)
        return Y


class DiscreteCDMEstimator(CDMEstimator):

    def __init__(self, order, dt):
        CDMEstimator.__init__(self, order)
        self._dt = dt
        self._model = DiscreteCommonDenominatorModel(None, None, dt)

    def _update_model(self, A, B):
        self._model = DiscreteCommonDenominatorModel(A, B, self._dt)

    @property
    def dt(self):
        return self._dt


class ContinuousCDMEstimator(CDMEstimator):

    def __init__(self, order):
        CDMEstimator.__init__(self, order)
        self._order = order

    def _update_model(self, A, B):
        self._model = CommonDenominatorModel(A, B)


class ModeEstimator(object):

    def estimate(self, response_data, **kwargs):
        raise NotImplementedError()


class PhysicalPoleFinder(object):

    def find(self, response):
        """Identify the Physical poles for a given response
        returns: list of physical poles
        """
        raise NotImplementedError()


class DiscretePoleFinder(PhysicalPoleFinder):

    def __init__(self, dt, order_list=None, frequency_range=None, reg_lambda=0.0, multi_thread=True):
        """ Class to estimate poles from frequency response data.
        :param dt: sample interval
        :param order_list: List of model orders used to identify physical poles, defaults to np.arange(40,81,5)
        :param frequency_range: Frequency range in which to search for poles, defaults to [0, 2000]
        :param reg_lambda: Regularization term used in parameter estimation
        :param multi_thread: Use a separate thread for each model order estimate (faster at the cost of memory and cpu consumption)
        """
        self._dt = ArgumentVerifier(float, dt).verify(dt)
        self._order_list = ArgumentVerifier(np.ndarray, np.arange(40,81,5)).verify(order_list)
        self._frange     = ArgumentVerifier(list, [0, 2500]).verify(frequency_range)
        self._lambda     = ArgumentVerifier(float, reg_lambda).verify(reg_lambda)
        self._multi_thread = ArgumentVerifier(bool, multi_thread).verify(multi_thread)
        self._response_data = None

    def find(self, response_data, dist_threshold=None, min_neighbors=None):
        """ Find physical poles by finding the roots for common denominator models of different order.
        The non-changing roots represent physical poles, and changing roots represent mathematical poles.

        :param response_data: List of spectral response data
        :rtype response_data: SpectralSampleList
        :return: List of physical poles
        """
        # Register settings for estimation:
        if issubclass(SpectralSampleList, type(response_data)) and not (response_data is self._response_data):
            # (Re)compute:
            self._response_data = response_data

            # Collect poles for different model orders:
            self._order_res = []
            self._thrds = []
            for i, order in enumerate(self._order_list):
                self._order_res.append(None)  # Make space to save results
                if self._multi_thread:
                    # Compute orders in seperate threads:
                    self._thrds.append(thrd.Thread(target=self._get_poles_for_order, args=[order,i]))
                    self._thrds[-1].start()
                else:
                    self._get_poles_for_order(order, save_index=i) # Call

            # Wait threads to finish:
            for thread in self._thrds:
                thread.join()

        # Compute non-changing poles:
        sel_poles, pole_neighbors = self.get_physical_poles(dist_threshold, min_neighbors)

        return sel_poles

    def _get_poles_for_order(self, order, save_index):
        """Compute poles for a given model order

        :param order:  Order of the model
        :param save_index: Index in self._order_res to save the results (is implemented for multi-threading purpose)
        :return:
        """

        # Estimate model parameters of a common denominator model
        d_est = DiscreteCDMEstimator(order, self._dt)
        d_est.estimate(self._response_data, lamb=self._lambda)

        # Convert discrete poles into continuous
        #d_poles = d_est.get_tfs()[0].poles
        d_poles = np.roots(d_est.model.A) # Discrete poles are the roots of A
        c_poles = np.log(d_poles)/self._dt

        # Select poles within frequency range
        ind = np.where( (self._frange[0] <= np.abs(c_poles.imag)/(2*np.pi)) &
                                           (np.abs(c_poles.imag)/(2*np.pi) <=self._frange[1])
                      )[0]
        self._order_res[save_index] = c_poles[ind]

        #print('Finished computing order {0}'.format(order))

    def get_physical_poles(self, dist_threshold=None, min_neighbors=None):
        """ Filter physical poles from estimated model orders based on n-neirest neighbors

        :param dist_threshold: Distance threshold for neighbor classification [Hz]
        :param min_neighbors: Minimum number of neighbors required to recognize a pole as physical pole
        :return: tuple consisting of selected poles, and neigbhor results
        """

        # Check arguments
        min_neighbors = ArgumentVerifier(int, int(len(self._order_list)/4)).verify(min_neighbors)
        dist_threshold = ArgumentVerifier(float, 5.0).verify(dist_threshold)

        if self._order_res is None:
            raise RuntimeError("No response data has been analyzed yet, firt call '.find' on this object")

        # Check distance between mode frequencies:
        ref_order = self._order_res[-1]
        pole_neighbors = []
        for item in ref_order:
            w = item.imag # Frequency [rad/s]
            neighbor_list = []

            # find closest frequency for each other order:
            for poles in self._order_res[:-1]:
                # Find closest distance to f
                diff = np.abs(np.array([pole.imag for pole in poles]) - w).min()

                # Check if it lies within desired range:
                if diff < dist_threshold*2*np.pi:
                    ind = np.argmin(np.abs(np.array([pole.imag for pole in poles]) - w))
                    # Add to list
                    neighbor_list.append(poles[ind])

            # Store results:
            pole_neighbors.append((item, neighbor_list))

        # Select poles that have atleast n_neighbors:
        sel_poles = [np.array(neighbor_list).mean() for (item, neighbor_list) in pole_neighbors
                     if (len(neighbor_list) >= min_neighbors)]

        return sel_poles, pole_neighbors


class SimplePoleFinder(PhysicalPoleFinder):

    def __init__(self, cwt_kwargs=None):

        # Default settings for cwt_peak finder
        widths = np.arange(0.1, 1, 0.1)
        default = {'widths': widths,
                   'min_length': len(widths)/4,
                   'gap_thresh': 2,
                   'max_distances': widths/4,
                   'noise_perc': 10,
                   'min_snr': 2
                   }

        self._cwt_kwargs = ArgumentVerifier({}.__class__, default).verify(cwt_kwargs)

    def find(self, spectral_sample):

        # Find poles using cwt:
        fn_list = self.get_natural_frequencies(spectral_sample)

        # Compute damping
        pole_list = []
        for fn in fn_list:
            zeta = SimplePoleFinder.get_damping(fn, spectral_sample)
            if zeta is not None:
                w_n = fn*np.pi*2
                pole_list.append(-zeta*w_n + 1j*w_n)
                pole_list.append(-zeta*w_n - 1j*w_n)

        return pole_list

    @staticmethod
    def get_damping(f_n, spectral_sample):

        #frequency index:
        H_abs = 20*np.log10(np.abs(spectral_sample.H))
        ind = np.argsort(np.abs(spectral_sample.frequencies - f_n))[0]
        H_fn = H_abs[ind]
        H_des = H_fn/np.sqrt(2)

        # Find half powerpoint before
        try:
            ind_low = np.where(H_abs[:ind] <= H_des)[0][-1]
            ind_up = np.where(H_abs[ind:] <= H_des)[0][0]+ind

            f1 = spectral_sample.frequencies[ind_low]
            f2 = spectral_sample.frequencies[ind_up]

            damping_ratio = (f2-f1)/(2*f_n)
        except:
            damping_ratio = None

        return damping_ratio

    def get_natural_frequencies(self, spectral_sample):
        """Returns an array of frequencies at which the transfer function peaks.
        kwargs is passed to symp.signal.find_peaks_cwt(...), and allows one to change peak selection criteria.
        The the function's manual for available arguments
        """
        ind_peaks = sp.signal.find_peaks_cwt(np.abs(spectral_sample.H), **self._cwt_kwargs)
        return spectral_sample.frequencies[ind_peaks]


class ButterWorthFilter(mbb.TraitsXMLSerializer, traitlets.TraitType):

    enabled = traitlets.Bool(default_value=True, help='Filter enabled (when not enabled, filter action is not applied')
    order = traitlets.CInt(default_value=1, help='Filter order')
    fs = traitlets.CFloat(help='Sampling frequency (optional)', default_value=1.0)
    frequency = traitlets.TraitType(default_value=0.0, help='Critical frequency')
    filter_type = traitlets.CUnicode(default_value='lowpass', help='Filter type: lowpass, highpass, bandpass or bandstop')
    filter_method = traitlets.CUnicode(default_value='gust', help='Either \'gust\' or \'pad\'')

    def __init__(self, frequency, fs, order, type, method='pad', enabled=True):
        """ Configures a Butterworth filter

        :param freq: (list of) frequencies to use in filtering [Hz]
        :param fs: sampling frequency [Hz]
        :param order: Filter order
        :param type: Filter type
        :return: Butterworth filter
        """
        super().__init__(fs=fs, frequency=frequency, order=order, filter_type=type, filter_method=method,
                         enabled=enabled)

        self._filter = sp.signal.butter(N=order, fs=fs, Wn=frequency, btype=type)

    @traitlets.observe('order', 'fs', 'frequency', 'filter_type')
    def _params_change(self, change):
        self._filter = sp.signal.butter(N=self.order, fs=self.fs, Wn=self.frequency, btype=self.filter_type)

    def filter(self, data, axis=-1):
        """ Filter data

        :param data:
        :param axis: Axis over which to filter data
        :return:
        """
        if self.enabled:
            return sp.signal.filtfilt(*self._filter, data, method=self.filter_method, axis=axis)
        else:
            return np.array(data)


class BandPassFilter(ButterWorthFilter):
    low_cut = traitlets.CFloat(default=0.0, help='Lowpass frequency')
    high_cut = traitlets.CFloat(default=0.0, help='Highpass frequency')

    def __init__(self, low_cut, high_cut, fs, order):
        """ Configures a Butterworth Bandpass filter

        :param low_cut: Low cutt-off frequency [Hz]
        :param high_cut: high cutt-off frequency [Hz]
        :param fs: sampling frequency [Hz]
        :param order: Filter order
        :return: Butterworth filter
        """

        low = low_cut
        high = high_cut

        ButterWorthFilter.__init__(self, [low, high], fs=fs, order=order, type='bandpass')
        self.low_cut=low_cut
        self.high_cut=high_cut

        self._var_names_mapping = [('low_cut', 'low_cut'), ('high_cut', 'high_cut'),
                                   ('fs', 'fs'), ('order', 'order'), ('type', 'filter_type')]

    @traitlets.observe('low_cut', 'high_cut')
    def _cut_change(self, change):
        low = self.low_cut #/ nyq
        high = self.high_cut #/ nyq

        if low < high:
            self.frequency = [low, high]


class LowPassFilter(ButterWorthFilter, traitlets.HasTraits):
    low_pass_frequency = traitlets.CFloat(default_value=0, help='Lowpass frequency')

    def __init__(self, low_pass_frequency, fs, order, enabled=True, *args, **kwargs):
        """ Configures a Butterworth Bandpass filter

        :param low_pass_frequency: Low cutt-off frequency [Hz]
        :param fs: sampling frequency [Hz]
        :param order: Filter order
        :return: Butterworth filter
        """

        super().__init__(frequency=low_pass_frequency, fs=fs, order=order, type='lowpass',
                         enabled=enabled, *args, **kwargs)
        self.low_pass_frequency=low_pass_frequency
        self._var_names_mapping = [('low_pass_frequency', 'low_pass_frequency'),
                                   ('fs', 'fs'),
                                   ('order', 'order'),
                                   ('enabled', 'enabled')]

    # Note that we do not respond correctly to changes in frequency
    @traitlets.observe('low_pass_frequency')
    def _lp_change(self, change):
        self.frequency = change['new']

    @traitlets.validate('low_pass_frequency')
    def _lp_validate(self, proposal):
        if proposal['value'] < self.fs/2:
            return proposal['value']
        else:
            raise traitlets.TraitError('Lowpass frequency should lie in range 0 < Wn < {} (fs/2)'.format(self.fs/2) +
                                      ', received {}'.format(proposal['value']))

    @traitlets.validate('frequency')
    def _lp_validate(self, proposal):
        if proposal['value'] < self.fs/2:
            return proposal['value']
        else:
            raise traitlets.TraitError('frequency should lie in range 0 < Wn < {} (fs/2)'.format(self.fs/2) +
                                      ', received {}'.format(proposal['value']))

    @traitlets.validate('fs')
    def _fs_validate(self, proposal):
        if proposal['value']/2 > self.low_pass_frequency:
            return proposal['value']
        else:
            raise traitlets.TraitError(
                'Sampling frequency should be at least 2 x lowpass frequency: {} !< 2 x {}'.format(proposal['value'],
                                                                                                   self.low_pass_frequency))

    @traitlets.validate('order')
    def _order_validate(self, proposal):
        if proposal['value'] > 0 and isinstance(proposal['value'], int):
            return proposal['value']
        else:
            raise traitlets.TraitError('Order should be integer and larger than 0, received: {}'.format(proposal['value']))


class HighPassFilter(ButterWorthFilter):
    high_pass_frequency=traitlets.CFloat(default_value=0.0, help='Highpass frequency')

    def __init__(self, high_pass_frequency, fs, order):
        """ Configures a Butterworth Bandpass filter

        :param high_pass_frequency: high pass cutt-off frequency [Hz]
        :param fs: sampling frequency [Hz]
        :param order: Filter order
        :return: Butterworth filter
        """

        high = high_pass_frequency

        super().__init__(frequency=high, fs=fs, order=order, type='highpass')
        self.high_pass_frequency = high_pass_frequency

    @traitlets.observe('high_pass_frequency')
    def _lp_change(self, change):
        #nyq = 0.5 * self.fs
        #high = change['new'] / nyq

        #self.frequency = high
        self.frequency = change['new']


class SignalEnvelope(object):

    def __init__(self, fs, signal_filter=None):
        """ Class that provides the ability compute Signal envelope and the envolope spectrum

        @param fs: Data sampling frequency
        @param signal_filter: Bandpass filter
        """
        self._fs = fs
        if signal_filter is None:
            self._filter = BandPassFilter(fs=fs, low_cut=10, high_cut=fs / 2, order=4)
        else:
            self._filter = signal_filter

    def envelope(self, data):
        """ Returns the signal envelop

        @param data:
        @return:
        """
        f_data = self._filter.filter(data)
        return np.abs(sp.signal.hilbert(f_data))

    def spectrum(self, data, df):
        # Compute number of points:
        N = int(self._fs / df)

        # Get envelope data:
        env_data = self.envelope(data)

        # Perform csd on signal:
        f, mag = sp.signal.csd(env_data, env_data, fs=self._fs, window='hann', nperseg=N)

        return f, mag


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    lp_filt = LowPassFilter(low_pass_frequency=0.1, fs=1.0, order=3)
    lp_filt.enabled=True

    print('Testing... ')


