import traitlets

import matplotlib.lines as lines
from matplotlib.figure import Figure

import numpy as np
import scipy as sp
import scipy.signal
import madernpytools.log as mlog
import madernpytools.plot as mplt
import madernpytools.tools.utilities as mutils
import madernpytools.backbone as mbb
import madernpytools.tools.frequency_response as mfrf


class ClassFactory(mbb.IClassFactory):
    """ Factory class which allows to generate class instances from this module

    """

    @staticmethod
    def get(name):
        return eval(name)


class LineProfile(traitlets.TraitType):

    def __init__(self, x=None, z=None, time=None, **kwargs):
        super().__init__()
        self._dict = dict([('x', x), ('z', z), ('time', time)], **kwargs)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()

    def pop(self, key):
        if key in self._dict and not key in ['x', 'z', 'time']:
            self._dict.pop(key)

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return self._dict.__iter__()

    def __getitem__(self, item):
        if isinstance(item, str):
            # Standard key request:
            return self._dict[item]
        elif isinstance(item, (slice, list, np.ndarray)):
            # We received an index request,
            selection = item
            sub_profile = LineProfile()
            for key, item in self._dict.items():
                # if item can be indexed, index:
                if isinstance(item, (np.ndarray, list)):
                    sub_profile[key] = item[selection]
                else:
                    sub_profile[key] = item
            return sub_profile

    @property
    def x(self) -> np.ndarray:
        return self['x']

    @property
    def time(self) -> float:
        return self['time']

    @property
    def z(self) -> np.ndarray:
        return self['z']


class KeyenceProfileLoader(object):

    @staticmethod
    def load(filename, filter_data=True, dx=None):
        # Load profile csv:
        loginfo, data = mlog.CSVLogReader().read(filename)

        # Create x-values:
        if dx is None:
            dx = float(loginfo['Lens calibration'])
            dx = dx if dx>0.0 else 0.21
        x = np.linspace(0, data.shape[0] * dx, data.shape[0], endpoint=True)[:, None]

        data = np.hstack([x, data])
        return data


def derive_dxdz(x, z, axis=-1, regularization=1e-10):
    """ Compute first and second derivate of dz/dx and d2z/dx2

    :param x: n-dimensional X-data
    :param z: n-dimensionalZ-data
    :param axis: Axis among which to perform derivative
    :param regularization: regularization value, if dx is below this value derivative is ignored
    :return: first and second derivative

    """
    # Gradients:
    dz = np.gradient(z, axis=axis)
    d2z = np.gradient(dz, axis=axis)
    dx = np.gradient(x, axis=axis)

    # Allocate memory and set to nan
    dzdx = np.zeros(dz.shape)*np.nan
    d2zdx2 = np.zeros(dz.shape)*np.nan

    # Derivatives:
    i_dx_nz = dx > regularization  # Non-zero elements of dx
    dzdx[i_dx_nz] = dz[i_dx_nz] / dx[i_dx_nz]
    d2zdx2[i_dx_nz] = d2z[i_dx_nz] / dx[i_dx_nz] ** 2

    return dzdx, d2zdx2


class LinePeak(LineProfile):

    def __init__(self, x, z, dzdx=None, d2zdx2=None, peak_index=None, f_height=np.median, **kwargs):
        super().__init__(x=x, z=z, dzdx=dzdx, d2zdx2=d2zdx2, peak_index=peak_index, **kwargs)
        self._fheight = f_height

        if x is not None:
            self['x_peak'] = self.peak_x
            self['width'] = self.width
        if z is not None:
            self['height'] = self.height
            self['z_peak'] = self.peak_z

    @property
    def dzdx(self):
        return self['dzdx']

    @property
    def d2zdx2(self):
        return self['d2zdx2']

    @property
    def peak_index(self):
        return self['peak_index']

    @property
    def peak_x(self):
        return self['x'][self.peak_index]

    @property
    def peak_z(self):
        return self['z'][self.peak_index]

    @property
    def width(self):
        return self.x[self.peak_index[-1]] - self.x[self.peak_index[0]]

    @property
    def height(self):
        return self._fheight(self.z[self.peak_index])


class AnalyzerThresholds(mbb.TraitsXMLSerializer, traitlets.TraitType):
    z = traitlets.CFloat(help='Allotted peak deviation')
    dzdx = traitlets.CFloat(help='Peak derivative threshold')
    d2zdx2 = traitlets.CFloat(help='Peak 2th derivative threshold')

    def __init__(self, z=0.0, dzdx=0.0, d2zdx2=0.0):
        """

        @param z:  Height threshold
        @param dzdx: z
        @param d2zdx2:
        """
        super().__init__(z=z, dzdx=dzdx, d2zdx2=d2zdx2)

    def __getitem__(self, key):
        return self._trait_values[key]

    def keys(self):
        return self._trait_values.keys()


class RotationRectifier(object):

    def __init__(self):
        # Filter settings
        b, a = sp.signal.butter(N=2,
                                Wn=0.05,
                                fs=1)
        self._filter_settings = {'a': a, 'b': b}

    def get_rise_and_fall(self, x, z):
        dzdx, d2zdx2 = derive_dxdz(x, z)

        i_rise = np.where(dzdx > (dzdx.max() - 1))[0]
        i_fall = np.where(dzdx < (dzdx.min() + 1))[0]

        m_rise = np.median(dzdx[i_rise])
        m_fall = np.median(dzdx[i_fall])

        return m_rise, m_fall

    def filter_profile(self, x, z):
        z_filt = sp.signal.filtfilt(self._filter_settings['b'],
                                    self._filter_settings['a'], z)
        x_filt = sp.signal.filtfilt(self._filter_settings['b'],
                                    self._filter_settings['a'], x)
        return x_filt, z_filt

    def rotate_profile(self, x, z, angle):
        # Define rotation matrix:
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])

        # Stack data and apply rotation
        tmp = np.vstack([x, z]).T
        tmp_mean = tmp.mean(axis=0)
        tmp = (tmp - tmp_mean).dot(R.T) + tmp_mean

        # Return
        return tmp[:, 0], tmp[:, 1]

    def correct_profile(self, x, z):

        #rise, fall = self.get_rise_and_fall(*self.filter_profile(x, z))
        rise, fall = self.get_rise_and_fall(x, z)
        angle = - (np.arctan(rise) + np.arctan(fall)) * 0.5

        x, z = self.rotate_profile(x, z, angle)

        return x, z


class PeakAnalyzerInterface(mbb.TraitsXMLSerializer):
    thresholds = AnalyzerThresholds()

    def analyze(self, profile: LineProfile, apply_filter=True):
        raise NotImplementedError()


class LinePeakAnalyzer(PeakAnalyzerInterface):
    thresholds = AnalyzerThresholds()

    def __init__(self, thresholds: AnalyzerThresholds=None, f_peak_height=np.max,
                 filter_N=4, filter_Wn=0.05, filter_fs=1):
        """
        :param dzdx_thr: Profile Velocity threshold for peak selection  (velocity below this value)
        :param d2zdx2_thr: Profile acceleration threshold for peak selection (acceleration below this value)
        :param z_fr: Height fraction threshold for peak
        :param filter_N: Filter order for profile filter  (see SciPi signal Butter for details)
        :param filter_Wn:
        :param filter_fs:
        """

        if thresholds is None:
            thresholds = AnalyzerThresholds(z=0.9, dzdx=0.5, d2zdx2=0.1)

        super().__init__(thresholds=thresholds,
                         varnames_mapping=[('thresholds', 'thresholds'),
                                           ('filter_N', 'filter_N'),
                                           ('filter_Wn', 'filter_Wn'),
                                           ('filter_fs', 'filter_fs')
                                           ]
                         )

        b, a = sp.signal.butter(N=filter_N,
                                Wn=filter_Wn,
                                fs=filter_fs)

        self._filter_settings = {'a': a, 'b': b}
        self._f_peak_height = f_peak_height

    def analyze(self, profile: LineProfile, apply_filter=True):
        """

        :param profile: Profile to analyze
        :param apply_filter: Indicate if additiona low-pass filter should be applied to the profile data
        :return:
        """
        # TODO: If the profile is slightly rotated, the velocity and acceleration profiles do not cross 'zero' at
        #  the cutting line peak. As a result, threshold fine-tuning on velocity and acceleration is hard. To resolve this
        #  an additional processing step is required: rotation correction before computation of the  velocity/acceleration
        #  profiles. This step has already been done for Keyence profile analysis, which corrects the rotation by assuming
        #  The cutting line flanks should exactly mirror (i.e. their angle is negated)

        # Apply low-pass filter on data if requested
        if apply_filter:
            z_filt = sp.signal.filtfilt(self._filter_settings['b'],
                                        self._filter_settings['a'], profile['z'])
            x_filt = sp.signal.filtfilt(self._filter_settings['b'],
                                        self._filter_settings['a'], profile['x'])
        else:
            x_filt = profile['x']
            z_filt = profile['z']

        # Compute derivatives:
        dzdx, d2zdx2 = derive_dxdz(x_filt, z_filt)

        # Select peak, based on data
        dzdx_thr = self.thresholds['dzdx']
        d2zdx2_thr = self.thresholds['d2zdx2']
        z_fr = self.thresholds['z']
        if not np.isnan(dzdx).any():
            sel_ind = np.logical_and.reduce((np.abs(dzdx) < dzdx_thr,
                                             np.abs(d2zdx2) < d2zdx2_thr,
                                             z_filt / (z_filt.max() + 1e-10) > z_fr,
                                             )
                                            )
        else:
            sel_ind = []

        peak_ind = np.where(sel_ind)[0]
        if len(peak_ind) > 0:
            return LinePeak(x_filt, z_filt, dzdx=dzdx, d2zdx2=d2zdx2, peak_index=peak_ind, f_height=self._f_peak_height,
                            # Maintain existing info but keys filled in by this method:
                            **{key: profile[key] for key in profile.keys()
                               if key not in ['x', 'z', 'dzdx', 'd2zdx2', 'peak_index', 'f_height']}
                            )

        else:
            return None


class LinePeakAnalyzer2(PeakAnalyzerInterface):
    filter = mfrf.LowPassFilter(fs=1.0, order=2, low_pass_frequency=0.1)
    thresholds = traitlets.TraitType()

    def __init__(self, thresholds: AnalyzerThresholds = None,
                 filter: mfrf.LowPassFilter=None, f_peak_height=np.max):
        """
        Defines the line peak based on the deviation from the heighest point on the line profile

                  _ _ _ _  _  Estimated peak area
                 |   ____   |  ________
                 |  /     \ |       |   Height threshold
                 |/_ _ _ _\ |  _____v_
                 /         \
                /           \

        @param thresholds: Analyzer thresholds object, only the z-value is used as the height-threshold for peak detection
        @param filter_N: Filter order for profile filter  (see SciPi signal Butter for details)
        @param filter_Wn:
        @param filter_fs:
        """

        # Use default thresholds if not specified:
        if thresholds is None:
            thresholds = AnalyzerThresholds(z=0.01, dzdx=1.5, d2zdx2=100.0)

        if filter is None:
            filter = mfrf.LowPassFilter(fs=1.0, low_pass_frequency=0.1, order=2)

        # Initalize base classes:
        super().__init__(thresholds=thresholds, filter=filter,
                         var_names_mapping=[('thresholds', 'thresholds'), ('filter', 'filter')])

        self._f_peak_height=f_peak_height

    def analyze(self, profile: LineProfile, apply_filter=True):
        """ Perform peak analysis for give profile

        :param profile: Profile to analyze
        :param apply_filter: Indicate if additiona low-pass filter should be applied to the profile data
        :return:
        """

        # Get derivatives:
        if 'dzdx' in profile.keys() and 'd2zdx2' in profile.keys():
            dzdx = profile['dzdx']
            d2zdx2 = profile['d2zdx2']
            x_filt = profile['x']
            z_filt = profile['z']
        else:
            if apply_filter:
                z_filt = self.filter.filter(profile['z'])
                x_filt = self.filter.filter(profile['x'])
            else:
                x_filt = profile['x']
                z_filt = profile['z']

            dzdx, d2zdx2 = derive_dxdz(x_filt, z_filt)

        dzdx_thr = self.thresholds['dzdx']
        d2zdx2_thr = self.thresholds['d2zdx2']
        z_thr = self.thresholds['z']

        # Find highest point:
        z_max = z_filt.max()  # Peak value

        if not np.isnan(dzdx).any():
            sel_ind = np.logical_and.reduce((np.abs(dzdx) < dzdx_thr,
                                             np.abs(d2zdx2) < d2zdx2_thr,
                                             z_filt > (z_max - z_thr)
                                             )
                                            )
        else:
            sel_ind = []

        peak_ind = np.where(sel_ind)[0]

        # Return peak index, if peak-points are found:
        if len(peak_ind) > 0:
            return LinePeak(x_filt, z_filt, dzdx=dzdx, d2zdx2=d2zdx2, peak_index=peak_ind, f_height=self._f_peak_height,
                            # Maintain existing info but keys filled in by this method:
                            **{key: profile[key] for key in profile.keys()
                               if key not in ['x', 'z', 'dzdx', 'd2zdx2', 'peak_index', 'f_height']}
                            )
        else:
            return None


class LineListPeakAnalyzer(PeakAnalyzerInterface):
    input_data = mutils.ListofDict()
    output_data = mutils.ListofDict()
    thresholds = AnalyzerThresholds()
    _peak_analyzer = traitlets.TraitType()
    _apply_filter = traitlets.CBool()

    def __init__(self, peak_analyzer: PeakAnalyzerInterface = None, apply_filter=True):
        """ Bulk analyzer for a list of peaks

        Analysis is performed when input_data is set (requirest ListofDict object filled with LineProfile objects)
        Analysis results are written to output_data

        Both input_data and output_data are traitlets, and can be observed/linked like traitlets

        @param LinePeakAnalyzer: LinePeakAnalyzer to use for bulk analysis
        @param apply_filter: Pre-filter line data with low-pass filtering
        """
        if peak_analyzer is None:
            peak_analyzer = LinePeakAnalyzer()

        super().__init__(input_data=mutils.ListofDict(),
                         output_data=mutils.ListofDict(),
                         _peak_analyzer=peak_analyzer,
                         _apply_filter=apply_filter,
                         thresholds=peak_analyzer.thresholds,
                         var_names_mapping=[('peak_analyzer', '_peak_analyzer'), ('apply_filter', '_apply_filter')])

        traitlets.link((self, 'thresholds'), (self._peak_analyzer, 'thresholds'))

    @traitlets.observe('thresholds')
    def _threshold_change(self, change):
        if isinstance(change['old'], AnalyzerThresholds):
            change['old'].unobserve(self._input_data_change, names=['z', 'dzdx', 'd2zdx2'])
        if isinstance(change['new'], AnalyzerThresholds):
            change['new'].observe(self._input_data_change, names=['z', 'dzdx', 'd2zdx2'])

    @traitlets.observe('input_data')
    def _input_data_change(self, change):
        if len(self.input_data) > 0:
            tmp_output = mutils.ListofDict()
            for item in self.input_data:
                res = self._peak_analyzer.analyze(item, apply_filter=self._apply_filter)

                # Add result if some peak was returned
                if isinstance(res, LinePeak):
                    tmp_output.append(res)
            self.output_data = tmp_output


class AnalysisResultFigure(traitlets.HasTraits):
    thresholds = AnalyzerThresholds(z=0, dzdx=0.0, d2zdx2=0.0)
    line_peak = LinePeak(x=None, z=None)

    def __init__(self, line_peak: LinePeak = None, sel_thresholds: AnalyzerThresholds = None, fig: Figure = None):
        """ Figure to display linepeak analysis results. It creates three axis to display: profile position,
        velocity and acceleration

        @param line_peak: Line peak object
        @param sel_thresholds:  selection criteria for analysis
        @param fig: [optional] figure on which to setup the axis
        """
        self._patches = {}
        self._lines = {}
        self._peaks = {}
        self._axs = {}
        self._fig = None
        super().__init__(thresholds=sel_thresholds, line_peak=line_peak)

        if fig is None:
            self._fig = Figure(tight_layout=True)
        else:
            self._fig = fig
        self._fig.clf()

        # Create axis:
        for i, key in enumerate(['z', 'dzdx', 'd2zdx2']):
            if i > 0:
                self._axs[key] = self._fig.add_subplot(1, 3, i + 1, sharex=self._axs['z'])
            else:
                self._axs[key] = self._fig.add_subplot(1, 3, i + 1)

            self._axs[key].set_xlabel(r'Section $\mu$m')
            self._axs[key].grid()

            if i > 0:
                self._axs[key].set_ylabel(r'$\frac{{d^{0}z}}{{dx^{0}}}$'.format(i))
            else:
                self._axs[key].set_ylabel(r'$z$ $\mu$m'.format(i))

        self._axs['z'].axis('equal')

        # Create patches and lines:
        for key, ax in self._axs.items():
            self._patches[key] = mplt.SquarePatch(ax, color='red', alpha=0.6, artist='artists')

            self._lines[key] = lines.Line2D([], [], lw=1, color='blue')
            ax.add_line(self._lines[key])

            self._peaks[key] = lines.Line2D([], [], lw=2, color='orange')
            ax.add_line(self._peaks[key])

            if key=='z':
                ax.set_xlim([-0.2, 0.2])

        if line_peak is not None:
            self.refresh()

    @property
    def fig(self):
        return self._fig

    @property
    def axs(self):
        return self._axs

    @traitlets.observe('thresholds', 'line_peak')
    def _threshold_change(self, change):
        # If object it self changes, set observer on object traits:
        if change['name'] == 'thresholds':
            if isinstance(change['old'], AnalyzerThresholds):
                change['old'].unobserve(self._threshold_change)
            if isinstance(change['new'], AnalyzerThresholds):
                change['new'].observe(self._threshold_change, names=['z', 'dzdx', 'd2zdx2'])

        self.refresh()

    def refresh(self):
        if self.line_peak is not None: #isinstance(self.line_peak, LineProfile): mz: weakened requirements to also allow other forms (e.g. dict)
            self._refresh_line_peak()
            self._refresh_threshold_patches()
            self._refresh_figure()

    def _refresh_threshold_patches(self):
        """ Refreshes the threshold data (if not None)

        :return:
        """
        if self.thresholds is not None:
            for key, ax in self._axs.items():
                # Update patches:
                x,y = self._lines[key].get_data()

                if (len(x) > 0) and (len(y) > 0):
                    if key == 'z':
                        self._patches[key].update([np.min(x), self.line_peak[key].max() ],
                                                  [np.max(x), self.line_peak[key].max() - self.thresholds[key]]
                                                  )
                    else:
                        if self.thresholds is not None:
                            self._patches[key].update([np.min(x), self.thresholds[key]],
                                                      [np.max(x), -self.thresholds[key]])

    def _refresh_line_peak(self):
        """ Refresh the line-peak data displayed on the figure

        :return:
        """
        # Update lines:
        for key in self._lines.keys():
            # Update line:
            if key in self.line_peak.keys() and 'x' in self.line_peak.keys():
                self._lines[key].set_data(self.line_peak['x'],
                                          self.line_peak[key])

            # Update peak
            if 'peak_index' in self.line_peak.keys() and 'x' in self.line_peak.keys(): #(self.line_peak, LinePeak):
                ind = self.line_peak.peak_index
                self._peaks[key].set_data(self.line_peak['x'][ind],
                                          self.line_peak[key][ind])

    def _refresh_figure(self):
        """ Refreshes axes limits and the figure canvas

        :return:
        """
        for key in self._lines.keys():
            ax = self._axs[key]

            # Setup display space
            ax.relim()  # Relim to center figure onto current data

            # Get data info displayed in current axis:
            _, z =self._lines['z'].get_data()
            x, y =self._lines[key].get_data()

            # Set xlim equal to the a pre-defined setting
            #ax.set_xlim(x[np.argmax(z)] + np.array([-0.2, 0.2]))

            # If position data, try to match 'axis equal' option but zoomed in
            # TODO: This can likely be implemented in a better way, but sort of works for the moment
            #if key == 'z':
            #    ax.set_ylim(y.max() + np.array([-0.20, 0.05]))
            ax.autoscale_view()

        if self._fig is not None:
            self._fig.canvas.draw()


if __name__ == "__main__":
    pass







