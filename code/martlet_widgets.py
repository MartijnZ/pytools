import os, base64, logging, inspect
import xml.etree.cElementTree as ET
from tkinter import filedialog, Tk
import ipywidgets as widgets
from IPython.display import display

import numpy as np
import traitlets
import matplotlib.lines as mpl_lines
from matplotlib.figure import Figure

import madernpytools.backbone as mbb
import madernpytools.models.toolset_model as mts
import madernpytools.tools.frequency_response as frf
import madernpytools.plot as mplt
import madernpytools.module_data as mdata
import madernpytools.tools.line_analysis as la
import madernpytools.tools.utilities as mutils
from madernpytools.models.toolset_model import MadernModelLibrary
from ipympl.backend_nbagg import Canvas, FigureManager

_logger = logging.getLogger(f'madernpytools.{__name__}')
fig_dir= '{0}/data/figures'.format(mdata.pkg_path)


class LabeledItem(widgets.Output, traitlets.HasTraits):

    value = traitlets.TraitType()

    def __init__(self, label, widget_class, unit, tooltip=None, unit_converter: mbb.IUnitConverter = None,
                 **widget_kwargs):
        widgets.Output.__init__(self)
        self._unit_converter = unit_converter
        self._unit = unit
        self._value = None

        # Define layouts:
        self.label_layout = widgets.Layout(width='4em', Positioning='right')
        self.unit_layout = widgets.Layout(width='4em', Positioning='right')
        self.item_layout = widgets.Layout(width='4em')
        self.HBox_layout = widgets.Layout(width='100%', align='right')

        self._label_widget = widgets.Label(label, layout=self.label_layout,
                                           description_tooltip=tooltip, disabled=True)
        self._value_widget = widget_class(layout=self.item_layout,
                                          description_tooltip=tooltip, **widget_kwargs)

        if self._unit == '':
            self._unit_widget = widgets.Label(' ', layout=self.unit_layout,
                                              description_tooltip=tooltip)
        else:
            self._unit_widget = widgets.Label(r'[{0}]'.format(unit), layout=self.unit_layout,
                                              description_tooltip=tooltip)

        self._HBox = widgets.HBox([self._label_widget, self._value_widget, self._unit_widget],
                                  layout=self.HBox_layout)

        with self:
            display(self._HBox)

        widgets.link((self._value_widget, 'value'), (self, 'value'))              # Link it to value widget

    @property
    def value_widget(self):
        return self._value_widget

    def observe(self, handler, names=None, **kwargs):
        """

        @param handler:
        @param names:
        @param kwargs:
        @return:
        """
        # Overwrite default observe behavior to only link_progress this observe to value
        if names is None:
            names = ['value']
        widgets.Output.observe(self, handler=handler, names=names)

    @property
    def output(self):
        print('Output is deprecated, please LabeledItem directly inherits from output, and can be used in that way')
        return self

    @property
    def converted_value(self):
        if self._unit_converter is None:
            return self.value.value
        elif isinstance(self._unit_converter, mbb.IUnitConverter):
            return self._unit_converter.convert(self.value, self._unit)
        else:
            raise ValueError("Unknown unit converter {0}".format(self._unit_converter))


class GridItem(LabeledItem):
    """ A labeled item with information about it's location in a GridspecLayout
    """

    def __init__(self, label, widget_class, unit, column, row,
                 linked_attributes: list = None, **kwargs):
        """ Labeled item with grid location information (row and column)
        """
        self._row = row
        self._column = column

        super().__init__(label=label, widget_class=widget_class, unit=unit, **kwargs)

    @property
    def row(self):
        """ Row in which grid item should be placed
        """

        return self._row

    @property
    def column(self):
        """ Column in which grid item should be placed
        """
        return self._column

    @traitlets.validate('value')
    def validation(self, proposal):
        return proposal['value']


class PlotOutput(widgets.Output):

    def __init__(self, **kwargs):
        widgets.Output.__init__(self, **kwargs)

        # self._output = wid.Output(**kwargs)

        self._fig = None
        self._ax = None

        self._fig = Figure(**kwargs)
        self._ax = self._fig.gca()

        # Define figure canvas and manager
        self._canvas = Canvas(self._fig)
        self._manager = FigureManager(self._canvas, num=0)
        self._canvas.toolbar_visible = True

        with self:
            self._manager.show()

        self.refresh()

    @property
    def ax(self):
        return self._ax

    @property
    def fig(self):
        return self._fig

    def refresh(self):
        with self:
            self._canvas.draw_idle()


class PDFDownloader(object):

    def __init__(self, target, filename):
        """
        """
        self._target = target
        self._filename = filename

    def get_download_widget(self, link_title='Download'):
        return widgets.HTML(self._create_download_link(file_path='{0}'.format(self._target),
                                                       filename=self._filename, title=link_title)
                            )

    def _create_download_link(self, file_path, title="Download file", filename="file.pdf"):
        """Create a download link_progress"""

        # Encode file:
        with open(file_path, 'rb') as im:
            b64 = base64.encodebytes(im.read())
            payload = b64.decode()

        html = '<a download="{filename}" href="data:document/pdf;base64,{payload}" target="_blank">{title}</a>'
        html = html.format(payload=payload, title=title, filename=filename)
        return html


def plot_EMA_window_settings(window_fraction, overlap_fraction,
                             filename, fdir,
                             signal_settings=None, xlim=None, **kwargs):
    """Plot the frequency analysis for given window_fraction and overlap fraction
    params:
    window_fraction  : Size of window as fraction of number of samples
    overlap_fraction : Overlap of window as fraction of window size
    filename         : File name of the data file to analyze (use regular expression syntax to enable multiple files), e.g:
                'EMA_infeed_pull_R.*\.csv'
    dir              : Directory in which the data files are located
    signal_settings  : {'i_input':2, 'i_output': 0}
    """
    if signal_settings is None:
        signal_settings = {'i_input': 2, 'i_output': 0}

    if xlim is None:
        xlim = [0, 1500]

    fig = Figure(figsize=(10, 5))

    # Plot results
    ma = frf.ModalAnalysis()
    ma.load_from_files(filename='{0}/{1}'.format(fdir, filename) , **signal_settings,
                       window_fraction=window_fraction, overlap_fraction=overlap_fraction,
                       f_range=xlim,
                       **kwargs
                       )

    mplt.SpectralPlot(figure=fig, plot_samples=True, xlim=xlim, spectral_data=ma.spectral_data)


def interact_EMA_window_settings(filename, dir='./data', **kwargs):
    """Show interactive widget which allows one to evaluate the Hann window settings: window fraction and overlap fraction.

    Window fraction indictes the fraction of the total number of samples which forms one window.
    The overlap fraction indicates the amount of overlap between windows

    params:
    filename: File name of the data file to analyze (use regular expression syntax to enable multiple files), e.g:
                'EMA_infeed_pull_R.*\.csv'

    optional params:
    dir : Directory in which the data files are located
    signal_settings: {'i_input':2, 'i_output': 0}
    """
    sl_window = widgets.FloatSlider(max=1.0, min=0, value=0.5, continous_update=False)
    sl_overlap = widgets.FloatSlider(max=1.0, min=0, value=0.5, continous_update=False)

    f_interact = lambda window_size, overlap_fraction: plot_EMA_window_settings(window_size, overlap_fraction,
                                                                                filename, dir, **kwargs)
    widgets.interact(f_interact, window_size=sl_window, overlap_fraction=sl_overlap)


def visualize_mode(modes, modal_analysis,
                   freq_index, save=False, **kwargs):
    """ Visualizes the eigen-frequencies and the mode of a selected frequency.
    :param modes         : ModeList() object
    :type modes          : frf.ModeList
    :param modal_analysis: ModalAnalysis object which corresponds to modelist
    :param freq_index    : Index of mode to display (starts at lowest frequency)
    :param save          : Flag which indicates to save current plot
    """
    f_sel = modes[0].get_frequencies()[freq_index]
    xlim = kwargs.get('xlim', [0, 2000])
    ylim = kwargs.get('ylim', [-1.1, 1.1])
    plot_keys = kwargs.get('plot_keys', 'H')

    # Create plot
    fig = Figure(figsize=(10, 10))
    ax1 = fig.add_subplot(211)
    ax1.set_ylabel('Magnitude [dB]')
    ax1.set_xlabel('Frequency  [Hz]')
    ax1.set_xlim(xlim)
    ax2 = fig.add_subplot(212)
    ax2.set_ylabel('Im(H)/max(Im(H)) [-]')
    ax2.set_xlabel('Cylinder location [mm]')
    ax3 = ax1.twinx()
    ax3.set_ylabel('Coherence [-]')

    # Plot spectral information:
    axs = {'magnitude': ax1, 'coherence': ax3}
    myplt = modal_analysis.plot_FRF(axs={'coherence': ax3, 'magnitude': ax1}, xlim=xlim,
                                    plot_keys=plot_keys)
    myplt.show_eigen_frequencies(freqs=np.array(modes[0].get_frequencies()))
    axs['coherence'].plot([f_sel, f_sel], [0, 1], '--', linewidth=3, color='purple')
    myplt.grid(True)

    for m in modes:
        freq = m.natural_frequency
        ind = np.where(freq <= f_sel)[0][-1]
        mplt.plot_mode(ax2, m[freq[ind]])
    ax2.set_ylim(ylim)
    ax2.set_title('Frequency: {0} [Hz]'.format(np.round(f_sel)))

    # plt.tight_layout()
    if save:
        fig.savefig('Mode{0}Hz.pdf'.format(f_sel), bbox_inches='tight')


def mode_interactive(modes, modal_analysis, **kwargs):
    """ Plot
    :param modes: ModalList object
    :param modal_analysis: A modal analysis used to generate the modes
    """
    if type(modes) is not list:
        modes = [modes]
    eig_freqs = kwargs.get('freqs', modes[0].get_frequencies())

    # Widgets:
    int_slider = widgets.IntSlider(value=5, min=0, max=len(eig_freqs), step=1, description='Frequency [Hz]',
                                   continuous_update=False, readout=True, readout_format='d')
    toggle_btn = widgets.ToggleButton(value=False, description='Save')

    # Define function:
    f_interact = lambda freq, save: visualize_mode(modes, modal_analysis,
                                                         freq_index=freq, save=save, **kwargs)
    widgets.interact(f_interact, freq=int_slider, save=toggle_btn)


def try_parse(s, cast_type=float):
    try:
        if cast_type is float:
            return cast_type(eval(s))
        else:
            return cast_type(s)
    except ValueError:
        raise


def get_file_name(file_types=None):

    file_types = mbb.ArgumentVerifier(list, [('XML files', '.xml')]).verify(file_types)

    master = Tk()
    master.withdraw()
    master.wm_attributes('-topmost', 1)
    fn = filedialog.asksaveasfilename(parent=master, initialdir=os.path.curdir, filetypes=file_types)
    master.destroy()

    return fn


class MadernDropDownItem(traitlets.HasTraits):

    item = mbb.MadernObject()
    name = traitlets.CUnicode()

    def __init__(self, item: mbb.MadernObject, name: str):
        super().__init__(item=item, name=name)

    def __str__(self):
        return self.name


class MadernPropertyDisplay(widgets.Output):

    def __init__(self, class_type: traitlets.traitlets.MetaHasTraits, library: MadernModelLibrary=None,
                 default_value: mbb.MadernObject=None, *args, **kwargs):
        """ Object allows to display Madern Object constructor argument display
        :param class_type:
        :param arg_dict:
        :param file_library:
        """
        widgets.Output.__init__(self, **kwargs)

        self._class_type = class_type
        self._prop_widgets = {}
        self._change_handler = None
        self._silent_mode = False

        # Image widget
        if class_type.illustration_path != '':
            _logger.info(f'Creating widget with image based on illustration_path: {class_type.illustration_path}')
            if os.path.exists(class_type.illustration_path):
                with open(class_type.illustration_path, 'rb') as f:
                    self._im = widgets.Image(value=f.read(), format='png', width='30%', height='auto')
            else:
                _logger.warning(f'Illustration path does not exist: {class_type.illustration_path}.')
        else:
            self._im = None

        self._library = library if (library is not None) else MadernModelLibrary()

        # Default options (already available in library)
        options = [MadernDropDownItem(item, key) for key, item in self._library.get_of_type(class_type).items()]
        self._existing_configs = widgets.Dropdown(description='Known configs',
                                                  options=options)
        self._existing_configs.observe(self.cb_sel_changed, names='value')

        # Load values:
        if default_value:
           self.load(default_value)
        elif self._existing_configs.value:
            self.load(self._existing_configs.value.item)
        else:
            self.load()

    @property
    def library(self):
        return self._library

    @library.setter
    def library(self, value: MadernModelLibrary):
        self._library = value
        self.load(default_value=self.get_instance())

    def cb_sel_changed(self, obj):
        self._silent_mode = True
        self.update_default(obj['new'].item)
        self._silent_mode = False
        self.cb_val_changed(obj)

    def cb_val_changed(self, _):
        if (self._change_handler is not None) and (not self._silent_mode):
            self._change_handler(self)

    def update_default(self, default: mbb.TraitsXMLSerializer):
        for arg_key, arg_type in self.class_type.class_traits().items():
            # Get property widget
            prop_widgets = self._prop_widgets[arg_key]
            new_item = default.__dict__['_trait_values'][dict(default.var_names_mapping)[arg_key]]

            # find 'option with name 'Current'
            if isinstance(prop_widgets, widgets.Dropdown):
                option_list = list(prop_widgets.options)
                sel_index = -1
                for i, w in enumerate(option_list):
                    if w.name=='Current':
                        option_list[i] = MadernDropDownItem(new_item, 'Current')
                        sel_index=i
                        break
                if sel_index == -1:
                    sel_index = 0
                    option_list.append(MadernDropDownItem(new_item, 'Current'))

                prop_widgets.options = tuple(option_list)
                prop_widgets.index =sel_index
            else:
                # Assume text value, replace text value
                prop_widgets.value = str(new_item)

    def load(self, default_value: MadernDropDownItem=None):
        """ Load display item

        @param default_value: default values to display
        @return:
        """

        # Loop through all arguments to set their value
        for arg_key, arg_type in self._class_type.class_traits().items():
            _logger.info(f' - Creating widget for {arg_key} of type {arg_type}')
            # We consider two types of objects: MadernObjects and others (i.e. str, float, int)
            # Madern objects are displayed as dropdown list (i.e. all available options)
            # Other objects are listed as Text where user can set their value

            # MadernObject attributes are traitlets.CFloat(). When arg_type is taken from class
            # definition, it is not a class, but MetaClass. We inspect object, and cast it to class:
            prop_widgets = None
            if not inspect.isclass(arg_type):
                arg_type = type(arg_type)

            if issubclass(arg_type, mbb.MadernObject):
                # Existing property, list options:
                default_item = None
                if default_value is not None:
                    # Collect item.
                    if arg_key in dict(default_value.var_names_mapping).keys():
                        default_item = getattr(default_value, dict(default_value.var_names_mapping)[arg_key], None)

                    if default_item:
                        # Get library items of this type:
                        lib_dict = self.library.get_of_type(arg_type)
                        lib_list = [tmp for _, tmp in lib_dict.items()]
                        if default_item in lib_list:
                            # If item is in library, set it by name
                            item_index = lib_list.index(default_item)
                            item_list = [MadernDropDownItem(lib_list[item_index], list(lib_dict.keys())[item_index])]
                        else:
                            # if item is not in library set as current
                            item_list = [MadernDropDownItem(default_item, 'current')]

                else:
                    item_list = []

                # Add other options:
                item_list += [MadernDropDownItem(name=key, item=item)
                              for key, item in self._library.get_of_type(arg_type).items() if item != default_item]

                # Create Dropdown box
                prop_widgets = widgets.Dropdown(description=arg_key, options=item_list)
            else:
                if default_value is not None:
                    if arg_key in dict(default_value.var_names_mapping).keys():
                        value = str(default_value.__dict__['_trait_values'][dict(default_value.var_names_mapping)[arg_key]])
                        prop_widgets = widgets.Text(description=arg_key,
                                                    value=value,
                                                    continuous_update=False)
                    else:
                        _logger.info(f'Argument {arg_key} missing in default value of type {arg_type}')

                else:
                    prop_widgets = widgets.Text(description=arg_key,
                                            continuous_update=False)


            if prop_widgets is not None:
                # Add change handler:
                prop_widgets.observe(self.cb_val_changed)
                self._prop_widgets[arg_key] = prop_widgets

            # Reload widgets
            self._reload_widgets()

    @property
    def class_type(self):
        return self._class_type

    def _reload_widgets(self):
        self.clear_output()
        with self:
            # Optionally show image:
            if self._im:
                display(widgets.HBox([widgets.VBox(self.widgets),self._im]))
            else:
                display(widgets.VBox([w for w in self.widgets]))

    @property
    def output(self):
        return self

    def on_value_change(self, handler=None):
        self._change_handler = handler

    @property
    def widgets(self):
        return [self._existing_configs] + [item for key, item in self._prop_widgets.items()]

    def get_instance(self):
        """Generates object instance based on current value

        :return:
        """

        args_dict = {}
        for key, prop_wid in self._prop_widgets.items():
            if isinstance(prop_wid.value, MadernDropDownItem):
                args_dict[key] = prop_wid.value.item
            else:
                try:
                    args_dict[key] = try_parse(prop_wid.value, float)
                except:
                    args_dict[key] = prop_wid.value

        return self._class_type(**args_dict)


class ModelEditor(widgets.Output):

    def __init__(self, class_type: type, library: MadernModelLibrary, *args, **kwargs):
        """

        :param class_type: Class type of madern object
        :param arg_dict: Dictionary of property names and types
        :param library: Library of standard madern objects
        """
        super().__init__(**kwargs)

        self._properties = MadernPropertyDisplay(class_type, library)
        self._save_btn = widgets.Button(description='Save')
        self._save_btn.on_click(self.cb_save_btn)

        with self:
            display(self._properties.output)
            display(self._save_btn)

    def update_library(self, library):
        self._properties.library = library

    def cb_save_btn(self, _):
        # Get item instance:
        item = self._properties.get_instance()

        # Collect filename:
        fn = get_file_name()
        if fn != '':
            ET.ElementTree(item.to_xml()).write(fn)

    def get_instance(self):
        return self._properties.get_instance()

    @property
    def output(self):
        return self


def refresh_library(widget_dict, lib=None):
    """

    :param _:
    :param widget_dict:
    :param lib:
    :return:
    """
    madern_lib=mbb.ArgumentVerifier(MadernModelLibrary, mts.get_module_library()).verify(lib)
    for _, item in widget_dict.items():
        item.update_library(madern_lib)


# Cylinder properties
class GridItem(LabeledItem):

    def __init__(self, label, widget_class, unit, column, row,
                 linked_attributes: list = None, **kwargs):
        """ Labeled item with grid location information (row and column)
        """
        self._row = row
        self._column = column

        super().__init__(label=label, widget_class=widget_class, unit=unit, **kwargs)

    @property
    def row(self):
        """ Row in which grid item should be placed
        """

        return self._row

    @property
    def column(self):
        """ Column in which grid item should be placed
        """
        return self._column

    @traitlets.validate('value')
    def validation(self, proposal):
        return proposal['value']


def generate_editors_in_tab(types: list, lib=None):
    """Generates Madern Model Editors for the object names in seperate tabs
    :param types : List of Madern object types
    :param lib: [Optional] Madern object library, if not specified, default library is loaded
    :returns    : editor tabs (widget.Tabs()), and model_editors (dict)

    """
    # Generate widgets
    model_editors = {}
    madern_lib = lib  #mbb.ArgumentVerifier(MadernModelLibrary, mts.get_module_library()).verify(lib)

    for mad_type in types:
        if issubclass(mad_type, mbb.MadernObject):
            _logger.info(f' Creating editor for MadernObject type {mad_type}')
            model_editors[mad_type.__name__] = ModelEditor(mad_type, library=madern_lib)

    editor_tabs = widgets.Tab()
    editor_tabs.children = [item.output for _, item in model_editors.items()]
    for i, name in enumerate([name for i, name in enumerate(list(model_editors.keys()))]):
        editor_tabs.set_title(i,name)

    return editor_tabs, model_editors


class IWidgetLinker(traitlets.HasTraits):
    _links = traitlets.List()

    def get_trait_links(self) -> dict:
        raise NotImplementedError()

    def get_widgets(self) -> dict:
        raise NotImplementedError()

    def link_objects(self):
        """ Create links between widgets and madern_object attributes
        """
        # Remove links if exist:
        _logger.info(f'Linking objects {self}')
        if len(self._links) > 0:
            for l in self._links:
                l.unlink()
            self._links = []

        # generate trait_links:
        trait_links = self.get_trait_links()
        wid_dict = self.get_widgets()

        # Create links:
        for key, links in trait_links.items():

            _logger.info(f' - linking {key}...')
            for link in links:
                # link = ([object], [attribute_name]:
                val = getattr(*link) # Get object attribute value, i.e. object.attritube


                # Set default value
                _logger.info(f' - setting default value of {link[1]}')
                if isinstance(wid_dict[key].value_widget, widgets.Dropdown):
                    # Get options in list:
                    o_list = [o.item for o in list(wid_dict[key].value_widget.options)]

                    if val in o_list:
                        _logger.info(f'     {val} in {o_list}')
                        index = o_list.index(val)
                        o_list[index] = val
                        wid_dict[key].value_widget.options[index].item = val # Ensure val matches selected object
                        wid_dict[key].value_widget.index = index              # Set index
                        _logger.info(f' {val} found in library    ')
                        self._links.append(traitlets.link(source=(wid_dict[key].value, 'item'), target=link))
                    else:
                         _logger.warning(f' {val} item not found in options')

                    #    wid_dict[key].value_widget.options.append(MadernDropDownItem(val, name='Custom'))
                    #    # Set index
                    #    wid_dict[key].value_widget.index =  # Set index
                    #    _logger.info(f' {val} added as custom: {wid_dict[key].value}')
                    #self._links.append(traitlets.link(source=(wid_dict[key].value, 'item'), target=link))
                elif isinstance(val, (float, int, str)):
                    wid_dict[key].value = val
                    self._links.append(traitlets.link(source=(wid_dict[key], 'value'), target=link))
                else:
                    raise ValueError(f'Cannot create link {link} for {key}.')


class IMadernObjectEditor(widgets.Output):
    """ Interface for Madern object editors

    """

    madern_object = mbb.MadernObject()
    widgets = traitlets.Dict()

    @property
    def object_type(self):
        return type(self.madern_object)

    @property
    def value(self):
        return self.madern_object


class AbstractMadernObjectEditor(IMadernObjectEditor, IWidgetLinker):

    """ Abstract implementation of a Madern object Editor

    This abstraction handles the widget creation, and links the widget output to the object attributes.

    When inheriting of this class the following should be defined:
    def create_grid(self):
        raise NotImplementedError()

    def get_trait_links(self) -> dict:
        raise NotImplementedError()

    def get_widgets(self) -> dict:
        raise NotImplementedError()

    """

    def __init__(self, default_object: mbb.MadernObject, image_path: str = '', *args, **kwargs):
        """

        :param default_object: The object to be edited, its attributes will be exposed as widgets
        :param image_path    : Image to be displayed together with the madern-object.
                               This can be used to visualize the object attributes

        """
        self._silent = True

        # Initialize object
        super().__init__(madern_object=default_object, *args, **kwargs)

        # Create widgets
        self.widgets = self.generate_widgets()
        self._grid = self.create_grid()
        self._place_widgets()

        self._im = None
        if os.path.exists(image_path):
            _logger.info('Creating widget with alternative image')
            with open(image_path, 'rb') as f:
                self._im = widgets.Image(value=f.read(), width='50%', height='auto')
        elif hasattr(default_object,'illustration_path'): # != '':
            _logger.info('Creating widget with image based on illustration_path')
            if os.path.exists(default_object.illustration_path):
                with open(default_object.illustration_path, 'rb') as f:
                    self._im = widgets.Image(value=f.read(), width='50%', height='auto')
            else:
                _logger.warning(f'Illustration path does not exist: {default_object.illustration_path}.')

        # Link widgets
        self.link_objects()

        # Display all widgets
        self.display_widgets()

        for _, w in self.widgets.items():
            w.observe(handler=self.cb_widget_change, names='value')

        self._silent = False

    def cb_widget_change(self, change):
        #print('call change')
        change['name'] = 'widget'
        self.notify_change(change)

    def display_widgets(self):
        with self:
            if self._im is not None:
                #hor_line = widgets.Output(layout={'border': '.1vh solid green', 'margin': '0px 0px 0px 0px'})
                hor_line = widgets.HTML('<hr style=\'height:2px;border-width:0;color:#b51b00;background-color:#b51b00\'>')
                display(widgets.VBox([self._im, hor_line, self._grid, hor_line],
                                     layout=widgets.Layout(align_items='center', width='auto')))
            else:
                display(widgets.VBox([self._grid],
                                     layout=widgets.Layout(align_items='center', width='auto')))

    def create_grid(self) -> widgets.GridBox:
        """ Creates the gridbox used to place the attribute widgets
        """
        raise NotImplementedError()

    def generate_widgets(self):
        raise NotImplementedError()

    def get_trait_links(self) -> dict:
        """ Generates the widgets which the user can use to manipulate the Madern object attribute information
        """
        raise NotImplementedError()

    def get_widgets(self) -> dict:
        """ Generates the links between the madern widgets and the object attributes.

        Through linking, a single widget can control multiple madern-object attributes. For example:
        the tool body diameter is the same as the bearer-ring diameter and the gear pitch-diameter.

        : returns: A dictionary whose keys correspond to the key in self._widgets dictionary and whose values
        consist of a tuple (object, attribute_name). This links the value of the widget to the attribute name of the object.

        """
        return self.widgets

    @traitlets.observe('madern_object')
    def cb_object_change(self, obj):
        """ Callback for madern_object changes.

        If object changes, new links are generated and activated
        """
        if not self._silent:
            self.generate_trait_links()
            self._link_objects()

    def _place_widgets(self):
        """ Place widgets on the grid according to their row and column property
        """
        # Place widgets in grid:
        for key, item in self.widgets.items():
            self._grid[item.row, item.column] = item


class MadernObjectPropertyWidget(AbstractMadernObjectEditor):

    def __init__(self, sel_item, lib, image_path='', *args, **kwargs):
        self._lib = lib
        super().__init__(default_object=sel_item, image_path=image_path,
                         *args, **kwargs)

    def create_grid(self) -> widgets.GridspecLayout:
        """ Create grid to house madern-object attributes

        @return: GridspecLayout
        """
        n_attr = len(self.madern_object.class_traits())
        return widgets.GridspecLayout(n_attr, 1, layout=widgets.Layout(width='60%'))

    def generate_widgets(self) -> dict:
        """ Generate widgets for Madern Object attributed modification

        @return: dictionary of widgets for each madern object attribute
        """
        # Find in default list
        arg_dict = self.madern_object.class_traits()

        wid_dict = {}
        i = 0
        for key, arg_type in arg_dict.items():
            if not inspect.isclass(arg_type):
                arg_type = type(arg_type)

            # print(key)
            # Get proper input for arg_type
            if issubclass(arg_type, (int, traitlets.CInt)):
                w_class = widgets.IntText
                options = None
            elif issubclass(arg_type, (float, traitlets.CFloat)):
                w_class = widgets.FloatText
                options = None
            elif issubclass(arg_type, (float, traitlets.CUnicode)):
                w_class = widgets.Text
                options = None
            elif issubclass(arg_type, mbb.MadernObject):
                w_class = widgets.Dropdown
                options = [MadernDropDownItem(item, name=key) for key, item in
                           self._lib.get_of_type(arg_type).items()]

                n_options = len(options)

                # Add item as custom:
                if not self.madern_object in options:
                    options.append(MadernDropDownItem(arg_type, name='Custom'))
                    _logger.info(f'Created dropdown of type: {arg_type} with {n_options} entries from library and custom.')
                else:
                    _logger.info(f'Created dropdown of type: {arg_type} with {n_options} entries from library.')
            else:
                w_class = None

            if w_class is not None:
                # Create widget
                wid_dict[key] = GridItem(label=key, widget_class=w_class, unit='', row=i, column=0, options=options)
                wid_dict[key].label_layout.width = '8em'
                wid_dict[key].item_layout.width = '12em'

            i += 1

        return wid_dict

    def get_trait_links(self) -> dict:
        """ Generates a dictionary of traitlets links between madern-object attribute and corresponding widget

        @return:
        """

        _logger.info('Generating trait links: ')
        wids = self.generate_widgets()

        link_dict = {}
        obj = self.madern_object
        if obj is not None:
            for key, w in wids.items():
                link_dict[key] = [(obj, key)]

        _logger.info(f' * Created {len(link_dict)} links for {self} .')

        return link_dict

    def display_widgets(self):
        """ Method which activates widget display in

        @return:
        """
        with self:
            hor_line = widgets.Output(layout={'border': '.1vh solid green', 'margin': '0px 0px 0px 0px'})
            if self._im is not None:
                self._im.width = '50%'
                display(widgets.HBox([self._grid, self._im],
                                     layout=widgets.Layout(align_items='center', width='auto')))
            else:
                display(widgets.HBox([self._grid],
                                     layout=widgets.Layout(align_items='center', width='auto')))


class CylinderWidget(AbstractMadernObjectEditor):

    def __init__(self, cylinder: mts.Cylinder, lib, *args, **kwargs):
        self._lib = lib
        super().__init__(default_object=cylinder, *args, **kwargs)

        if self._im is not None:
            self._im.width = '80%'

    def create_grid(self):
        return widgets.GridspecLayout(4, 3, layout=widgets.Layout(width='80%'))

    def generate_widgets(self):
        return {
            'd_shaftOS': GridItem(label=r'$d_\text{OS}$', widget_class=widgets.FloatText, unit='mm',
                                  row=0, column=0, value=1.0),
            'l_shaftOS': GridItem(label=r'$l_\text{OS}$', widget_class=widgets.FloatText, unit='mm',
                                  row=1, column=0, value=1.0),
            'd_body': GridItem(label=r'$d_\text{body}$', widget_class=widgets.FloatText, unit='mm',
                               row=0, column=1, value=1.0,
                               ),
            'l_body': GridItem(label=r'$l_\text{body}$', widget_class=widgets.FloatText, unit='mm',
                               row=1, column=1, value=1.0,
                               ),
            'd_shaftDS': GridItem(label=r'$d_\text{DS}$', widget_class=widgets.FloatText, unit='mm',
                                  row=0, column=2, value=1.0,
                                  ),
            'l_shaftDS': GridItem(label=r'$l_\text{DS}$', widget_class=widgets.FloatText, unit='mm',
                                  row=1, column=2, value=1.0,
                                  ),
            'w_br': GridItem(label=r'$w_\text{br}$', widget_class=widgets.FloatText, unit='mm',
                             row=2, column=0, value=1.0,
                             ),
            'l_br': GridItem(label=r'$l_\text{br}$', widget_class=widgets.FloatText, unit='mm',
                             row=2, column=1, value=1.0,
                             ),
            'alpha_br': GridItem(label=r'$\alpha_\text{br}$', widget_class=widgets.FloatText, unit=r'$^o$',
                                 row=2, column=2, value=1.0,
                                 ),
            'material': GridItem(label=r'Material', widget_class=widgets.Dropdown, unit='',
                                 options=[MadernDropDownItem(item, key)
                                          for key, item in self._lib.get_of_type(mts.Material).items()],
                                 row=3, column=0,
                                 ),
        }

    def get_trait_links(self):

        obj = self.madern_object
        if obj is not None:
            return {
                'd_shaftOS': [(obj.os_shaft, 'outer_diameter')],
                'l_shaftOS': [(obj.os_shaft, 'length')],
                'd_body': [(obj.body, 'outer_diameter'),
                           (obj.bearer_ring, 'diameter')],
                'l_body': [(obj.body, 'length')],
                'd_shaftDS': [(obj.ds_shaft, 'outer_diameter')],
                'l_shaftDS': [(obj.ds_shaft, 'length')],
                'w_br': [(obj.bearer_ring, 'width')],
                'l_br': [(obj, 'br_location')],
                'alpha_br': [(obj.bearer_ring, 'angle')],
                'material': [(obj.os_shaft, 'material'),
                             (obj.ds_shaft, 'material'),
                             (obj.body, 'material')
                             ],
            }
        else:
            return dict()


class BasicToolsetPropertiesWidget(AbstractMadernObjectEditor):

    def __init__(self, toolset: mts.IToolset, lib, *args, **kwargs):

        self._lib = lib
        super().__init__(default_object=toolset, image_path='{}/cylinder_dimensions.png'.format(fig_dir),
                         *args, **kwargs)

        if self._im is not None:
            self._im.width = '80%'

    def create_grid(self):
        return widgets.GridspecLayout(5, 2, layout=widgets.Layout(width='80%'))

    def generate_widgets(self):
        wids = {
            'd_body': GridItem(label=r'$d_\text{body}$', widget_class=widgets.FloatText, unit='mm',
                               row=0, column=0, value=1.0),
            'w_br': GridItem(label=r'$w_\text{br}$', widget_class=widgets.FloatText, unit='mm',
                             row=1, column=0, value=1.0,
                             ),
            'l_br': GridItem(label=r'$l_\text{br}$', widget_class=widgets.FloatText, unit='mm',
                             row=2, column=0, value=1.0,
                             ),
            'F_load': GridItem(label=r'$F_\text{load}$', widget_class=widgets.FloatText, unit='N',
                               row=3, column=0, value=1.0,
                               ),
            'd_shaft': GridItem(label=r'$d_\text{shaft}$', widget_class=widgets.FloatText, unit='mm',
                               row=0, column=1, value=1.0,
                               ),
            'q_cut': GridItem(label=r'$q_\text{cut}$', widget_class=widgets.FloatText, unit='N/mm',
                              row=4, column=0, value=1.0,
                              ),
        }

        for _, w in wids.items():
            w.label_layout.width = '8em'
            w.item_layout.width = '8em'

        return wids

    def get_trait_links(self):
        obj = self.madern_object
        if obj is not None:
            return {
                'd_body': [
                    (obj.upper_cylinder.body, 'outer_diameter'),
                    (obj.upper_cylinder.bearer_ring, 'diameter'),
                    (obj.lower_cylinder.body, 'outer_diameter'),
                    (obj.lower_cylinder.bearer_ring, 'diameter'),
                    (obj.upper_gear, 'd_pitch'),
                    (obj.lower_gear, 'd_pitch'),
                ],
                'd_shaft': [
                    (obj.upper_cylinder.ds_shaft, 'outer_diameter'),
                    (obj.upper_cylinder.os_shaft, 'outer_diameter'),
                    (obj.lower_cylinder.ds_shaft, 'outer_diameter'),
                    (obj.lower_cylinder.os_shaft, 'outer_diameter'),
                ],
                'w_br': [
                    (obj.upper_cylinder.bearer_ring, 'width'),
                    (obj.lower_cylinder.bearer_ring, 'width')
                ],
                'l_br': [
                    (obj.upper_cylinder, 'br_location'),
                    (obj.lower_cylinder, 'br_location')
                ],
                'F_load': [(obj, 'F_t')],
                'q_cut': [(obj, 'q_cut')]
            }
        else:
            return dict()


class GearWidget(MadernObjectPropertyWidget):

    def __init__(self, gear: mts.Gear, lib, **kwargs):
        super().__init__(gear, lib, **kwargs)


class BearingBlockWidget(MadernObjectPropertyWidget):
    # TODO:
    # This widget generates a dropdown widget for bearings.
    # The loads of these bearings should be linked to eachother (which is not the case)

    def __init__(self, bearing_block: mts.IBearingBlock, lib, **kwargs):
        _logger.info(f'Creating new bearing block widget for {bearing_block}')
        super().__init__(bearing_block, lib,
                         **kwargs)


class ToolsetWidget(IMadernObjectEditor, IWidgetLinker):

    def __init__(self, toolset: mts.IToolset, lib):
        _logger.info('Creating ToolsetWidget')
        self._editor_tabs = widgets.Tab()
        self._widgets = None
        self._lib = lib
        super().__init__(madern_object=toolset)

        # Tabs
        # ----
        # Loads
        with self:
            display(self._editor_tabs)

    def cb_widget_change(self, change):
        self.notify_change(change)

    @traitlets.observe('madern_object')
    def cb_madern_object(self, change):
        self._widgets = self.generate_widgets(self._lib)

        self._editor_tabs.children = []
        self._widgets['Basic settings'].madern_object = change['new']
        self._widgets['Lower cylinder'].madern_object = change['new'].lower_cylinder
        self._widgets['Upper cylinder'].madern_object = change['new'].upper_cylinder

        self._editor_tabs.selected_index = None
        self._editor_tabs.children = [item for _, item in self._widgets.items()]
        self._editor_tabs.titles = [name for i, name in enumerate(list(self._widgets.keys()))]
        #    self._editor_tabs.set_title(i, name)
        self._editor_tabs.selected_index = 0

        # Ensure Toolset forces are correct by enforcing a change
        self.madern_object.F_t +=1
        self.madern_object.F_t -=1

    def generate_widgets(self, lib):
        toolset = self.madern_object
        self._editor_tabs.children = []

        ts_wids = {
            # Basic settings:
            'Basic settings': BasicToolsetPropertiesWidget(toolset=toolset, lib=lib),
            # Detailed settings:
            'Lower cylinder': CylinderWidget(toolset.lower_cylinder, lib),
            'Upper cylinder': CylinderWidget(toolset.upper_cylinder, lib),
            # Upper and lower bearing block:
            'Upper bb': BearingBlockWidget(toolset.upper_bearing_block, lib),
            'Lower bb': BearingBlockWidget(toolset.lower_bearing_block, lib),
            # Upper and lower gear:
            'Upper gear': GearWidget(toolset.upper_gear, lib),
            'Lower gear': GearWidget(toolset.lower_gear, lib),
            #
            'Layout': MadernObjectPropertyWidget(lib=lib, sel_item=toolset.layout),
            'Spacer': MadernObjectPropertyWidget(lib=lib, sel_item=toolset.spacer),
            'Tensioner': MadernObjectPropertyWidget(lib=lib, sel_item=toolset.tensioner),
        }

        for key, w in ts_wids.items():
            w.observe(handler=self.cb_widget_change, names=['widget'])

        return ts_wids

    def get_trait_links(self) -> dict:
        pass

    def get_widgets(self) -> dict:
        return self._widgets


class ProfileDisplay(widgets.Output):
    profile = traitlets.TraitType()
    thresholds = traitlets.TraitType()

    def __init__(self):
        """ Display widget to visualize LineProfile objects
        """

        # Create plot object:
        self._plot = PlotOutput(figsize=(10, 3))
        self._plot.fig.clf()

        # Setup axis & Create lines::
        self.lines = {}
        self._range_patches = {}
        self.sel_lines = {}
        self.axs = {'z': self._plot.fig.add_subplot(131),
                    'dzdx': self._plot.fig.add_subplot(132),
                    'd2zdx2': self._plot.fig.add_subplot(133),
                    }

        # Fill axis and lines:
        for key, ax in self.axs.items():
            # Line to hold fulle profile:
            self.lines[key] = mpl_lines.Line2D([], [], lw=2, color='orange', marker='.', ms=5)
            ax.add_line(self.lines[key])

            # Line to hold selected profile:
            self.sel_lines[key] = mpl_lines.Line2D([], [], lw=2, color='blue', linestyle='-')
            ax.add_line(self.sel_lines[key])

            # Patch to visualize selection criteria
            self._range_patches[key] = mplt.SquarePatch(ax, alpha=0.4, color='red')

            ax.set_xlabel('x (mm)')
            ax.grid(True)

        # Setup y-axis labels:
        self.axs['z'].set_ylabel('z (mm)')
        self.axs['dzdx'].set_ylabel(r'$\frac{dz}{dx}$ (-)')
        self.axs['d2zdx2'].set_ylabel(r'$\frac{d^2z}{dx^2}$ (-)')

        # Update layout
        self._plot.fig.tight_layout()
        self._plot.refresh()

        # initialize super classes:
        super().__init__(profile=None, thresholds=None)

        # Display contents:
        with self:
            display(self._plot)

    @traitlets.observe('thresholds')
    def thresholds_change(self, change):
        """ Handles change of thresholds object
        """

        if isinstance(change['old'], traitlets.HasTraits):
            # Remove observers of old thresholds
            change['old'].unobserve(self.threshold_value_change, names=['z', 'dzdx', 'd2zdx2'])
        if isinstance(change['new'], traitlets.HasTraits):
            # Add observers to new thresholds
            change['new'].observe(self.threshold_value_change, names=['z', 'dzdx', 'd2zdx2'])

    def threshold_value_change(self, change):
        """ Handles changes of threshold values
        """
        self.update_patches()

    def update_patches(self):
        """ Updates threshold patches
        """

        for key in ['dzdx', 'd2zdx2']:
            # 'dzdx' and 'd2zdx2' have symmetric patches:
            patch = self._range_patches[key]
            ax = self.axs[key]
            p1 = (ax.get_xlim()[0], -getattr(self.thresholds, key))
            p2 = (ax.get_xlim()[1], getattr(self.thresholds, key))
            patch.update(p1, p2)

        if self.profile is not None:
            # 'dzdx' and 'd2zdx2' has an  asymmetric patche:
            patch = self._range_patches['z']
            ax = self.axs['z']

            z_max = np.max(self.profile['z'])
            p1 = (ax.get_xlim()[0], z_max)
            p2 = (ax.get_xlim()[1], z_max - self.thresholds.z)
            patch.update(p1, p2)

        self._plot.refresh()

    @traitlets.observe('profile')
    def profile_change(self, change):
        """ Method to handle profile changes
        """
        if isinstance(change['new'], la.LineProfile):
            for label in ['z', 'dzdx', 'd2zdx2']:
                if label in self.profile.keys():
                    # Set new data:
                    self.lines[label].set_data(self.profile['x'], self.profile[label])
                    if 'peak_index' in self.profile.keys():
                        self.sel_lines[label].set_data(self.profile['x'][self.profile['peak_index']],
                                                       self.profile[label][self.profile['peak_index']])
                    if label == 'z':
                        self.axs[label].set_xlim([self.profile['x'].min(), self.profile['x'].max()])
                        self.axs[label].set_ylim([self.profile['z'].max()-0.1, self.profile['z'].max()+0.01])
                    else:
                        self.axs[label].set_ylim(np.array([-3, 3]) * getattr(self.thresholds, label))
                        self.axs[label].set_xlim([self.profile['x'].min(), self.profile['x'].max()])
                    self.axs[label].autoscale_view()
                else:
                    # Erase data:
                    self.lines[label].set_data([], [])
                    self.sel_lines[label].set_data([], [])

            self.update_patches()
            self._plot.refresh()


class ProfileListDisplay(widgets.Output):
    profile_list = traitlets.TraitType(help='Profile list from which uses can select profiles')
    thresholds = traitlets.TraitType(help='Tresholds to visualize in profile')

    def __init__(self):
        """ Class allows the user to select profile information from a list of profiles
        """

        super().__init__(profile_list=None, thresholds=None)

        # Profile display to visualize selected profile & link_progress thresholds to thresholds of this object:
        self.profile_display = ProfileDisplay()
        traitlets.link((self.profile_display, 'thresholds'), (self, 'thresholds'))

        # Slider to allow user to select profile
        self.slider = widgets.IntSlider(min=1, max=1, value=1, description='Profile index', continuous_update=False)
        self.slider.observe(self.index_change, 'value')

        # Display:
        with self:
            display(self.profile_display)
            display(self.slider)

    @traitlets.observe('profile_list')
    def list_changed(self, change):
        """ Updates profile list on change
        """
        if isinstance(change['new'], mutils.ListofDict):
            if len(change['new']) > 0:
                self.slider.max = len(change['new'])  # Update max value of slider
                self.profile_display.profile = self.profile_list[self.slider.value - 1]  # Set new slider value

    def index_change(self, change):
        # Update index change
        self.profile_display.profile = self.profile_list[change['new'] - 1]


class PeakAnalyzerThresholds(widgets.Output):
    z = traitlets.Float()
    dzdx = traitlets.Float()
    d2zdx2 = traitlets.Float()

    def __init__(self):
        """ Display

        """

        self.wids = {
            'z': widgets.FloatText(description=r'$x$', value=0, width=5,
                                   layout=widgets.Layout(width='auto'),
                                   step=0.01, min=0, max=1, continuous_update=False),
            'dzdx': widgets.FloatText(description=r'$\frac{dz}{dx}$', value=0, width=5,
                                      layout=widgets.Layout(width='auto'),
                                      step=0.1, min=0, max=10, continuous_update=False),
            'd2zdx2': widgets.FloatText(description=r'$\frac{d^2z}{dx^2}$', value=0, width=5,
                                        layout=widgets.Layout(width='auto'),
                                        step=1, min=0, max=500, continuous_update=False),
        }

        super().__init__(**{key: item.value for key, item in self.wids.items()})

        with self:
            for key, item in self.wids.items():
                traitlets.link((self, key), (item, 'value'))
                display(item)


if __name__ == "__main__":
    lib = mts.MadernModelLibrary().load('./data/library/', verbose=False)
    #parts_output, parts_widgets = generate_editors_in_tab(['Material', 'Thread',
    #                                                       'TensionRod', 'TensionNTieRods', 'RiRoTensionScrew',
    #                                                       'SetScrew', 'SpacerNSetScrews', 'RiRoSpacer'],
    #                                                       lib=lib)

    #ts_output,  ts_widgets = generate_editors_in_tab(['BobstToolset', 'Cylinder',
                                                           #'BearerRing', 'Shaft',

                                                               #'BearingBlock', 'SimpleLayout', 'Gear'], lib=lib)

    option_list = []
    for key, item in lib.get_of_type(mts.IToolset).items():
        option_list.append(MadernDropDownItem(item=item, name=key))

    dd_ts = widgets.Dropdown(options=option_list)
    print('Loading toolset widget')
    w_ts = ToolsetWidget(toolset=dd_ts.value.item, lib=lib)
    print('Done')