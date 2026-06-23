import os,re, subprocess
import numpy as np
import madernpytools.backbone as mbb
import xml.etree.cElementTree as ET
import traitlets


class Listof(traitlets.TraitType):
    """ ListOf class

    :param items: Items to initalize list of with
    :param args: optional arguments
    :param item_type: type of items contained in this list
    :param kwargs: optional keyword arguments
    """

    def __init__(self, items: list = None, *args, item_type:type=object, **kwargs):
        """Constructor

        """
        super().__init__(*args, **kwargs)
        self._item_type = item_type
        self._list = items if isinstance(items, list) else []

    def __iter__(self):
        return self._list.__iter__()

    def __len__(self):
        return len(self._list)

    def insert(self, index, item):
        """Inserts item at specified index. The items trailing index are shifted

        @param index: Index at which item should be added
        @param item: Item to add
        @return:
        """
        if isinstance(item, self._item_type):
            self._list.insert(index, item)
        else:
            raise TypeError('Expected {0} got {1}'.format(self._item_type, type(item)))

    def remove(self, item):
        """Removes given item from list

        @param item: Item to remove
        @return:
        """
        self._list.remove(item)

    def append(self, item):
        """Append list with item

        @param item: Item to append
        @return:
        """
        if isinstance(item, self._item_type):
            self._list.append(item)
        else:
            raise TypeError('Expected {0} got {1}'.format(self._item_type, type(item)))

    def pop(self, index):
        """Pops item at given index

        @param index: index to pop
        @return:
        """
        self._list.pop(index)

    def clear(self):
        """ Clears list

        @return:
        """
        self._list.clear()

    def get_with_key_value(self, key, value):
        """Get a sublist of items for which the key-value matches a specific value

        :param key: The key to assess
        :param value: The value to match (can be a regular expression pattern)
        """
        sub_list = ListofDict()
        for item in self:
            if isinstance(value, str):
                if (key in item) and re.match(value, str(item[key])):
                    sub_list.append(item)
            else:
                if (key in item) and value==item[key]:
                    sub_list.append(item)

        return sub_list

    def __setitem__(self, i: int, o: dict):
        if isinstance(o, self._item_type):
            list.__setitem__(self._list, i, o)
        else:
            raise TypeError("Object should be of type dict")

    def __getitem__(self, item):
        if isinstance(item, int):
            return list.__getitem__(self._list, item)
        elif isinstance(item, slice):
            return type(self)(items=list.__getitem__(self._list, item))
        elif isinstance(item, (list, np.ndarray)):
            # Allow for indexing with list or ndarray of integers
            return type(self)(items=[list.__getitem__(self._list, i) for i in item])

    def __add__(self, other):
        if isinstance(other, Listof):
            return type(self)(items=self._list + other._list)
        else:
            raise TypeError('Cannot combine {} and {}'.format(type(self), type(other)))

    def __contains__(self, item):
        return self._list.__contains__(item)

    def get_key_vals(self, key):
        """Get items (deprecated, use get_key_values()

        @param key:
        @return:
        """
        print("The function 'get_key_vals(...)' is deprecated and will be removed please use get_key_values() instead.")
        return self.get_key_values(key)

    def get_key_values(self, key):
        """Returns all values dictionaries values that match the given key
        :param key: key value to match
        :returns: List of values
        """
        data = []
        for item in self:
            if key in item:
                data.append(item[key])
        return data

    def get_unique_values(self, key):
        """Get list of unique values for the given key

        :param key: Key value to use
        """
        data = np.array(self.get_key_values(key))
        return list(np.unique(data))

    def get_keys(self):
        """Get list of keys that exist in this dictionary list

        :return: a list with all unique object keys
        """

        keys = []
        for item in self:
            for k in item.keys():
                if not k in keys:
                    keys.append(k)
        return keys

    def __str__(self):
        return f'List of {type(self._item_type).__name__}, length: {len(self)}'


class ListofDict(Listof):
    """Implementation of ListOf for dictionaries

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, item_type=object, **kwargs) # Todo: this should be set to dict, but as many items still rely on object, it has not been chnaged


class PDFDocumentGenerator(mbb.ProcessStatus):
    """This class allows the generation of a PDF file from a Latex 'template'. Inspiration for this class
    is obtained through the following link_progress:
    http://akuederle.com/Automatization-with-Latex-and-Python-1

    The template_file is a standard LateX file which class instances compiles. By incorporating LateX commands into
    the template. The variables are introduced in the LateX file.
    """

    def __init__(self, template_file='./sample_doc.tex', build_folder=None, compile_command=None):
        """Constructor

        :param template_file: Relative path to Latex template
        :param build_folder: Location to build latex files, if None, a build folder will be created in the current path.
        :param compile_command: Latex compile string is used as os.system(compile_command.format(build_folder, build_folder))
        defaults to 'pdflatex -output-directory {0} {1}'
        """
        super().__init__()
        self._template_dir = os.path.dirname(template_file)
        self._template_fn = os.path.splitext(os.path.basename(template_file))[0]
        self._compile_command = mbb.ArgumentVerifier(str, "pdflatex -output-directory {0} {1}").verify(compile_command)

        if build_folder is not None:
            self._build_folder = build_folder
            if not os.path.exists(build_folder):
                os.mkdir(build_folder)
        else:
            if not os.path.exists('./build/'):
                os.mkdir('./build/')
            self._build_folder = './build/'

    def generate(self, values_dict: dict = None, verbose: bool = False):
        """ Generate LateX file using commands defined in value dict.
        For each 'key' and 'value' in value_dict the following commands in the template file are redefined

        :param values_dict: dictionary with values
        :param verbose: print  compile progress
        :return:
        """

        values_dict = mbb.ArgumentVerifier(dict, {}).verify(values_dict)

        # define build directory
        out_file = "{}/commands".format(self._build_folder)
        compile_file = '{0}/{1}.tex'.format(self._template_dir, self._template_fn)

        # Generate tex commands:
        if verbose:
            print('Generating {0}'.format(out_file))
        tex_code = ""
        for key, value in values_dict.items():
            comm = "\\renewcommand{{\\{}}}{{{}}}\n".format(key, value)
            tex_code+= comm
            if verbose:
                print('- {0}'.format(comm))

        self.status_message = f'Writing commands...'
        if verbose:
            print('- writing to file'.format(out_file))
        with open(out_file + ".tex", "w") as f:  # saves tex_code to output file
            f.write(tex_code)

        tmp = "pdflatex -output-directory {0} {1}".format(self._build_folder,
                                                          compile_file)
        if verbose:
            print("Compiling LateX: {0}".format(tmp))
        # Compile twice to ensure all references are ok
        self.progress = 0.1
        for i in range(2):
            if verbose:
                print('- {0} compilation...'.format(i+1))

            self.progress = 0.1 + 0.4*i
            self.status_message = f'Compiling {i+1}/2'
            subprocess.run(['pdflatex', '-output-directory', self._build_folder, compile_file])

        if verbose:
            print('Done')
        self.progress = 1.0


class XMLFileManager(object):

    @staticmethod
    def load(filename: str, class_factory: mbb.IClassFactory=None):
        """Deserialize xml_item using the specified class factory.
        The class factory ensures proper deserialization. It ensures the objects are iniatialized in the proper scope.

        :param filename: Filename of XML item to deserialize
        :param class_factory: Class Factor to use for serialization
        :return:
        """
        # Define class factory:
        class_factory = class_factory if class_factory is not None else mbb.IClassFactory

        # Load item
        root_elem = ET.parse(filename).getroot()
        root_type = class_factory.get(root_elem.get('Type'))
        return root_type.from_xml(root_elem, class_factory)

    @staticmethod
    def save(item: mbb.IXML, filename: str):
        """

        @param item:
        @param filename:
        @return:
        """
        if filename != '':
            xml_item = item.to_xml()
            ET.ElementTree(xml_item).write(filename)


if __name__ == "__main__":
    gen = PDFDocumentGenerator(template_file='./latex_templates/minimal_template.tex')
    gen.generate(values_dict={'Author': 'John Doe'}, verbose=True)
