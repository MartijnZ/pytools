import traitlets


class AbstractMutableContainer(traitlets.TraitType):

    def __init__(self, *args, **kwargs):
        """ Provides an interface to get trait notifications on list/dict mutations.
        This class can be used in combination with the following decorators:
        * mutate_on_set (e.g. to decorate __setitem__)
        * mutate_on_insert
        * mutate_on_pop
        * mutate_on_remove
        * mutate_on_clear

        These decorators create notification for container mutations

        :param args:
        :param kwargs:
        """
        super(AbstractMutableContainer, self).__init__(*args, **kwargs)
        self._parent = None

    def __set__(self, obj, value):
        if isinstance(obj, traitlets.HasTraits):
            value._parent = obj
            value.name = self.name
        traitlets.TraitType.__set__(self, obj, value)

    def _notify(self, old, new, index):
        if isinstance(self._parent, traitlets.HasTraits):
            self._parent.notify_change(traitlets.Bunch(new=new,
                                                       owner=self._parent,
                                                       name=self.name,
                                                       old=old,
                                                       index=index,
                                                       type='mutation'))


# Define decorators:
def mutate_on_set(container_attribute):
    """ Decorator that handles trait-notifications on container (list/dict) changes. It notifies about a change at
    given key

    :param container_attribute:
    :return:
    """
    def inner_set(mutate_function):
        """

        :param mutate_function: Function to decorate
        :param key: key of setitem method
        :param value: value to set
        :return: decorated mutate_function
        """
        def wrapper(self, key, value):
            # Check if self is mutable container:
            if not isinstance(self, AbstractMutableContainer):
                raise RuntimeError("mutate_on_set decorator can only operate on AbstractMutableContainer objects.")

            container = getattr(self, container_attribute)
            # Check if __setitem__ attribute is defined:
            if not hasattr(container, '__getitem__'):
                raise AttributeError("Attribute has no get item")

            # Get old values:
            if isinstance(container, dict):
                old = container[key] if key in container.keys() else None
            elif isinstance(container, list):
                old = container[key] if key < len(container) else None
            else:
                raise TypeError(f'Unknown container type {type(container)}.')


            # Apply mutation:
            res = mutate_function(self, key, value)

            # Notify mutation
            self._notify(old, value, key)

            return res
        return wrapper

    return inner_set


def mutate_on_append(container_attribute):
    """ Decorator that handles trait-notifications on container (list) append. On append, it raises a notification that
    includes the container index of the appended item, and 'None' for old (as there previously was no value on this item)

    :param container_attribute:
    :return:
    """

    def inner_set(mutate_function):
        """

        :param mutate_function: Function to decorate
        :param item: value to set
        :return: decorated mutate_function
        """

        def wrapper(self, item):
            # Check if self is mutable container:
            if not isinstance(self, AbstractMutableContainer):
                raise RuntimeError("mutate_on_set decorator can only operate on AbstractMutableContainer objects.")

            container = getattr(self, container_attribute)
            # Check if __setitem__ attribute is defined:
            if not hasattr(container, '__getitem__'):
                raise AttributeError("Attribute has no __getitem__")

            # Apply mutation:
            res = mutate_function(self, item)

            # Notify mutation
            self._notify(None, item, len(container)-1)

            return res

        return wrapper

    return inner_set


def mutate_on_insert(container_attribute):
    """ Decorator that handles trait-notifications on container insert. It notifies about all changes that occured due
    to the insert (i.e. about the inserted index and all trailing indices

    :param container_attribute:
    :return:
    """

    def inner_set(mutate_function):
        """

        :param mutate_function: Function to decorate
        :return: decorated mutate_function
        """

        def wrapper(self, index, item):
            """

            :param index: index at which item is added
            :param item: item to add
            """
            # Check if self is mutable container:
            if not isinstance(self, AbstractMutableContainer):
                raise RuntimeError("mutate_on_set decorator can only operate on AbstractMutableContainer objects.")

            container = getattr(self, container_attribute)
            # Check if __setitem__ attribute is defined:
            if not hasattr(container, '__getitem__'):
                raise AttributeError("Attribute has no __getitem__")

            # Apply mutation:
            res = mutate_function(self, index, item)

            # Notify observers of shift in container:
            N = len(container)
            for i in range(index, N):
                if i < (N - 1):
                    # Notify about shifted item:
                    self._notify(old=container[i + 1],
                                 new=container[i], index=i)
                else:
                    # Insert causes the container to grow, the last item has no 'old' value:
                    self._notify(old=None,
                                 new=container[i],
                                 index=i)

            return res

        return wrapper

    return inner_set


def mutate_on_pop(container_attribute):
    """ Decorator that handles trait-notifications on container insert. It notifies about all changes that occured due
    to the pop (i.e. about the poped index and all trailing indices

    :param container_attribute:
    :return:
    """

    def inner_set(mutate_function):
        """

        :param mutate_function: Function to decorate
        :return: decorated mutate_function
        """

        def wrapper(self, *args, **kwargs):
            """

            :param index: index at which item is added
            :param item: item to add
            """
            # Check if self is mutable container:
            if not isinstance(self, AbstractMutableContainer):
                raise RuntimeError("mutate_on_set decorator can only operate on AbstractMutableContainer objects.")

            container = getattr(self, container_attribute)
            # Check if __setitem__ attribute is defined:
            if not hasattr(container, '__getitem__'):
                raise AttributeError("Attribute has no __getitem__")

            if len(args) > 0:
                index = args[0]
            else:
                index = -1

            # Apply mutation:
            old = mutate_function(self, *args, **kwargs)

            # Notify observers of shift in container:
            N = len(container)
            if len(args) > 0:
                # Pop, caused an item to be removed from the list, and lowers the index of all trailing items
                index = args[0]
                for i in range(index, N):
                    if i==index:
                        # Notify about shifted item:
                        self._notify(old=old,
                                     new=container[i],
                                     index=i)
                    else:
                        # Insert causes the container to grow, the last item has no 'old' value:
                        self._notify(old=container[i-1],
                                     new=container[i],
                                     index=i)

            else:
                # Pop, only removed of last item, notify observers of this (index)
                self._notify(old=old,
                             new=None,
                             index=N # we pass the old index, is this ok?
                             )
            return old

        return wrapper

    return inner_set


def mutate_on_remove(container_attribute):
    """ Decorator that handles trait-notifications on container insert. It notifies about all changes that occured due
    to the pop (i.e. about the poped index and all trailing indices

    :param container_attribute:
    :return:
    """

    def inner_set(mutate_function):
        """

        :param mutate_function: Function to decorate
        :return: decorated mutate_function
        """

        def wrapper(self, key):
            """

            :param index: index at which item is added
            :param item: item to add
            """
            # Check if self is mutable container:
            if not isinstance(self, AbstractMutableContainer):
                raise RuntimeError("mutate_on_set decorator can only operate on AbstractMutableContainer objects.")

            container = getattr(self, container_attribute)
            # Check if __setitem__ attribute is defined:
            if not hasattr(container, '__getitem__'):
                raise AttributeError("Attribute has no __getitem__")

            if isinstance(container, dict):
                # Get old item:
                old = container[key]

                # Perform mutation:
                res = mutate_function(self, key)

                # Notify observer:
                self._notify(old=old,
                             new=None,
                             index=key)
                return res
            elif isinstance(container, list):
                # Get old info:
                old = key
                index = container.index(key)

                # Perform mutation:
                res = mutate_function(self, key)

                # Notify observers:
                N = len(container)
                for i in range(index, N):
                    if i==index:
                        # Notify about shifted item:
                        self._notify(old=old,
                                     new=container[i],
                                     index=i)
                    else:
                        # Insert causes the container to grow, the last item has no 'old' value:
                        self._notify(old=container[i-1],
                                     new=container[i],
                                     index=i)
                return res
        return wrapper
    return inner_set


def mutate_on_clear(container_attribute):
    """ Decorator that handles trait-notifications on container insert. It notifies about all changes that occured due
    to the pop (i.e. about the poped index and all trailing indices

    :param container_attribute:
    :return:
    """

    def inner_set(mutate_function):
        """

        :param mutate_function: Function to decorate
        :return: decorated mutate_function
        """

        def wrapper(self):
            """

            :param index: index at which item is added
            :param item: item to add
            """
            # Check if self is mutable container:
            if not isinstance(self, AbstractMutableContainer):
                raise RuntimeError("mutate_on_set decorator can only operate on AbstractMutableContainer objects.")

            container = getattr(self, container_attribute)
            # Check if __setitem__ attribute is defined:
            if not hasattr(container, '__getitem__'):
                raise AttributeError("Attribute has no __getitem__")

            # Get old item:
            old_container = container.copy()

            # Perform mutation:
            res = mutate_function(self)

            # Notify observer:
            if isinstance(old_container, dict):
                for key, item in old_container.items():
                    self._notify(old=item,
                                 new=None,
                                 index=key)
            elif isinstance(old_container, list):
                for index, item in enumerate(old_container):
                    self._notify(old=item,
                                 new=None,
                                 index=index)
            else:
                raise TypeError(f'Unknown container type {type(old_container)} for {container_attribute}.')
            return res

        return wrapper
    return inner_set


class MutableDict(AbstractMutableContainer):

    def __init__(self, default_value=traitlets.Undefined, allow_none=False, **kwargs):
        if default_value is traitlets.Undefined:
            default_value = {}
        super(MutableDict, self).__init__(default_value=default_value,
                                                  allow_none=allow_none, **kwargs)

        self._dict = default_value

    @mutate_on_set('_dict')
    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, item):
        return self._dict[item]

    def items(self):
        return self._dict.items()

    def keys(self):
        return self._dict.keys()

    def __len__(self):
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    @mutate_on_clear('_dict')
    def clear(self):
        self._dict.clear()


class MutableList(AbstractMutableContainer):

    def __init__(self, default_value=traitlets.Undefined, allow_none=False, **kwargs):
        """ A List which notifies of

        :param values:
        """
        if default_value is traitlets.Undefined:
            default_value = []

        super(MutableList, self).__init__(default_value=default_value,
                                                  allow_none=allow_none, **kwargs)
        self._list = default_value

    @mutate_on_set('_list')
    def __setitem__(self, key, value):
        self._list[key] = value

    def __getitem__(self, item):
        return self._list[item]

    def __iter__(self):
        return iter(self._list)

    @mutate_on_append('_list')
    def append(self, item):
        self._list.append(item)

    @mutate_on_remove('_list')
    def remove(self, item):
        self._list.remove(item)

    @mutate_on_pop('_list')
    def pop(self, *args, **kwargs):
        return self._list.pop(*args, **kwargs)

    @mutate_on_insert('_list')
    def insert(self, index, item):
        self._list.insert(index, item)

    @mutate_on_clear('_list')
    def clear(self):
        return self._list.clear()


if __name__ == '__main__':
    # Some elementary test on Mutable containers

    class TraitContainer(traitlets.HasTraits):
        some_list = MutableList(default_value=[])
        some_dict = MutableDict(default_value={})

        def __init__(self):
            super().__init__(some_list=MutableList(), some_dict=MutableDict())

        @traitlets.observe('some_list', 'some_dict', type='mutation')
        def _list_mutation(self, change):
            print('Mutation received: ', change)


    cont = TraitContainer()
    print('---- List operations -----')
    cont.some_list.append(1)
    cont.some_list.append(2)
    cont.some_list.append(3)
    cont.some_list.insert(0, 4)
    cont.some_list.clear()

    print('Dict operations')
    cont.some_dict['test'] = 'value'
    cont.some_dict.clear()





