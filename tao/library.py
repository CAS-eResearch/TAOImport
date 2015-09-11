class LazyReference(object):

    def __init__(self, library, key):
        self.library = library
        self.key = key

    def get(self):
        return self.library.items[self.key]

    def __lt__(self, other):
        return self.get() < other

    def __gt__(self, other):
        return self.get() > other

    def __le__(self, other):
        return self.get() <= other

    def __ge__(self, other):
        return self.get() >= other

    def __eq__(self, other):
        return self.get() == other

    def __ne__(self, other):
        return self.get() != other

    def __str__(self):
        return str(self.get())

    def __sub__(self, other):
        return self.get() - other

class Library(object):

    def __init__(self):
        self.items = {}

    def __getitem__(self, key):
        return LazyReference(self, key)

    def __setitem__(self, key, value):
        self.items[key] = value

library = Library()
