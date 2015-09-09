import numpy as np
from datatype import datatype

class Converter(object):

    def __init__(self, modules):
        self.modules = modules

    def convert_tree(self, src_tree):
        dst_tree = np.empty(len(src_tree), datatype)

        # Do conversion first.
        fields = {}
        for mod in self.modules:
            mod.convert_tree(src_tree, fields)

        # Next check for any issues.
        for mod in self.modules:
            mod.validate_fields(fields)

        # Now perform generation.
        for mod in self.modules:
            mod.generate_fields(fields)

        # Now we can merge the fields into the tree.
        for name, values in fields.iteritems():
            x = dst_tree[name]
            x[:] = values

        return dst_tree
