import numpy as np

class Converter(object):

    def __init__(self, modules, mapping, galaxy_type):
        self.modules = modules
        self.mapping = mapping
        self.galaxy_type = galaxy_type

    def convert_tree(self, src_tree):
        dst_tree = np.empty(len(src_tree), self.galaxy_type)

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

        # Perform a direct transfer of fields from within the
        # mapping that are flagged.
        for field, dtype in self.mapping.fields:
            dst_tree[field] = src_tree[field]

        return dst_tree
