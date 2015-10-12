import numpy as np

class Generator(object):

    def get_field(self, fields, name, dtype='f'):
        fld = fields.get(name, None)
        if fld is None:
            size = len(fields[fields.keys()[0]])
            fld = np.empty(size, dtype)
            fields[name] = fld
        return fld

    def generate_fields(self, fields):
        pass

    def post_conversion(self, tree):
        pass

class GlobalIndices(Generator):
    fields = [
        ('globalindex', np.uint64),
    ]

    def __init__(self, *args, **kwargs):
        super(GlobalIndices, self).__init__(*args, **kwargs)
        self.index = 0

    def generate_fields(self, fields):
        gidxs = self.get_field(fields, 'globalindex', np.uint64)
        for ii in range(len(gidxs)):
            gidxs[ii] = self.index + ii
        self.index += ii

class TreeIndices(Generator):
    fields = [
        ('treeindex', np.uint32),
    ]

    def __init__(self, *args, **kwargs):
        super(TreeIndices, self).__init__(*args, **kwargs)
        self.index = 0

    def generate_fields(self, fields):
        tidxs = self.get_field(fields, 'treeindex', np.uint32)
        tidxs[:] = self.index
        self.index += 1

class TreeLocalIndices(Generator):
    fields = [
        ('localindex', np.uint32),
    ]

    def generate_fields(self, fields):
        lidxs = self.get_field(fields, 'localindex', np.uint32)
        for ii in range(len(lidxs)):
            lidxs[ii] = ii

class GlobalDescendants(Generator):
    fields = [
        ('globaldescendant', np.int64),
    ]

    def generate_fields(self, fields):
        gdescs = self.get_field(fields, 'globaldescendant', np.int64)
        descs = fields['descendant']
        gidxs = fields['globalindex']
        for ii in range(len(gidxs)):
            if descs[ii] != -1:
                gdescs[ii] = gidxs[descs[ii]]
            else:
                gdescs[ii] = -1

class DepthFirstOrdering(Generator):
    fields = [
        ('subsize', np.uint32),
    ]

    def post_conversion(self, tree):
        dfi = [0]
        parents = {}
        order = np.empty(len(tree), np.uint32)

        def _recurse(idx):
            order[idx] = dfi[0]
            dfi[0] += 1
            tree['subsize'][idx] = 1
            if idx in parents:
                for par in parents[idx]:
                    tree['subsize'][idx] += _recurse(par)
            return tree['subsize'][idx]

        # Find the roots and parents.
        roots = []
        for ii in range(len(tree)):
            desc = tree['descendant'][ii]
            if desc == -1:
                roots.append(ii)
            else:
                parents.setdefault(desc, []).append(ii)

        # Recurse to find the new ordering.
        for ii in roots:
            _recurse(ii)

        # Remap everything.
        for ii in range(len(tree)):
            tree['localindex'][ii] = order[tree['localindex'][ii]]
            if tree['descendant'][ii] != -1:
                tree['descendant'][ii] = order[tree['descendant'][ii]]
                tree['globaldescendant'][ii] = tree['globalindex'][tree['descendant'][ii]]

        # Sort the array.
        tree.sort(order=['localindex'])

        # Run some final checks on the descendants.
        for ii in range(len(tree)):
            desc = tree['descendant'][ii]
            if desc != -1:
                assert desc != ii, 'Descendant references same object.'
                assert desc < len(tree), 'Invalid descendant index.'
