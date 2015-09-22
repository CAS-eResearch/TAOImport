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
        ('global_index', np.uint64),
    ]

    def __init__(self, *args, **kwargs):
        super(GlobalIndices, self).__init__(*args, **kwargs)
        self.index = 0

    def generate_fields(self, fields):
        gidxs = self.get_field(fields, 'global_index', np.uint64)
        for ii in range(len(gidxs)):
            gidxs[ii] = self.index + ii
        self.index += ii

class TreeIndices(Generator):
    fields = [
        ('tree_index', np.uint32),
    ]

    def __init__(self, *args, **kwargs):
        super(TreeIndices, self).__init__(*args, **kwargs)
        self.index = 0

    def generate_fields(self, fields):
        tidxs = self.get_field(fields, 'tree_index', np.uint32)
        tidxs[:] = self.index
        self.index += 1

class TreeLocalIndices(Generator):
    fields = [
        ('local_index', np.uint32),
    ]

    def generate_fields(self, fields):
        lidxs = self.get_field(fields, 'local_index', np.uint32)
        for ii in range(len(lidxs)):
            lidxs[ii] = ii

class GlobalDescendants(Generator):
    fields = [
        ('global_descendant', np.int64),
    ]

    def generate_fields(self, fields):
        gdescs = self.get_field(fields, 'global_descendant', np.uint32)
        descs = fields['descendant']
        gidxs = fields['global_index']
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
        order = np.empty(len(tree))

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
            tree['local_index'][ii] = order[tree['local_index'][ii]]
            if tree['descendant'][ii] != -1:
                tree['descendant'][ii] = order[tree['descendant'][ii]]
                tree['global_descendant'][ii] = tree['global_index'][tree['descendant'][ii]]
