import numpy as np

class Generator(object):

    def get_field(self, fields, name, dtype='f'):
        fld = fields.get(name, None)
        if fld is None:
            size = len(fields[fields.keys()[0]])
            fld = np.empty(size, dtype)
            fields[name] = fld
        return fld

class GlobalIndices(Generator):

    def __init__(self, *args, **kwargs):
        super(GlobalIndices, self).__init__(*args, **kwargs)
        self.index = 0

    def generate_fields(self, fields):
        gidxs = self.get_field(fields, 'global_index', np.uint64)
        for ii in range(len(gidxs)):
            gidxs[ii] = self.index + ii
        self.index += ii

class TreeIndices(Generator):

    def __init__(self, *args, **kwargs):
        super(TreeIndices, self).__init__(*args, **kwargs)
        self.index = 0

    def generate_fields(self, fields):
        tidxs = self.get_field(fields, 'tree_index', np.uint32)
        tidxs[:] = self.index
        self.index += 1

class TreeLocalIndices(Generator):

    def generate_fields(self, fields):
        lidxs = self.get_field(fields, 'local_index', np.uint32)
        for ii in range(len(lidxs)):
            lidxs[ii] = ii

class GlobalDescendants(Generator):

    def generate_fields(self, fields):
        gdescs = self.get_field(fields, 'global_descendants', np.uint32)
        descs = fields['descendants']
        gidxs = fields['global_index']
        for ii in range(len(lidxs)):
            if descs[ii] != -1:
                gdescs[ii] = gids[descs[ii]]
            else:
                gdescs[ii] = -1
