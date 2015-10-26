from __future__ import print_function
import numpy as np
import time as time
import sys

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
        ('globalindex', np.int64),
    ]

    def __init__(self, *args, **kwargs):
        super(GlobalIndices, self).__init__(*args, **kwargs)
        self.index = 0

    def generate_fields(self, fields):
        gidxs = self.get_field(fields, 'globalindex', np.int64)
        gidxs[:] = np.arange(self.index, self.index+len(gidxs),1,dtype=np.int64)
        self.index += len(gidxs)-1

class TreeIndices(Generator):
    fields = [
        ('treeindex', np.int32),
    ]

    def __init__(self, *args, **kwargs):
        super(TreeIndices, self).__init__(*args, **kwargs)
        self.index = 0

    def generate_fields(self, fields):
        tidxs = self.get_field(fields, 'treeindex', np.int32)
        tidxs[:] = self.index
        self.index += 1

class TreeLocalIndices(Generator):
    fields = [
        ('localindex', np.int32),
    ]

    def generate_fields(self, fields):
        lidxs = self.get_field(fields, 'localindex', np.int32)
        lidxs[:] = np.arange(0,len(lidxs),1,dtype=np.int32)
            
class GlobalDescendants(Generator):
    fields = [
        ('globaldescendant', np.int64),
    ]

    def generate_fields(self, fields):
        gdescs = self.get_field(fields, 'globaldescendant', np.int64)
        descs = fields['descendant']
        gidxs = fields['globalindex']

        gdescs[:] = descs
        ind = (np.where(descs != -1))[0]
        if len(ind) > 0:
            gdescs[ind] = gidxs[descs[ind]]

class DepthFirstOrdering(Generator):
    fields = [
        ('subsize', np.int32),
    ]

    def post_conversion(self, tree):
        tstart = time.time()
        dfi = [0]
        parents = {}
        order = np.empty(len(tree), np.int32)

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
