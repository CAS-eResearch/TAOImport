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
        t0 = time.time()
        ind = (np.where(tree['descendant'] == -1))[0]
        if len(ind) > 0:
            roots = list(ind)

        ind = (np.where(tree['descendant'] != -1))[0]
        if len(ind) > 0:
            for ii, desc in zip(ind,tree['descendant'][ind]):
                parents.setdefault(desc,[]).append(ii)
                
        roots_time = time.time() - t0

        ## used to time the previous recursive chunk.
        ## now here so I don't break the code or file
        ## output columns
        recurse_time = 0.0
        
        ## find new ordering. avoid (slow) recursion
        t0 = time.time()
        start_dfi = [0]
        tree['subsize'] = 0
        for ii in roots:
            stack = [ii]
            visited = []
            while len(stack) > 0:
                idx = stack.pop()
                visited.append(idx)
                tree['subsize'][visited] += 1
                order[idx] = start_dfi[0]
                start_dfi[0] += 1
                if idx in parents:
                    for par in parents[idx]:
                        stack.append(par)
            
        new_recurse_time = time.time() - t0
        
        # Remap everything.
        t0 = time.time()
        tree['localindex'] = order[tree['localindex']]
        ind = (np.where(tree['descendant'] != -1))[0]
        if len(ind) > 0:
            tree['descendant'][ind] = order[tree['descendant'][ind]]
            tree['globaldescendant'][ind] =  tree['globalindex'][tree['descendant'][ind]]
        
        remapping_time = time.time() - t0

        # Sort the array.
        t0 = time.time()
        tree.sort(order=['localindex'])
        sort_time = time.time() - t0
        
        # Run some final checks on the descendants.
        t0 = time.time()
        assert np.max(tree['descendant']) < len(tree),"Invalid descendant index."
        ind = (np.where(tree['descendant'] != -1))[0]
        if len(ind) > 0:
            diff = set(tree['descendant'][ind] - ind) 
            assert  0 not in diff,"Descendant references same object."
                
        validation_time = time.time() - t0
        total_time = time.time() - tstart
        # print(" {:12d} {:10.6f}({:4.1f}%) {:10.6f}({:4.1f}%)  {:10.6f}({:4.1f}%) {:10.6f}({:4.1f}%) {:10.6f}({:4.1f}%) {:10.6f}({:4.1f}%)  {:10.6f}".format(len(tree),
        #                                                                                                                                 remapping_time,remapping_time/total_time*100.0,
        #                                                                                                                                 sort_time,sort_time/total_time*100.0,
        #                                                                                                                                 validation_time,validation_time/total_time*100.0,
        #                                                                                                                                 roots_time,roots_time/total_time*100.0,
        #                                                                                                                                 recurse_time,recurse_time/total_time*100.0,
        #                                                                                                                                 new_recurse_time,new_recurse_time/total_time*100.0,
        #                                                                                                                                 total_time, file=sys.stderr))
