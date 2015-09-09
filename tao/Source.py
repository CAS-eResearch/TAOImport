import numpy as np

# class LazyTreeLoader(object):

#     def __init__(self, tree_sizes_fn, trees_fn):
#         self.open_npy(open(tree_sizes_fn, 'rb'), 'tree_sizes')
#         self.open_npy(open(trees_fn, 'rb'), 'trees')

#     def __iter__(self):
#         while 1:
#             self.read_next()
#             yield self.tree

#     def read_next(self):
#         self.tree_size = self.read_npy('tree_sizes', 1)
#         self.tree = self.read_npy('trees', self.tree_size)

#     def open_npy(self, fh, attr):
#         major, minor = np.lib.format.read_magic(fh)
#         shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
#         assert not fortran, "Fortran order arrays not supported"

#         # Get the number of elements in one 'row' by taking
#         # a product over all other dimensions.
#         row_size = numpy.prod(shape[1:])
#         row_stride = row_size * dtype.itemsize

#         # Store things.
#         setattr(self, attr + '_fh', fh)
#         setattr(self, attr + '_row_size', row_size)
#         setattr(self, attr + '_row_stride', row_stride)
#         setattr(self, attr + '_dtype', dtype)
#         setattr(self, attr + '_shape', shape)
#         setattr(self, attr + '_start_row', 0)

#     def read_npy(self, attr, num_rows):
#         fh = getattr(self, attr + '_fh')
#         shape = getattr(self, attr + '_shape')
#         dtype = getattr(self, attr + '_dtype')
#         row_size = getattr(self, attr + '_row_size')
#         start_row = getattr(self, attr + '_start_row')
#         assert start_row < shape[0], 'start_row is beyond end of file'
#         assert start_row + num_rows <= shape[0], 'start_row + num_rows > shape[0]'
#         n_items = row_size*num_rows
#         flat = numpy.fromfile(fh, count=n_items, dtype=dtype)
#         return flat.reshape((-1,) + shape[1:])

class LazyLoader(object):

    def __init__(self, filename):
        self.fh = open(filename, 'rb')
        major, minor = np.lib.format.read_magic(fh)
        self.shape, fortran, self.dtype = np.lib.format.read_array_header_1_0(fh)
        assert not fortran, "Fortran order arrays not supported"

        # Get the number of elements in one 'row' by taking
        # a product over all other dimensions.
        self.row_size = numpy.prod(self.shape[1:])
        self.row_stride = self.row_size*self.dtype.itemsize
        self.start_row = 0

    def read(self, num_rows):
        assert self.start_row < self.shape[0], 'start_row is beyond end of file'
        assert self.start_row + num_rows <= self.shape[0], 'start_row + num_rows > shape[0]'
        self.start_row += num_rows
        n_items = self.row_size*num_rows
        flat = numpy.fromfile(self.fh, count=n_items, dtype=self.dtype)
        return flat.reshape((-1,) + self.shape[1:])

    def done(self):
        return self.start_row >= self.shape[0]

class Source(object):

    def __init__(self, tree_sizes_fn, trees_fn, mapping=None):
        self.tree_sizes = LazyLoader(tree_sizes_fn)
        self.trees = LazyLoader(trees_fn)
        self.load_mapping(mapping)

    def load_mapping(self, mapping):
        self.mapping = mapping

    def __iter__(self):
        while not self.tree_sizes.done():
            tree_size = self.tree_sizes.read(1)
            yield self.trees.read(tree_size)
