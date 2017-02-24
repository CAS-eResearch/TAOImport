import h5py
import logging
import numpy as np
from LightCone import LightCone

logger = logging.getLogger(__name__)


class Exporter(object):

    def __init__(self, filename, converter):
        self.converter = converter
        self.chunk_size = 10000

        if filename[-3:] != '.h5':
            filename += '.h5'
        
        self.open_file(filename)

    def open_file(self, filename):
        self.file = h5py.File(filename, 'w')
        self.tree_counts = self.file.create_dataset(
            'tree_counts', (0,), dtype='uint32',
            chunks=(self.chunk_size,),
            maxshape=(None,)
        )
        self.tree_displs = self.file.create_dataset(
            'tree_displs', (1,), dtype='uint64',
            chunks=(self.chunk_size,),
            maxshape=(None,)
        )
        self.tree_displs[0] = 0

        self.galaxies = self.file.create_dataset(
            'galaxies', (0,), dtype=self.converter.galaxy_type,
            chunks=(self.chunk_size,),
            maxshape=(None,)
        )

        self.redshifts = self.file.create_dataset(
            'snapshot_redshifts', (0,), dtype='f',
            chunks=(100,),
            maxshape=(None,)
        )
        self.cosmology = self.file.create_group('cosmology')
        self.box_size = self.cosmology.create_dataset('box_size',
                                                      (1,), dtype='f')
        self.hubble = self.cosmology.create_dataset('hubble', (1,), dtype='f')
        self.omega_m = self.cosmology.create_dataset('omega_m', (1,),
                                                     dtype='f')
        self.omega_l = self.cosmology.create_dataset('omega_l', (1,),
                                                     dtype='f')

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):

        self.file.close()

    def add_tree(self, tree):
        cnt = np.uint32(len(tree))
        displ = self.tree_displs[-1]
        self.tree_counts.resize((self.tree_counts.shape[0] + 1,))
        self.tree_displs.resize((self.tree_displs.shape[0] + 1,))
        self.tree_counts[-1] = cnt
        self.tree_displs[-1] = displ + cnt
        self.galaxies.resize((self.galaxies.shape[0] + cnt,))
        dst_tree = self.converter.convert_tree(tree)
        self.galaxies[displ:displ + cnt] = dst_tree

    def set_cosmology(self, hubble, omega_m, omega_l):
        self.hubble[0] = float(hubble)
        self.omega_m[0] = float(omega_m)
        self.omega_l[0] = float(omega_l)

    def set_box_size(self, box_size):
        self.box_size[0] = float(box_size)

    def set_redshifts(self, redshifts):
        self.redshifts.resize((len(redshifts),))
        self.redshifts[:] = redshifts
