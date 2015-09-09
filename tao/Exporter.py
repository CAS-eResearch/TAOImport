import h5py, logging
from LightCone import LightCone
from Converter import Converter
from datatype import datatype

logger = logging.getLogger(__name__)

class Exporter(object):

    def __init__(self, filename, modules, mapping=None, converter=None, arguments=None):
        self.arguments = arguments
        self.modules = modules
        self.galaxy_type = datatype
        self.chunk_size = 10000
        self.open_file(filename + '.h5')
        self.mapping = mapping
        self.converter = converter if converter else self.default_converter()

    def default_converter(self):
        modules = [cls(self.mapping, self.arguments) for cls in self.modules]
        return Converter(modules)

    def open_file(self, filename):
        self.file = h5py.File(filename, 'w')
        self.tree_counts = self.file.create_dataset(
            'tree_counts', (0,), dtype='i',
            chunks=(self.chunk_size,),
            maxshape=(None,)
        )
        self.tree_displs = self.file.create_dataset(
            'tree_displs', (1,), dtype='i',
            chunks=(self.chunk_size,),
            maxshape=(None,)
        )
        self.tree_displs[0] = 0
        self.galaxies = self.file.create_dataset(
            'galaxies', (0,), dtype=self.galaxy_type,
            chunks=(self.chunk_size,),
            maxshape=(None,)
        )
        self.redshifts = self.file.create_dataset(
            'snapshot_redshifts', (0,), dtype='f',
            chunks=(100,),
            maxshape=(None,)
        )
        self.cosmology = self.file.create_group('cosmology')
        self.box_size = self.cosmology.create_dataset('box_size', (1,), dtype='f')
        self.hubble = self.cosmology.create_dataset('hubble', (1,), dtype='f')
        self.omega_m = self.cosmology.create_dataset('omega_m', (1,), dtype='f')
        self.omega_l = self.cosmology.create_dataset('omega_l', (1,), dtype='f')

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.file.close()

    def add_tree(self, tree):
        logger.info('Resizing counts from %d to %d'%(self.tree_counts.shape[0], self.tree_counts.shape[0] + 1))
        self.tree_counts.resize((self.tree_counts.shape[0] + 1,))
        logger.info('Resizing displs from %d to %d'%(self.tree_displs.shape[0], self.tree_displs.shape[0] + 1))
        self.tree_displs.resize((self.tree_displs.shape[0] + 1,))
        self.tree_counts[-1] = len(tree)
        self.tree_displs[-1] = self.tree_displs[-2] + len(tree)
        logger.info('Resizing galaxies from %d to %d'%(self.galaxies.shape[0], self.galaxies.shape[0] + len(tree)))
        self.galaxies.resize((self.galaxies.shape[0] + len(tree),))
        dst_tree = self.converter.convert_tree(tree)
        logger.info('Writing tree from %d to %d'%(self.tree_displs[-2], self.tree_counts[-1]))
        for field in  dst_tree.dtype.names:
            logger.info('Writing field %s'%field)
            self.galaxies[self.tree_displs[-2]:self.tree_displs[-1],field] = dst_tree

    def set_cosmology(self, hubble, omega_m, omega_l):
        self.hubble[0] = hubble
        self.omega_m[0] = omega_m
        self.omega_l[0] = omega_l

    def set_box_size(self, box_size):
        self.box_size[0] = box_size

    def set_redshifts(self, redshifts):
        self.redshifts.resize((len(redshifts),))
        self.redshifts[:] = redshifts
