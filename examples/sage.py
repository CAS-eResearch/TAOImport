"""Convert SAGE output to TAO.

A control script to be used with `taoconvert` to convert SAGE output binary data
into HDF5 input for TAO.
"""

import re, os
import numpy as np
import tao

class SAGEConverter(tao.Converter):
    """Subclasses tao.Converter to perform SAGE output conversion."""

    @classmethod
    def add_arguments(cls, parser):
        """Adds extra arguments required for SAGE conversion.

        Extra arguments required for conversion are:
          1. The location of the SAGE output trees.
          2. The simulation box size.
          3. The list of expansion factors (a-list).
          4. The SAGE parameters file.
        """

        parser.add_argument('--trees-dir', default='.',
                            help='location of SAGE trees')
        parser.add_argument('--box-size', help='simulation box-size')
        parser.add_argument('--a-list', help='a-list file')
        parser.add_argument('--parameters', help='SAGE parameter file')

    def get_simulation_data(self):
        """Extract simulation data.

        Extracts the simulation data from the SAGE parameter file and
        returns a dictionary containing the values.
        """

        if not self.args.box_size:
            raise tao.ConversionError('Must specify a box-size')
        if not self.args.parameters:
            raise tao.ConversionError('Must specify a SAGE parameter file')
        par = open(self.args.parameters, 'r').read()
        hubble = re.search(r'Hubble_h\s+(\d*\.?\d*)', par, re.I).group(1)
        omega_m = re.search(r'Omega\s+(\d*\.?\d*)', par, re.I).group(1)
        omega_l = re.search(r'OmegaLambda\s+(\d*\.?\d*)', par, re.I).group(1)
        return {
            'box_size': self.args.box_size,
            'hubble': hubble,
            'omega_m': omega_m,
            'omega_l': omega_l,
        }

    def get_snapshot_redshifts(self):
        """Parse and convert the expansion factors.

        Uses the expansion factors to calculate snapshot redshifts. Returns
        a list of redshifts in order of snapshots.
        """

        if not self.args.a_list:
            raise tao.ConversionError('Must specify a filename for the a-list')
        redshifts = []
        with open(self.args.a_list, 'r') as file:
            for line in file:
                redshifts.append(1.0/float(line) - 1.0)
        return redshifts

    def get_mapping_table(self):
        """Returns a mapping from TAO fields to SAGE fields."""

        return {
            'posx': 'Pos_x',
            'posy': 'Pos_y',
            'posz': 'Pos_z',
            'velx': 'Vel_x',
            'vely': 'Vel_y',
            'velz': 'Vel_z',
            'snapnum': 'SnapNum',
            'mergetype': 'mergeType',
            'dt': 'dT',
            'sfrdisk': 'SfrDisk',
            'sfrbulge': 'SfrBulge',
            'sfrdiskz': 'SfrDiskZ',
            'sfrbulgez': 'SfrBulgeZ',
            'coldgas': 'ColdGas',
            'metalscoldgas': 'MetalsColdGas',
            'diskscaleradius': 'DiskScaleRadius',
        }

    def get_extra_fields(self):
        """Returns a list of SAGE fields and types to include."""

        return [
            ('GalaxyIndex', np.int64),
            ('CentralGalaxyIndex', np.int64),
            ('SAGEHaloIndex', np.int32),
            ('SAGETreeIndex', np.int32),
            ('SimulationFOFHaloIndex', np.int32),
            ('mergeIntoID', np.int32),
            ('mergeIntoSnapNum', np.int32),
            ('Spin_x', np.float32),
            ('Spin_y', np.float32),
            ('Spin_z', np.float32),
            ('Len', np.int32),
            ('Mvir', np.float32),
            ('CentralMvir', np.float32),
            ('Rvir', np.float32),
            ('Vvir', np.float32),
            ('Vmax', np.float32),
            ('VelDisp', np.float32),
            ('StellarMass', np.float32),
            ('BulgeMass', np.float32),
            ('HotGas', np.float32),
            ('EjectedMass', np.float32),
            ('BlackHoleMass', np.float32),
            ('ICS', np.float32),
            ('MetalsStellarMass', np.float32),
            ('MetalsBulgeMass', np.float32),
            ('MetalsHotGas', np.float32),
            ('MetalsEjectedMass', np.float32),
            ('MetalsICS', np.float32),
            ('Cooling', np.float32),
            ('Heating', np.float32),
            ('QuasarModeBHaccretionMass', np.float32),
            ('TimeSinceMajorMerger', np.float32),
            ('TimeSinceMinorMerger', np.float32),
            ('OutflowRate', np.float32),
            ('infallMvir', np.float32),
            ('infallVvir', np.float32),
            ('infallVmax', np.float32),
        ]

    def map_descendant(self, tree):
        """Calculate the SAGE tree structure.

        SAGE does not output the descendant information in its tree files
        in a directly usable format. To calculate it we need to capitalise
        on the snapshot ordering of the input data, the GalaxyIndex field,
        and the mergeIntoID field.
        """
        descs = np.empty(len(tree), np.int32)
        descs.fill(-1)

        ### Now my attempt at this mapping descendants 
        ### First, sort the entire tree into using GalaxyIndex as
        ### primary key and then snapshot number as secondary key.
        ### This sorted indices will naturally flow a galaxy from
        ### earlier times (lower snapshot numbers) to later times (larger
        ### snapshot number)
        sorted_ind = np.argsort(tree, order = ('GalaxyIndex','SnapNum'))

        all_gal_idx = tree['GalaxyIndex']

        ### If a galaxy continues, then consecutive galaxy indices should
        ### be identical (within the sorted galaxyindex array). That means,
        ### a diff between neighbouring array elements should be zero. 
        diff = all_gal_idx[sorted_ind] - np.roll(all_gal_idx[sorted_ind],-1)

        ind = (np.where(diff == 0))[0]
        if len(ind) > 0:
            ### what happens if the last element and first element
            ### of all_gal_idx are the same? Do we get a run-time
            ### out-of-bounds array exception since ind+1 will be
            ### more than len(tree)? Not sure... -> MS 20th Oct, 2015
            ### Yes !! It does happen. Need to check for indices within
            ### bounds --> MS 21st Oct, 2015
            ind1 = (np.where(ind < (len(tree)-1)))[0]
            if len(ind1) > 0:
                ### map the descendants -> looks more complicated than it is
                ### First, pretend that the entire tree is already sorted
                ### --> the sorted_ind array can be removed.
                ###
                ### say, the sorted galaxy indices look like:
                ### [1 1  1  2  2  3 3 3 3  3  4  4  5 5 5 5], then, shifting the array left by 1 spot (using np.roll) gives
                ### [1 1  2  2  3  3 3 3 3  4  4  5  5 5 5 1], with the differences (in the variable diff)
                ### [0 0 -1  0 -1  0 0 0 0 -1  0 -1  0 0 0 4]
                ### Thus, the locations with 0 immediately give galaxies
                ### that have descendants. And the location of the descendants
                ### are the next array index (in the sorted array). 
                
                descs[sorted_ind[ind[ind1]]] = sorted_ind[ind[ind1]+1]

            ### The last index in the sorted_ind array can not have a descendant
            ### since there are no more galaxies in the future! 
            descs[sorted_ind[-1]] = -1
        
        return descs

    def map_dt(self, tree):
        """Convert SAGE dT values to Gyrs"""

        return tree['dT']*1e-3

    def iterate_trees(self):
        """Iterate over SAGE trees."""

        src_type = np.dtype([
            ('SnapNum', np.int32),
            ('ObjectType', np.int32),
            ('GalaxyIndex', np.int64),
            ('CentralGalaxyIndex', np.int64),
            ('SAGEHaloIndex', np.int32),
            ('SAGETreeIndex', np.int32),
            ('SimulationFOFHaloIndex', np.int32),
            ('mergeType', np.int32),
            ('mergeIntoID', np.int32),
            ('mergeIntoSnapNum', np.int32),
            ('dT', np.float32),
            ('Pos_x', np.float32), ('Pos_y', np.float32), ('Pos_z', np.float32),
            ('Vel_x', np.float32), ('Vel_y', np.float32), ('Vel_z', np.float32),
            ('Spin_x', np.float32),
            ('Spin_y', np.float32),
            ('Spin_z', np.float32),
            ('Len', np.int32),
            ('Mvir', np.float32),
            ('CentralMvir', np.float32),
            ('Rvir', np.float32),
            ('Vvir', np.float32),
            ('Vmax', np.float32),
            ('VelDisp', np.float32),
            ('ColdGas', np.float32),
            ('StellarMass', np.float32),
            ('BulgeMass', np.float32),
            ('HotGas', np.float32),
            ('EjectedMass', np.float32),
            ('BlackHoleMass', np.float32),
            ('ICS', np.float32),
            ('MetalsColdGas', np.float32),
            ('MetalsStellarMass', np.float32),
            ('MetalsBulgeMass', np.float32),
            ('MetalsHotGas', np.float32),
            ('MetalsEjectedMass', np.float32),
            ('MetalsICS', np.float32),
            ('SfrDisk', np.float32),
            ('SfrBulge', np.float32),
            ('SfrDiskZ', np.float32),
            ('SfrBulgeZ', np.float32),
            ('DiskScaleRadius', np.float32),
            ('Cooling', np.float32),
            ('Heating', np.float32),
            ('QuasarModeBHaccretionMass', np.float32),
            ('TimeSinceMajorMerger', np.float32),
            ('TimeSinceMinorMerger', np.float32),
            ('OutflowRate', np.float32),
            ('infallMvir', np.float32),
            ('infallVvir', np.float32),
            ('infallVmax', np.float32),
        ])

        entries = [e for e in os.listdir(self.args.trees_dir) if os.path.isfile(os.path.join(self.args.trees_dir, e))]
        entries = [e for e in entries if e.startswith('model_z')]
        redshift_strings = list(set([re.match(r'model_z(\d+\.?\d*)_\d+', e).group(1) for e in entries]))
        group_strings = list(set([re.match(r'model_z\d+\.?\d*_(\d+)', e).group(1) for e in entries]))

        group_strings.sort(lambda x,y: -1 if int(x) < int(y) else 1)
        redshift_strings.sort(lambda x,y: 1 if float(x) < float(y) else -1)

        for group in group_strings:
            files = []
            for redshift in redshift_strings:
                fn = 'model_z%s_%s'%(redshift, group)
                files.append(open(os.path.join(self.args.trees_dir, fn), 'rb'))

            n_trees = [np.fromfile(f, np.uint32, 1)[0] for f in files][0]
            n_gals = [np.fromfile(f, np.uint32, 1)[0] for f in files]
            chunk_sizes = [np.fromfile(f, np.uint32, n_trees) for f in files]
            tree_sizes = sum(chunk_sizes)
            
            for ii in xrange(n_trees):
                tree_size = tree_sizes[ii]
                tree = np.empty(tree_size, dtype=src_type)
                offs = 0
                for jj in xrange(len(chunk_sizes)):
                    chunk_size = chunk_sizes[jj][ii]
                    data = np.fromfile(files[jj],src_type, chunk_size)
                    tree[offs:offs + chunk_size] = data
                    offs += chunk_size
                yield tree

            for file in files:
                file.close()
