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
            'position_x': 'Pos_x',
            'position_y': 'Pos_y',
            'position_z': 'Pos_z',
            'velocity_x': 'Vel_x',
            'velocity_y': 'Vel_y',
            'velocity_z': 'Vel_z',
            'snapshot': 'SnapNum',
            'merge_type': 'mergeType',
            'dt': 'dT',
            'sfr_disk': 'SfrDisk',
            'sfr_bulge': 'SfrBulge',
            'sfr_disk_z': 'SfrDiskZ',
            'sfr_bulge_z': 'SfrBulgeZ',
            'cold_gas': 'ColdGas',
            'metals_cold_gas': 'MetalsColdGas',
            'disk_scale_radius': 'DiskScaleRadius',
        }

    def get_extra_fields(self):
        """Returns a list of SAGE fields and types to include."""

        return [
            ('Type', np.int32),
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
        par_map = {}
        for ii in range(len(tree)):
            gal_idx = tree['GalaxyIndex'][ii]
            par = par_map.get(gal_idx, None)
            if par is not None:
                descs[par] = ii
            if tree['mergeIntoID'][ii] < 0:
                par_map[gal_idx] = ii
            elif par is not None:
                del par_map[gal_idx]
        return descs

    def map_dt(self, tree):
        """Convert SAGE dT values to Gyrs"""

        return tree['dT']/1000.0

    def iterate_trees(self):
        """Iterate over SAGE trees."""

        src_type = np.dtype([
            ('SnapNum', np.int32),
            ('Type', np.int32),
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

            for ii in range(n_trees):
                tree_size = tree_sizes[ii]
                tree = np.empty(tree_size, dtype=src_type)
                offs = 0
                for jj in range(len(chunk_sizes)):
                    chunk_size = chunk_sizes[jj][ii]
                    data = np.fromfile(files[jj],src_type, chunk_size)
                    tree[offs:offs + chunk_size] = data
                    offs += chunk_size
                yield tree

            for file in files:
                file.close()
