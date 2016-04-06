"""Convert SAGE output to TAO.

A control script to be used with `taoconvert` to convert SAGE output binary data
into HDF5 input for TAO.
"""

import re, os
import numpy as np
import tao
from collections import OrderedDict

class SAGEConverter(tao.Converter):
    """Subclasses tao.Converter to perform SAGE output conversion."""

    def __init__(self, *args, **kwargs):
        src_fields_dict = OrderedDict([
                ('StellarMass', {
                        'type': np.float32,
                        'label': "Total Stellar Mass",
                        'description': "Total stellar mass of galaxy",
                        'units': "10^10 Msun/h",
                        'group': "Galaxy Masses",
                        'order': 0,
                        }),
                ('BulgeMass', {
                        'type': np.float32,
                        'label': "Bulge Stellar Mass",
                        'description': "Stellar mass in the bulge component "\
                            "of the galaxy",
                        'units': '10^10 Msun/h',
                        'group': "Galaxy Masses",                        
                        'order': 1,
                        }),
                ('BlackHoleMass', {
                        'type': np.float32,
                        'label': "Black Hole Mass",
                        'units': "10^10 Msun/h",
                        'description': "Mass of the black hole",
                        'group': "Galaxy Masses",
                        'order': 2,
                        }),
                ('ColdGas', {
                        'type': np.float32,
                        'label': "Cold Gas Mass",
                        'units': "10^10 Msun/h",
                        'description': "Cold gas mass of the galaxy",
                        'group': "Galaxy Masses",
                        'order': 3,
                        }),
                ('HotGas', {
                        'type': np.float32,
                        'label': "Hot Gas Mass",
                        'units': "10^10 Msun/h",
                        'description': "Hot gas mass of the galaxy",
                        'group': "Galaxy Masses",                        
                        'order': 4,
                        }),
                ('EjectedMass', {
                        'type': np.float32,
                        'label': "Ejected Gas Mass",
                        'units': "10^10 Msun/h",
                        'description': "Mass of gas ejected from both the "\
                            "galaxy and the halo",
                        'group': "Galaxy Masses",
                        'order': 5,
                        }),
                ('ICS', {
                        'type': np.float32,
                        'label': "Intracluster Stellar Mass",
                        'units': "10^10 Msun/h",
                        'description': "Stellar mass in the intracluster stars",
                        'group': "Galaxy Masses",                        
                        'order': 6,
                        }),
                ('MetalsStellarMass', {
                        'type': np.float32,
                        'label': "Total Metal Mass in Stars",
                        'description': "Mass of metals in the galaxy stars "\
                            "(bulge + disk)",
                        'units': "10^10 Msun/h",
                        'group': "Galaxy Masses",                        
                        'order': 7,
                        }),
                ('MetalsBulgeMass', {
                        'type': np.float32,
                        'label': "Total Metal Mass in Bulge Stars",
                        'description': "Mass of metals in the bulge stars",
                        'units': "10^10 Msun/h",
                        'group': "Galaxy Masses",                        
                        'order': 8,
                        }),
                ('MetalsColdGas', {
                        'type': np.float32,
                        'label': "Total Metal Mass in Cold Gas",
                        'description': "Mass of metals in the cold gas phase",
                        'units': "10^10 Msun/h",
                        'group': "Galaxy Masses",                        
                        'order': 9,
                        }),
                ('MetalsHotGas', {
                        'type': np.float32,
                        'label': "Total Metal Mass in Hot Gas",
                        'description': "Mass of metals in the hot gas phase",
                        'units': "10^10 Msun/h",
                        'group': "Galaxy Masses",                        
                        'order': 10,
                        }),
                ('MetalsEjectedMass', {
                        'type': np.float32,
                        'label': "Total Metal Mass in Ejected Gas",
                        'description': "Mass of metals in the gas ejected "\
                            "from the galaxy",
                        'units': "10^10 Msun/h",
                        'group': "Galaxy Masses",                        
                        'order': 11,
                        }),
                ('MetalsICS', {
                        'type': np.float32,
                        'label': "Total Metal Mass in IntraCluster Stars",
                        'description': "Mass of metals in the intracluster "\
                            "stars",
                        'units': "10^10 Msun/h",
                        'group': "Galaxy Masses",                        
                        'order': 12,
                        }),
                ('ObjectType', {
                        'type': np.int32,
                        'label': "Galaxy Classification",
                        'description': "Galaxy type: 0-central, 1-satellite",
                        'group': "Galaxy Properties",
                        'order': 13,
                        }),
                ('DiskScaleRadius', {
                        'type': np.float32,
                        'label': "Disk Scale Radius",
                        'description': "Scale radius of the stellar disk",
                        'group': "Galaxy Properties",
                        'units': "Mpc/h",
                        'order': 14,
                        }),
                ('SfrDisk', {
                        'type': np.float32,
                        'label': "Star Formation Rate in the Disk",
                        'description': "Star formation rate in the disk",
                        'group': "Internal",
                        'units': "Msun/year",
                        'order': -1,
                        }),
                ('SfrBulge', {
                        'type': np.float32,
                        'label': "Star formation Rate in the Bulge",
                        'description': "Star formation rate in the bulge",
                        'group': "Internal",
                        'units': "Msun/year",
                        'order': -1,
                        }),
                ('SfrDiskZ', {
                        'type': np.float32,
                        'label': "Avg. Metallicity of Star-forming Disk Gas",
                        'description': "Metallicty of star forming disk gas "\
                            "(averaged over timesteps between two snapshots)"\
                            "(Mass of metals)/(Mass of star forming disk gas)",
                        'group': "Internal",
                        'units': "fraction",
                        'order': -1,
                        }),
                ('SfrBulgeZ', {
                        'type': np.float32,
                        'label': "Avg. Metallicity of Star-forming Bulge Gas",
                        'description': "Metallicty of star forming bulge gas "\
                            "(averaged over timesteps between two snapshots)"\
                            "(Mass of metals)/(Mass of star forming bulge gas)",
                        'group': "Internal",
                        'units': "fraction",
                        'order': -1,
                        }),
                ('Cooling', {
                        'type': np.float32,
                        'label': "Hot Halo Gas Cooling Rate",
                        'description': "Hot halo gas cooling rate",
                        'group': "Galaxy Properties",
                        'units': "erg/s",
                        'order': 19,
                        }),
                ('Heating', {
                        'type': np.float32,
                        'label': "AGN Heating Rate",
                        'description': "AGN Heating Rate",
                        'group': "Galaxy Properties",
                        'units': "erg/s",
                        'order': 20,
                        }),
                ('QuasarModeBHaccretionMass', {
                        'type': np.float32,
                        'label': "Quasar Mode Black Hole Accretion Mass",
                        'description': "The mass accreted onto the black hole "\
                            "in the quasar mode across the current simulation "\
                            "time step.",
                        'group': "Galaxy Properties",
                        'units': "10^10 Msun/h",
                        'order': 21,
                        }),
                ('TimeofLastMajorMerger', {
                        'type': np.float32,
                        'label': "Time of Last Major Merger",
                        'description': "Time of last major merger",
                        'group': "Galaxy Properties",
                        'units': "Myr/h",
                        'order': 22,
                        }),
                ('TimeofLastMinorMerger', {
                        'type': np.float32,
                        'label': "Time of Last Minor Merger",
                        'description': "Time of last minor merger",
                        'group': "Galaxy Properties",
                        'units': "Myr/h",
                        'order': 23,
                        }),
                ('OutflowRate', {
                        'type': np.float32,
                        'label': "Supernova Cold Gas Outflow Rate",
                        'description': "Cold gas outflow rate from "\
                            "supernovae",
                        'group': "Galaxy Properties",                        
                        'units': "Msun/yr",
                        'order': 24,
                        }),
                ('Mvir', {
                        'type': np.float32,
                        'label': "Virial Mass",
                        'description': "Virial mass of the (sub)halo",
                        'group': "Halo Properties",
                        'units': "10^10 Msun/h",
                        'order': 25,
                        }),
                ('Rvir', {
                        'type': np.float32,
                        'label': "Virial Radius",
                        'description': "Physical virial radius of the (sub)halo",
                        'group': "Halo Properties",
                        'units': "Mpc/h",
                        'order': 26,
                        }),
                ('Vvir', {
                        'type': np.float32,
                        'label': "Virial Speed",
                        'description': "Virial Speed of the (sub)halo",
                        'group': "Halo Properties",
                        'units': "km/s",
                        'order': 27,
                        }),
                ('Vmax', {
                        'type': np.float32,
                        'label': "Vmax",
                        'description': "Maximum circular speed of the (sub)halo",
                        'group': "Halo Properties",
                        'units': "km/s",
                        'order': 28,
                        }),
                ('VelDisp', {
                        'type': np.float32,
                        'label': "Velocity Dispersion",
                        'description': "Velocity dispersion of the (sub)halo",
                        'group': "Halo Properties",
                        'units': "km/s",
                        'order': 29,
                        }),
                ('Spin_x', {
                        'type': np.float32,
                        'label': "X Spin",
                        'description': 'X-component of specific angular '\
                            'momentum. Defined as Spin[3] := '\
                            ' 1/N \sum_{i=1}^{N} (r_i - rcen) \cross '\
                            ' (v_i - vcen)' \
                            ', where r_i is the physical position of the '\
                            "i'th particle, rcen is the position of the "\
                            "center of mass, v_i is the peculiar velocity"\
                            "of the particle and vcen is the bulk velocity"\
                            "of the halo", 
                        'group': "Halo Properties",
                        'units': 'Mpc * km/s',
                        'order': 30,
                        }),
                ('Spin_y', {
                        'type': np.float32,
                        'label': "Y Spin",
                        'description': 'Y-component of specific angular '\
                            'momentum. Defined as Spin[3] := '\
                            ' 1/N \sum_{i=1}^{N} (r_i - rcen) \cross '\
                            ' (v_i - vcen)' \
                            ', where r_i is the physical position of the '\
                            "i'th particle, rcen is the position of the "\
                            "center of mass, v_i is the peculiar velocity"\
                            "of the particle and vcen is the bulk velocity"\
                            "of the halo", 
                        'group': "Halo Properties",
                        'units': 'Mpc * km/s',
                        'order': 31,
                        }),
                ('Spin_z', {
                        'type': np.float32,
                        'label': "Z Spin",
                        'description': 'Z-component of specific angular '\
                            'momentum. Defined as Spin[3] := '\
                            ' 1/N \sum_{i=1}^{N} (r_i - rcen) \cross '\
                            ' (v_i - vcen)' \
                            ', where r_i is the physical position of the '\
                            "i'th particle, rcen is the position of the "\
                            "center of mass, v_i is the peculiar velocity"\
                            "of the particle and vcen is the bulk velocity"\
                            "of the halo", 
                        'group': "Halo Properties",
                        'units': 'Mpc * km/s',
                        'order': 32,
                        }),
                ('Len', {
                        'type': np.int32,
                        'label': "Total Particles",
                        'description': "Total number of simulation particles "\
                            "in the dark matter halo",
                        'group': "Halo Properties",
                        'order': 33,
                        }),
                ('CentralMvir', {
                        'type': np.float32,
                        'label': "Central Galaxy Mvir",
                        'description': "Virial mass of the central galaxy halo",
                        'group': "Halo Properties",
                        'units': "10^10 Msun/h",
                        'order': 34,
                        }),
                ('infallMvir', {
                        'type': np.float32,
                        'label': "Subhalo Mvir at Infall",
                        'description': "Subhalo Mvir at infall",
                        'group': "Halo Properties",
                        'units': "10^10 Msun/h",                        
                        'order': 35,
                        }),
                ('infallVvir', {
                        'type': np.float32,
                        'label': "Subhalo Vvir at Infall",
                        'description': "Subhalo Vvir at infall",
                        'group': "Halo Properties",
                        'units': "km/s",
                        'order': 36,
                        }),
                ('infallVmax', {
                        'type': np.float32,
                        'label': "Subhalo Vmax at Infall",
                        'description': "Subhalo Vmax at infall",
                        'group': "Halo Properties",
                        'units': "km/s",                        
                        'order': 37,
                        }),
                ('Pos_x', {
                        'type': np.float32,
                        'label': "X",
                        'description': "Co-moving X position of the (sub)halo",
                        'group': "Positions & Velocities",
                        'units': "Mpc/h",
                        'order': 38,
                        }),
                ('Pos_y', {
                        'type': np.float32,
                        'label': "Y",
                        'description': "Co-moving Y position of the (sub)halo",
                        'group': "Positions & Velocities",
                        'units': "Mpc/h",
                        'order': 39,
                        }),
                ('Pos_z', {
                        'type': np.float32,
                        'label': "Z",
                        'description': "Co-moving Z position of the (sub)halo",
                        'group': "Positions & Velocities",
                        'units': "Mpc/h",
                        'order': 40,
                        }),
                ('Vel_x', {
                        'type': np.float32,
                        'label': "X Velocity",
                        'description': "X component of the galaxy/halo velocity",
                        'group': "Positions & Velocities",
                        'units': "km/s",
                        'order': 41,
                        }),
                ('Vel_y', {
                        'type': np.float32,
                        'label': "Y Velocity",
                        'description': "Y component of the galaxy/halo velocity",
                        'group': "Positions & Velocities",
                        'units': "km/s",
                        'order': 42,
                        }),
                ('Vel_z', {
                        'type': np.float32,
                        'label': "Z Velocity",
                        'description': "Z component of the galaxy/halo velocity",
                        'group': "Positions & Velocities",
                        'units': "km/s",
                        'order': 43,
                        }),
                ('SnapNum', {
                        'type': np.int32,
                        'label': "Snapshot Number",
                        'description': "Snapshot number in the simulation",
                        'group': "Simulation",
                        'order': 44,
                        }),
                ('GalaxyIndex', {
                        'type': np.int64,
                        'label': "Galaxy ID",
                        'description': "Unique galaxy ID that identifies "\
                            "a galaxy (preserved across snapshots for the "\
                            "same galaxy)",
                        'group': "Simulation",
                        'order': 45,
                        }),
                ('CentralGalaxyIndex', {
                        'type': np.int64,
                        'label': "Central Galaxy ID",
                        'description': "Unique galaxy ID that identifies "\
                            "the central galaxy ",
                        'group': "Simulation",
                        'order': 46,
                        }),
                ('SimulationFOFHaloIndex', {
                        'type': np.int32,
                        'label': "Simulation Halo Index",
                        'description': "Index for the (sub)halo as identified "\
                            "in the simulation",
                        'group': "Simulation",
                        'order': 47,
                        }),
                ('SAGEHaloIndex', {
                        'type': np.int32,
                        'label': "SAGE Halo Index",
                        'description': "Halo index within the tree file",
                        'group': "Simulation",
                        'order': 48,
                        }),
                ('SAGETreeIndex', {
                        'type': np.int32,
                        'label': "SAGE Tree Index",
                        'description': "The index for the simulation tree file "\
                            "that this halo belongs to",
                        'group': "Simulation",
                        'order': 49,
                        }),
                ('mergeIntoID', {
                        'type': np.int32,
                        'label': "Descendant Galaxy Index",
                        'description': "Index for the descendant galaxy "\
                            "after a merger",
                        'group': "Simulation",
                        'order': 50,
                        }),
                ('mergeIntoSnapNum', {
                        'type': np.int32,
                        'label': "Descendant Snapshot",
                        'description': "Snapshot for the descendant galaxy",
                        'group': "Simulation",
                        'order': 51,
                        }),
                ('mergeType', {
                        'type': np.int32,
                        'label': "Merger Type",
                        'description': "Merger type: "\
                            "0=none; 1=minor merger; 2=major merger; "\
                            "3=disk instability; 4=disrupt to ICS",
                        'group': "Simulation",
                        'order': 52,
                        }),
                ('dT', {
                        'type': np.float32,
                        'label': "Galaxy Age",
                        'group': "Simulation",
                        'order': 53,
                        }),
                ('TotSfr', {
                        'type': np.float32,
                        'label': "Total Star Formation Rate",
                        'description': "Total star formation rate, "\
                            "(includes both disk and bulge components)",
                        'group': "Galaxy Properties",
                        'units': "Msun/year",
                        'order': 15,
                        }),
                ])

        self.src_fields_dict = src_fields_dict
        super(SAGEConverter, self).__init__(*args, **kwargs)
        
    
    @classmethod
    def add_arguments(cls, parser):
        """Adds extra arguments required for SAGE conversion.

        Extra arguments required for conversion are:
          1. The location of the SAGE output trees.
          2. The simulation box size.
          3. The list of expansion factors (a-list).
          4. The SAGE parameters file.
          5. The name of the simulation (dark matter/hydro).
          6. The name of the galaxy formation model (simulation name in case of hydro)
        """

        parser.add_argument('--trees-dir', default='.',
                            help='location of SAGE trees')
        parser.add_argument('--box-size', help='simulation box-size')
        parser.add_argument('--a-list', help='a-list file')
        parser.add_argument('--parameters', help='SAGE parameter file')
        parser.add_argument('--sim-name', help='name of the dark matter or '\
                                'hydro simulation')
        parser.add_argument('--model-name', help='name of the SAM. Set to '\
                                'simulation name for a hydro sim')

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

        mapping = {'posx': 'Pos_x',
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
                   'objecttype': 'ObjectType',
                   }

        return mapping
        
    def get_extra_fields(self):
        """Returns a list of SAGE fields and types to include."""
        wanted_field_keys = [
            'GalaxyIndex',
            'CentralGalaxyIndex', 
            'SimulationFOFHaloIndex', 
            'mergeIntoID', 
            'mergeIntoSnapNum', 
            'Spin_x', 
            'Spin_y', 
            'Spin_z', 
            'Len', 
            'Mvir', 
            'CentralMvir', 
            'Rvir', 
            'Vvir', 
            'Vmax', 
            'VelDisp', 
            'StellarMass', 
            'BulgeMass', 
            'HotGas', 
            'EjectedMass', 
            'BlackHoleMass', 
            'ICS', 
            'MetalsStellarMass', 
            'MetalsBulgeMass', 
            'MetalsHotGas', 
            'MetalsEjectedMass', 
            'MetalsICS', 
            'Cooling', 
            'Heating', 
            'QuasarModeBHaccretionMass', 
            'TimeofLastMajorMerger', 
            'TimeofLastMinorMerger', 
            'OutflowRate', 
            'infallMvir', 
            'infallVvir', 
            'infallVmax',
            'TotSfr',
        ]

        fields = OrderedDict()
        for k in wanted_field_keys:
            try:
                fields[k] = self.src_fields_dict[k]
            except KeyError:
                try:
                    fields[k] = self.src_fields_dict[k.lower()]
                except:
                    raise

        return fields
            
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
        for ii, idx in enumerate(all_gal_idx[sorted_ind]):
            jj = ii + 1
            if (jj < len(tree)) and (idx == all_gal_idx[sorted_ind[jj]]):
                descs[sorted_ind[ii]] = sorted_ind[jj]
        
        return descs

    def totsfr(self, tree):
        """ Calculate the total star formation rate.

        Just sum the disk and bulge star formation rates
        """
        
        return tree['SfrDisk'] + tree['SfrBulge']
        
    
    def map_dt(self, tree):
        """Convert SAGE dT values to Gyrs"""
        
        return tree['dT']*1e-3

    def iterate_trees(self):
        """Iterate over SAGE trees."""

        file_order = ['SnapNum', 
                      'ObjectType', 
                      'GalaxyIndex',
                      'CentralGalaxyIndex', 
                      'SAGEHaloIndex', 
                      'SAGETreeIndex', 
                      'SimulationFOFHaloIndex', 
                      'mergeType', 
                      'mergeIntoID', 
                      'mergeIntoSnapNum', 
                      'dT', 
                      'Pos_x',  'Pos_y',  'Pos_z', 
                      'Vel_x',  'Vel_y', 'Vel_z', 
                      'Spin_x',  'Spin_y', 'Spin_z', 
                      'Len', 
                      'Mvir', 
                      'CentralMvir', 
                      'Rvir', 
                      'Vvir', 
                      'Vmax', 
                      'VelDisp', 
                      'ColdGas', 
                      'StellarMass', 
                      'BulgeMass', 
                      'HotGas', 
                      'EjectedMass', 
                      'BlackHoleMass', 
                      'ICS', 
                      'MetalsColdGas', 
                      'MetalsStellarMass', 
                      'MetalsBulgeMass', 
                      'MetalsHotGas', 
                      'MetalsEjectedMass', 
                      'MetalsICS', 
                      'SfrDisk', 
                      'SfrBulge', 
                      'SfrDiskZ', 
                      'SfrBulgeZ', 
                      'DiskScaleRadius', 
                      'Cooling', 
                      'Heating', 
                      'QuasarModeBHaccretionMass', 
                      'TimeofLastMajorMerger', 
                      'TimeofLastMinorMerger', 
                      'OutflowRate', 
                      'infallMvir', 
                      'infallVvir', 
                      'infallVmax'
                      ]

        ordered_dtype = []
        for k in file_order:
            field_dict = self.src_fields_dict[k]
            ordered_dtype.append((k, field_dict['type']))
            
        computed_fields = {'TotSfr': self.totsfr}
        computed_field_list = []
        for f in computed_fields:
            if not f in field_dict.keys():
                assert "Computed field = {0} must still be defined "\
                    "in the module level field_dict".format(f)

            computed_field_list.append((f,field_dict['type']))


        # print("ordered_dtype = {0}".format(ordered_dtype))
        from_file_dtype = np.dtype(ordered_dtype)
        # print("from file type = {0}".format(from_file_dtype))
        ordered_dtype.extend(computed_field_list)
        src_type = np.dtype(ordered_dtype)
        # print("src_type = {0}".format(src_type))

        
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
                    data = np.fromfile(files[jj],from_file_dtype, chunk_size)
                    tree[offs:offs + chunk_size] = data
                    offs += chunk_size
                    
                for f in computed_field_list:
                    tree[f[0]] = (f[1]) (tree)
                
                yield tree

            for file in files:
                file.close()
