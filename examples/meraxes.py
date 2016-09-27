"""Convert MERAXES output to TAO.

A control script to be used with `taoconvert` to convert SAGE output
binary data into HDF5 input for TAO.
"""

import re
import os
import numpy as np
import tao
from collections import OrderedDict
import progressbar
import h5py


class MERAXESConverter(tao.Converter):
    """Subclasses tao.Converter to perform MERAXES output conversion."""

    def __init__(self, *args, **kwargs):
        src_fields_dict = OrderedDict([
                ("id_MBP", {
                        "type": np.int64,
                        "label": "MBP ID",
                        "order": 0,
                        "units": "None",
                        "group": "Halo Properties",
                        "description": "Most-bound particle ID"
                        }),
                ("ID", {
                        "type": np.int64,
                        "label": "Galaxy ID",
                        "order": 1,
                        "units": "None",
                        "group": "Galaxy Properties",
                        "description": "Unique galaxy ID"
                        }),
                ("Type", {
                    "type": np.int32,
                    "label": "Galaxy type",
                    "order": 2,
                    "units": "None",
                    "group": "Galaxy Properties",
                    "description": "Galaxy type: 0->Central, 1->Satellite, "\
                        "2->Orphan"
                    }),
                ("CentralGal", {
                        "type": np.int32,
                        "label": "Central galaxy index",
                        "order": 3,
                        "units": "None",
                        "group": "Galaxy Properties",
                        "description": "Index of the central galaxy of the "\
                            "host FOF group"
                        }),
                ("GhostFlag", {
                    "type": np.int32,
                    "label": "Ghost galaxy flag",
                    "order": 4,
                    "units": "None",
                    "group": "Galaxy Properties",
                    "description": "Ghost galaxy flag: 0->Host identified, 1->Ghost"
                    }),
                ("Len", {
                        "type": np.int32,
                        "label": "Particle number",
                        "order": 5,
                        "units": "None",
                        "group": "Halo Properties",
                        "description": "Number of particles in host subhalo"
                        }),
                ("MaxLen", {
                    "type": np.int32,
                    "label": "Maximum particle number",
                    "order": 6,
                    "units": "None",
                    "group": "Halo Properties",
                    "description": "Maximum number of particles in host subhalo"
                    }),
                ("PhysicsFlags", {
                    "type": np.int32,
                    "label": "Physics debug flags",
                    "order": -1,
                    "units": "None",
                    "group": "Internal",
                    "description": "Debugging flags for physical recipes"
                    }),
                ("Pos_0", {
                        "type": np.float32,
                        "label": "Subhalo X-postion",
                        "order": 7,
                        "units": "Mpc/h",
                        "group": "Positions and velocities",
                        "description": "Position of host subhalo at snapshot "\
                            "when last identified"
                        }),
                ("Pos_1", {
                        "type": np.float32,
                        "label": "Subhalo Y-postion",
                        "order": 8,
                        "units": "Mpc/h",
                        "group": "Positions and velocities",
                        "description": "Position of host subhalo at snapshot "\
                            "when last identified"
                        }),
                ("Pos_2", {
                        "type": np.float32,
                        "label": "Subhalo Z-postion",
                        "order": 9,
                        "units": "Mpc/h",
                        "group": "Positions and velocities",
                        "description": "Position of host subhalo at snapshot "\
                            "when last identified"
                        }),
                ("Vel_0", {
                    "type": np.float32,
                    "label": "Subhalo X-velocity",
                    "order": 10,
                    "units": "km/s",
                    "group": "Positions and velocities",
                    "description": "Comoving velocity of subhalo at last "\
                        "snapshot identified"
                    }),
                ("Vel_1", {
                    "type": np.float32,
                    "label": "Subhalo Y-velocity",
                    "order": 11,
                    "units": "km/s",
                    "group": "Positions and velocities",
                    "description": "Comoving velocity of subhalo at last "\
                        "snapshot identified"
                    }),
                ("Vel_2", {
                    "type": np.float32,
                    "label": "Subhalo Z-velocity",
                    "order": 12,
                    "units": "km/s",
                    "group": "Positions and velocities",
                    "description": "Comoving velocity of subhalo at last "\
                        "snapshot identified"
                    }),
                 ("Spin", {
                        "type": np.float32,
                        "label": "Subhalo spin",
                        "order": 13,
                        "units": "None",
                        "group": "Halo Properties",
                        "description": "Bullock style lambda parameter"
                        }),
                ("Mvir", {
                        "type": np.float32,
                        "label": "Subhalo virial mass",
                        "order": 14,
                        "units": "1e10 solMass/h",
                        "group": "Halo Properties",
                        "description": "Virial mass of host subhalo"
                        }),
                ("Rvir", {
                        "type": np.float32,
                        "label": "Subhalo virial radius",
                        "order": 15,
                        "units": "Mpc/h",
                        "group": "Halo Properties",
                        "description": "Virial radius of host subhalo"
                        }),
                ("Vvir", {
                        "type": np.float32,
                        "label": "Subhalo virial velocity",
                        "order": 16,
                        "units": "km/s",
                        "group": "Halo Properties",
                        "description": "Virial velocity of host subhalo"
                        }),
                ("Vmax", {
                        "type": np.float32,
                        "label": "Subhalo maximum circular velocity",
                        "order": 17,
                        "units": "km/s",
                        "group": "Halo Properties",
                        "description": "Maximum circular velocity of host subhalo"
                        }),
                ("FOFMvir", {
                    "type": np.float32,
                    "label": "FOF virial mass",
                    "order": 18,
                    "units": "1e10 solMass/h",
                    "group": "Halo Properties",
                    "description": "Virial mass of the host friends-of-friends group"
                    }),
                ("HotGas", {
                        "type": np.float32,
                        "label": "Hot gas mass",
                        "order": 19,
                        "units": "1e10 solMass/h",
                        "group": "Galaxy Masses",
                        "description": "Mass of hot gas"
                        }),
                ("MetalsHotGas", {
                        "type": np.float32,
                        "label": "Hot metals mass",
                        "order": 20,
                        "units": "1e10 solMass/h",
                        "group": "Galaxy Masses",
                        "description": "Mass of metals contained within hot gas component"
                        }),
                ("ColdGas", {
                        "type": np.float32,
                        "label": "Cold gas mass",
                        "order": 21,
                        "units": "1e10 solMass/h",
                        "group": "Galaxy Masses",
                        "description": "Mass of cold gas"
                        }),
                ("MetalsColdGas", {
                        "type": np.float32,
                        "label": "Cold gas metals",
                        "order": 22,
                        "units": "1e10 solMass/h",
                        "group": "Galaxy Masses",
                        "description": "Mass of metals contained within cold gas component"
                        }),
                ("Mcool", {
                        "type": np.float32,
                        "label": "Cooling mass",
                        "order": 23,
                        "units": "1e10 solMass/h",
                        "group": "Galaxy Masses",
                        "description": "Mass of baryons cooling onto the galaxy in current timestep"
                        }),
                ("DiskScaleLength", {
                        "type": np.float32,
                        "label": "Disk scale length",
                        "order": 24,
                        "units": "Mpc/h",
                        "group": "Galaxy Properties",
                        "description": "Scale length of the galaxy disk"
                        }),
                ("StellarMass", {
                        "type": np.float32,
                        "label": "Stellar mass",
                        "order": 25,
                        "units": "1e10 solMass/h",
                        "group": "Galaxy Masses",
                        "description": "Galaxy stellar mass"
                        }),
                ("GrossStellarMass", {
                        "type": np.float32,
                        "label": "Gross stellar mass",
                        "order": 26,
                        "units": "1e10 solMass/h",
                        "group": "Galaxy Masses",
                        "description": "Galaxy stellar mass excluding SN mass loss"
                        }),
                ("MetalsStellarMass", {
                        "type": np.float32,
                        "label": "Stellar metals",
                        "order": 27,
                        "units": "1e10 solMass/h",
                        "group": "Galaxy Masses",
                        "description": "Mass of metals in stellar component"
                        }),
                ("Sfr", {
                        "type": np.float32,
                        "label": "Star formation rate",
                        "order": 28,
                        "units": "solMass/yr",
                        "group": "Galaxy Properties",
                        "description": "Galaxy star formation rate"
                        }),
                ("EjectedGas", {
                        "type": np.float32,
                        "label": "Ejected gas",
                        "order": 29,
                        "units": "1e10 solMass/h",
                        "group": "Galaxy Masses",
                        "description": "Mass of gas ejected from system in current timestep"
                        }),
                ("MetalsEjectedGas", {
                        "type": np.float32,
                        "label": "Ejected metals",
                        "order": 30,
                        "units": "1e10 solMass/h",
                        "group": "Galaxy Masses",
                        "description": "Mass of metals in ejected component"
                        }),
                ("BlackHoleMass", {
                        "type": np.float32,
                        "label": "",
                        "order": -1,
                        "units": "1e10 solMass/h",
                        "group": "Internal",
                        "description": "Currently unused"
                        }),
                ("MaxReheatFrac", {
                        "type": np.float32,
                        "label": "Maximum reheated mass fraction",
                        "order": -1,
                        "units": "None",
                        "group": "Internal",
                        "description": "Debugging property"
                        }),
                ("MaxEjectFrac", {
                        "type": np.float32,
                        "label": "Maximum ejected mass fraction",
                        "order": -1,
                        "units": "None",
                        "group": "Internal",
                        "description": "Debugging property"
                        }),
                ("Rcool", {
                        "type": np.float32,
                        "label": "Cooling radius",
                        "order": 31,
                        "units": "Mpc/h",
                        "group": "Galaxy Properties",
                        "description": "Cooling radius"
                        }),
                ("Cos_Inc", {
                        "type": np.float32,
                        "label": "Disk inclination",
                        "order": 32,
                        "units": "None",
                        "group": "Galaxy Properties",
                        "description": "Inclination of galaxy disk (random)"
                        }),
                ("MergTime", {
                        "type": np.float32,
                        "label": "Time remaining until merger",
                        "order": -1,
                        "units": "Myr/h",
                        "group": "Internal",
                        "description": "Time remaining until merger"
                        }),
                ("MergerStartRadius", {
                        "type": np.float32,
                        "label": "Merger start radius",
                        "order": -1,
                        "units": "Mpc/h",
                        "group": "Internal",
                        "description": "Radius at which host halo lost"
                        }),
                ("BaryonFracModifier", {
                        "type": np.float32,
                        "label": "Baryon fraction modifier",
                        "order": 33,
                        "units": "None",
                        "group": "Galaxy Properties",
                        "description": "Fractional modification to the infalling baryon mass due to photoionization supression from the UVB."
                        }),
                ("MvirCrit", {
                        "type": np.float32,
                        "label": "Critical virial mass for photo supression",
                        "order": 34,
                        "units": "1e10 solMass/h",
                        "group": "Galaxy Properties",
                        "description": "The halo mass at which 50% of baronic infall would be supressed, given the intensity of the local UVB."
                        }),
                ("MWMSA", {
                        "type": np.float32,
                        "label": "Mass weighted mean stellar age",
                        "order": 35,
                        "units": "Myr/h",
                        "group": "Galaxy Properties",
                        "description": "The mass weighted mean stellar age of the galaxy"
                        }),
                ("NewStars_0", {
                        "type": np.float32,
                        "label": "New stars",
                        "order": -1,
                        "units": "1e10 solMass/h",
                        "group": "Internal",
                        "description": "The stars formed in this snapshot."
                        }),
                ("NewStars_1", {
                        "type": np.float32,
                        "label": "New stars",
                        "order": -1,
                        "units": "1e10 solMass/h",
                        "group": "Internal",
                        "description": "The stars formed in current snapshot - 1."
                        }),
                ("NewStars_2", {
                        "type": np.float32,
                        "label": "New stars",
                        "order": -1,
                        "units": "1e10 solMass/h",
                        "group": "Internal",
                        "description": "The stars formed in current snapshot - 2."
                        }),
                ("NewStars_3", {
                        "type": np.float32,
                        "label": "New stars",
                        "order": -1,
                        "units": "1e10 solMass/h",
                        "group": "Internal",
                        "description": "The stars formed in current snapshot - 3."
                        }),
                ("NewStars_4", {
                        "type": np.float32,
                        "label": "New stars",
                        "order": -1,
                        "units": "1e10 solMass/h",
                        "group": "Internal",
                        "description": "The stars formed in current snapshot - 4."
                        }),
                ('snapnum', {
                        'type': np.int32,
                        'label': "Snapshot Number",
                        'description': "Snapshot number in the simulation",
                        'group': "Simulation",
                        'order': 36,
                        }),
                ('sfrdisk', {
                        'type': np.float32,
                        'label': "Star Formation Rate in the Disk",
                        'description': "Star formation rate in the disk",
                        'group': "Internal",
                        'units': "Msun/year",
                        'order': -1,
                        }),
                ('sfrbulge', {
                        'type': np.float32,
                        'label': "Star formation Rate in the Bulge",
                        'description': "Star formation rate in the bulge",
                        'group': "Internal",
                        'units': "Msun/year",
                        'order': -1,
                        }),
                ('sfrdiskz', {
                        'type': np.float32,
                        'label': "Avg. Metallicity of Star-forming Disk Gas",
                        'description': "Metallicty of star forming disk gas "\
                            "(averaged over timesteps between two snapshots)"\
                            "(Mass of metals)/(Mass of star forming disk gas)",
                        'group': "Internal",
                        'units': "fraction",
                        'order': -1,
                        }),
                ('sfrbulgez', {
                        'type': np.float32,
                        'label': "Avg. Metallicity of Star-forming Bulge Gas",
                        'description': "Metallicty of star forming bulge gas "\
                            "(averaged over timesteps between two snapshots)"\
                            "(Mass of metals)/(Mass of star forming bulge gas)",
                        'group': "Internal",
                        'units': "fraction",
                        'order': -1,
                        }),
                ('Vpeak', {
                        'type': np.float32,
                        'label': "Maximum circular velocity of the halo",
                        'description': "Maximum circular velocity attained "
                        "in the assembly history (susceptible to spikes "
                        "during mergers, see Vrelax for a better property)",
                        'group': "Halo Properties",
                        'units': "km/s",
                        'order': 37,
                        }),
                 ])

        self.src_fields_dict = src_fields_dict
        self.sim_file = ''
        self.snapshots = []
        super(MERAXESConverter, self).__init__(*args, **kwargs)
    

    @classmethod
    def add_arguments(cls, parser):
        """Adds extra arguments required for MERAXES conversion.

        Extra arguments required for conversion are:
          1. The directory containing MERAXES output hdf5 file.
          2. The filename for the MERAXES hdf5 file.
          3. The name of the simulation (dark matter/hydro).
          4. The name of the galaxy formation model (simulation name in case of hydro)
        """

        parser.add_argument('--trees-dir', default='.',
                            help='location of MERAXES trees')
        parser.add_argument('--meraxes-file', default='meraxes.hdf5',
                            help='name of the MERAXES hdf5 file')
        parser.add_argument('--sim-name', help='name of the dark matter or '
                            'hydro simulation')
        parser.add_argument('--model-name', help='name of the SAM. Set to '
                            'simulation name for a hydro sim')

    def read_input_params(self, fname, quiet=False):
        """ Read in the input parameters from a Meraxes hdf5 output file.

        Parameters
        ----------
        fname : str
            Full path to input hdf5 master file.


        Returns
        -------
        dict
            All run properties.
        """

        def arr_to_value(d):
            for k, v in d.iteritems():
                if v.size is 1:
                    try:
                        d[k] = v[0]
                    except IndexError:
                        d[k] = v

        def visitfunc(name, obj):
            if isinstance(obj, h5.Group):
                props_dict[name] = dict(obj.attrs.items())
                arr_to_value(props_dict[name])


        # Open the file for reading
        fin = h5py.File(fname, 'r')

        group = fin['InputParams']

        props_dict = dict(group.attrs.items())
        arr_to_value(props_dict)
        group.visititems(visitfunc)
        
        fin.close()

        return props_dict

    def get_simfilename(self):
        if not self.args.trees_dir:
            msg = 'Must specify trees directory containing MERAXES hdf5 file'
            raise tao.ConversionError(msg)

        sim_file = '{0}/{1}'.format(self.args.trees_dir,
                                    self.args.meraxes_file)
        return sim_file
        
    def get_simulation_data(self):
        """Extract simulation data.

        Extracts the simulation data from the MERAXES parameter file and
        returns a dictionary containing the values.
        """
        sim_file = self.get_simfilename()
        self.sim_file = sim_file
        params_dict = self.read_input_params(sim_file)
        hubble = params_dict['Hubble_h']
        if hubble < 1.0:
            hubble = hubble * 100.0
        msg = 'Hubble parameter must be in physical units (not little h)'
        assert hubble > 1.0, msg
        sim_data = dict()
        sim_data['box_size'] = params_dict['BoxSize']
        sim_data['hubble'] = hubble
        sim_data['omega_m'] = params_dict['OmegaM']
        sim_data['omega_l'] = params_dict['OmegaLambda']
        print("sim_data = {0}".format(sim_data))
        return sim_data


    def has_tree_counts(self):
        sim_file = self.get_simfilename()
        with h5py.File(sim_file, 'r') as hf:
            keys = hf.keys()
            if 'TreeCounts' in keys:
                return True

        return False

    @profile
    def get_tree_counts(self):
        has_tree_counts = self.has_tree_counts()
        ntrees = OrderedDict()
        nhalos_per_tree_per_snapshot = OrderedDict()
        fn =  self.get_simfilename()
        with h5py.File(fn, "r") as fin:
            lk = list(fin.keys())
            all_snaps = np.asarray([k for k in lk if 'Snap' in k])
            all_snaps.sort()
            ncores = fin.attrs['NCores'][0]
            snap_group = fin[all_snaps[-1]]
            for icore in xrange(ncores):
                this_core_group = 'Core{0:d}'.format(icore)
                if has_tree_counts:
                    n_trees[icore] = snap_group['{0}/NTrees'.\
                                                    format(this_core_group)]
                    nhalos_per_tree[icore] = snap_group['{0}/Treecounts'.\
                                                            format(this_core_group)]
                else:
                    galaxies = snap_group['{0}/Galaxies'.\
                                              format(this_core_group)]
                    forestids = galaxies['ForestID']
                    if len(forestids) > 0:
                        uniq_fids = np.unique(forestids)
                        ntrees[icore] = len(uniq_fids)
                    
                    nhalos_per_tree_per_snapshot[icore] = OrderedDict()
                    for snap in all_snaps:
                        this_snap_group = fin[snap]
                        galaxies = this_snap_group['{0}/Galaxies'.\
                                                       format(this_core_group)]
                        forestids = galaxies['ForestID']
                        if len(forestids) > 0:
                            _, nhalos = np.unique(forestids,
                                                  return_counts=True)
                            nhalos_per_tree_per_snapshot[icore][snap] = nhalos

        print("ntrees = {0}".format(ntrees))
        # print("nhalos_per_tree_per_snapshot = {0}".format(nhalos_per_tree_per_snapshot))
        return ntrees, nhalos_per_tree_per_snapshot
                        
    def get_snapshot_redshifts(self):
        """Parse and convert the expansion factors.

        Uses the expansion factors to calculate snapshot redshifts. Returns
        a list of redshifts in order of snapshots.
        """

        sim_file = self.get_simfilename()
        snaps, redshifts, lt_times = self.read_snaplist(sim_file)
        if len(redshifts) == 0:
            msg = "Could not parse any redshift values in file {0}"\
                .format(sim_file)
            raise tao.ConversionError(msg)
        print("Found {0} redshifts in file {1}".format(len(redshifts),
                                                       sim_file))
        return redshifts

    def get_mapping_table(self):
        """Returns a mapping from TAO fields to MERAXES fields."""

        mapping = {'posx': 'Pos_0',
                   'posy': 'Pos_1',
                   'posz': 'Pos_2',
                   'velx': 'Vel_0',
                   'vely': 'Vel_1',
                   'velz': 'Vel_2',
                   'coldgas': 'ColdGas',
                   'metalscoldgas': 'MetalsColdGas',
                   'diskscaleradius': 'DiskScaleLength',
                   'objecttype': 'Type',
                   }

        return mapping

    def get_extra_fields(self):
        """Returns a list of MERAXES fields and types to include."""
        wanted_field_keys = [
            "id_MBP",
            "ID",
            "snapnum",
            "Type",
            "CentralGal",
            "GhostFlag",
            "Len",
            "MaxLen",
            "PhysicsFlags",
            "Spin",
            "Mvir",
            "Rvir",
            "Vvir",
            "Vmax",
            "FOFMvir",
            "HotGas",
            "MetalsHotGas",
            # "ColdGas",
            # "MetalsColdGas",
            "Mcool",
            "DiskScaleLength",
            "StellarMass",
            "GrossStellarMass",
            "MetalsStellarMass",
            "Sfr",
            "EjectedGas",
            "MetalsEjectedGas",
            "BlackHoleMass",
            "MaxReheatFrac",
            "MaxEjectFrac",
            "Rcool",
            "Cos_Inc",
            "MergTime",
            "MergerStartRadius",
            "BaryonFracModifier",
            "MvirCrit",
            "MWMSA",
            "NewStars_0",
            "NewStars_1",
            "NewStars_2",
            "NewStars_3",
            "NewStars_4",
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
    

    def read_snaplist(self, fname):

        """ Read in the list of available snapshots from the Meraxes hdf5 file.

        Parameters
        ----------
        fname : str
            Full path to input hdf5 master file.


        Returns
        -------
        snaps : array
            snapshots

        redshifts : array
            redshifts

        lt_times : array
            light travel times (Myr)
        """

        zlist = []
        snaplist = []
        lt_times = []

        with h5py.File(fname, 'r') as fin:
            for snap in fin.keys():
                try:
                    zlist.append(fin[snap].attrs['Redshift'][0])
                    snaplist.append(int(snap[-3:]))
                    lt_times.append(fin[snap].attrs['LTTime'][0])
                except KeyError:
                    pass
                
        lt_times = np.array(lt_times, dtype=float)

        return np.array(snaplist, dtype=int), np.array(zlist, dtype=float),\
            lt_times


    @profile
    def map_descendant(self, tree):
        """Calculate the SAGE tree structure.

        SAGE does not output the descendant information in its tree files
        in a directly usable format. To calculate it we need to capitalise
        on the snapshot ordering of the input data, the GalaxyIndex field,
        and the mergeIntoID field.
        """
        print("in map_descendant. len(tree) = {0}".format(len(tree)))
        descs = np.empty(len(tree), np.int32)
        descs.fill(-1)

        """
        Now my attempt at this mapping descendants
        First, sort the entire tree into using GalaxyIndex as
        primary key and then snapshot number as secondary key.
        This sorted indices will naturally flow a galaxy from
        earlier times (lower snapshot numbers) to later times (larger
        snapshot number)
        """
        sorted_ind = np.argsort(tree, order=('ID', 'snapnum'))
        all_gal_idx = tree['ID']
        for ii, idx in enumerate(all_gal_idx[sorted_ind]):
            jj = ii + 1
            if (jj < len(tree)) and (idx == all_gal_idx[sorted_ind[jj]]):
                assert descs[sorted_ind[ii]] == -1
                # assert tree['SnapNum'][sorted_ind[jj]] > \
                #     tree['SnapNum'][sorted_ind[ii]]
                # assert tree['GalaxyIndex'][sorted_ind[ii]] == \
                #     tree['GalaxyIndex'][sorted_ind[jj]]
                descs[sorted_ind[ii]] = sorted_ind[jj]

        # Run validation on descendants
        for ii, desc in enumerate(descs):
            if desc == -1:
                this_galidx = tree['ID'][ii]
                this_snapnum = tree['snapnum'][ii]

                # No descendant -> there can not be any galaxy
                # with the same galaxy index at a higher snapshot
                ind = (np.where((tree['ID'] == this_galidx) &
                                (tree['snapnum'] > this_snapnum)))[0]
                msg = "desc == -1 but real descendant = {0}\n".format(ind)
                if len(ind) != 0:
                    print("tree['ID'][{0}] = {1} at snapshot = {2} "
                          "should be a descendant for ii = {3} with idx = {4} "
                          "at snapshot = {5}".format(
                            ind, tree['ID'][ind],
                            tree['snapnum'][ind], ii,
                            this_galidx, this_snapnum))
                assert len(ind) == 0, msg
            else:
                assert tree['snapnum'][desc] > tree['snapnum'][ii]
                assert tree['ID'][desc] == tree['ID'][ii]

        return descs

    @profile
    def Vpeak(self, tree):
        """
        Calculates the max. of Vmax during the halo history
        """
        vpeak = np.empty(len(tree), np.float32)

        # By pre-filling vpeak with Vmax, I don't have to
        # worry about cases where there are no descendants
        # (although the code should cover that case)
        vpeak[:] = tree['Vmax']

        sorted_ind = np.argsort(tree, order=('ID', 'snapnum'))
        all_vmax = tree['Vmax']
        all_gal_idx = tree['ID']
        vmax = []
        curr_idx = all_gal_idx[sorted_ind[0]]
        for ii, idx in enumerate(all_gal_idx[sorted_ind]):
            if curr_idx != idx:
                vmax = []
                curr_idx = idx

            vmax.append(all_vmax[sorted_ind[ii]])
            vpeak[sorted_ind[ii]] = max(vmax)

        return vpeak

    def sfrdisk(self, tree):
        """
        Returns the disk SFR from MERAXES.

        Assumes *all* SF occurs in disks
        """
        return tree['Sfr']

    def sfrbulge(self, tree):
        """
        Return bulge SFR for MERAXES

        Assume *NO* SF occurs in bulges
        """
        return np.zeros(len(tree))

    def sfrdiskz(self, tree):
        """
        Return avg. metallicity of star forming disk cold gas in
        sub-steps within MERAXES
        
        """
        return tree['MetalsColdGas']

    def sfrbulgez(self, tree):
        """
        Return avg. metallicity of star forming bulge cold gas in
        sub-steps within MERAXES
        """

        return np.zeros(len(tree))

    @profile
    def iterate_trees(self):
        """Iterate over MERAXES trees."""

        computed_fields = {'Vpeak': self.Vpeak,
                           'sfrdisk': self.sfrdisk,
                           'sfrbulge': self.sfrbulge,
                           'sfrdiskz': self.sfrdiskz,
                           'sfrbulgez': self.sfrbulgez,
                           }
        
        computed_field_list = [('snapnum',  self.src_fields_dict['snapnum']['type'])]
        allkeys = [k.lower() for k in self.src_fields_dict.keys()]
        for f in computed_fields:
            if f.lower() not in allkeys:
                assert "Computed field = {0} must still be defined "\
                    "in the module level field_dict".format(f)

            field_dtype_dict = self.src_fields_dict[f]
            computed_field_list.append((f, field_dtype_dict['type']))

        sim_file = self.get_simfilename()
        params = self.read_input_params(sim_file)
        snaps, redshifts, lt_times = self.read_snaplist(sim_file)
        rev_sorted_ind = np.argsort(snaps)[::-1]
        snaps = snaps[rev_sorted_ind]
        redshift = redshifts[rev_sorted_ind]
        lt_times = lt_times[rev_sorted_ind]
        
        ntrees, nhalos_per_tree_per_snapshot = self.get_tree_counts()
        totntrees = sum(ntrees.values())

        with h5py.File(sim_file, "r") as fin:
            ncores = fin.attrs['NCores'][0]
            snap_group = fin['Snap%03d' % snaps[0]]
            file_dtype = snap_group['Core0/Galaxies'].dtype
            ordered_type = []
            for d in file_dtype.descr:
                try:
                    name, typ = d
                    ordered_type.append((name, typ))
                except ValueError:
                    name, typ, shape = d
                    for k in xrange(shape[0]):
                        ordered_type.append(('{0}_{1}'.format(name, k), typ))


        ordered_type.extend(computed_field_list)
        src_type = np.dtype(ordered_type)

        numtrees_processed = 0
        bar = progressbar.ProgressBar(max_value=totntrees)
        print("totntrees = {0}".format(totntrees))

        with h5py.File(sim_file, "r") as fin:
            for icore in xrange(ncores):
                n_trees_this_core = ntrees[icore]
                tree_sizes = np.empty(n_trees_this_core, dtype=np.int64)
                
                snap_tree_offsets = np.zeros(max(snaps) + 1, dtype=np.int64)
                for itree in xrange(n_trees_this_core):
                    tree_size = 0
                    for snap in snaps:
                        try:
                            nhalos = nhalos_per_tree_per_snapshot[icore]['Snap{0:03d}'.format(snap)]
                            if len(nhalos) <= itree:
                                continue

                            tree_size += nhalos[itree]
                            snap_tree_offsets[snap] += nhalos[itree]
                            
                        except KeyError:
                            continue
                        
                    # print("tree_size_over_all_snapshots = {0}".format(tree_size_over_all_snapshots))
                    tree_sizes[itree] = tree_size


                # print("icore = {0}; ntrees = {1} ".format(icore, n_trees_this_core))
                tree_offsets_per_snap = np.zeros(max(snaps)+1,dtype=np.int64)
                for itree in xrange(n_trees_this_core):
                    tree_size = tree_sizes[itree]
                    
                    # print("Working on itree = {0} on core = {1}. Tree size = {2}".format(itree, icore, tree_size))
                    tree = np.empty(tree_size, dtype=src_type)
                    
                    offs = 0
                    for snap in snaps:
                        try:
                            all_tree_chunk_sizes = nhalos_per_tree_per_snapshot[icore]['Snap{0:03d}'.format(snap)]
                            if len(nhalos_per_tree_per_snapshot[icore]['Snap{0:03d}'.format(snap)]) <= itree:
                                # print("itree = {0} snap = {1} --breaking".format(itree, snap))
                                continue
                            
                        except KeyError:
                            continue

                        tree_start_offsets = np.roll(all_tree_chunk_sizes.cumsum(),1)
                        tree_start_offsets[0] = 0
                        
                        chunk_size = all_tree_chunk_sizes[itree]
                        # print("all_tree_chunk_sizes = {0}".format(all_tree_chunk_sizes))
                        # print("cum. chunk size = {0}".format(tree_start_offsets))
                        # print("chunk_size = {0}".format(chunk_size))
                        
                        # print("Reading from 'Snap{0:03d}/Core{1:d}/Galaxies' chunk_size = {2}".
                        #       format(snap, icore, chunk_size))
                        start_offset = tree_start_offsets[itree]
                        galaxies = fin['Snap{0:03d}/Core{1:d}/Galaxies'.
                                       format(snap, icore)]
                        source_sel = np.s_[start_offset: start_offset + chunk_size]
                        dest_sel = np.s_[offs:offs + chunk_size]
                        gal_data = np.empty(chunk_size, file_dtype)
                        galaxies.read_direct(gal_data,
                                             source_sel=source_sel)

                        tree[dest_sel] = gal_data
                        tree[dest_sel]['snapnum'] = snap

                        this_centrals = tree['CentralGal'][dest_sel]
                        centralgalind = (np.where(this_centrals >= 0))[0]
                        if len(centralgalind) > 0:
                            prev_offset = tree_offsets_per_snap[snap]
                            min_this_centrals = min(this_centrals[centralgalind])
                            if (min_this_centrals + offs - prev_offset) < 0:
                                msg = "ERROR: Shifting centralgals will result "\
                                    "in negative indices. min_this_centrals = {0}"\
                                    "offs = {1} prev_offset = {2}"\
                                    .format(min_this_centrals,
                                            offs,
                                            prev_offset)
                                raise ValueError(msg)
                            
                            this_centrals[centralgalind] += (offs - prev_offset)
                        
                        # Fix centralgal offsets
                        tree[dest_sel]['CentralGal'] = this_centrals
                        if len(centralgalind) > 0:
                            if (min(this_centrals[centralgalind]) < offs) or \
                                    (max(this_centrals) >= offs + chunk_size):
                                msg = 'Error: Centrals at snap = {0} must be within '\
                                    'offs = {1} and offs + chunksize = {2}. '\
                                    'Central lies in range [{3}, {4}]. \n'\
                                    'Galaxies["CentralGal"] is in range [{5}, {6}] \n'\
                                    'this_centrals = {7}'\
                                    .format(snap, offs, offs+chunk_size,
                                            min(this_centrals[centralgalind]), max(this_centrals),
                                            min(galaxies['CentralGal']), max(galaxies['CentralGal']),
                                            this_centrals)
                                    
                                raise ValueError(msg)

                        tree_offsets_per_snap[snap] += chunk_size    
                        offs += chunk_size
                        if offs > tree_size:
                            msg = 'For tree = {0}, the start offset can at most be '\
                                'the tree size = {1}. However, offset = {2} has '\
                                'occurred. Bug in code'.format(itree, tree_size,
                                                               offs)
                            raise ValueError(msg)

                    if offs != tree_size:
                        msg = "For tree = {0}, expected to find total number of "\
                            "halos = {1} but during loading found = {2} instead"\
                            .format(itree, tree_size, offs)
                        raise AssertionError(msg)

                    
                    # One tree has been completely loaded (vertical tree now)
                    for fieldname, conv_func in computed_fields.items():
                        tree[fieldname] = conv_func(tree)

                    # First validate some fields.
                    for f in ['Type', 'ID', 'snapnum']:
                        if min(tree[f]) < 0:
                            msg = "ERROR; min(tree[{0}]) = {1} should be non-zero "\
                                .format(f, min(tree[f]))

                            ind = (np.where(tree[f] < 0))[0]
                            msg += "tree[f] = {0}".format(tree[ind][f])
                            msg += "tree[snapnum] = {0}".format(tree[ind]['snapnum'])
                            raise ValueError(msg)
                        
                    # Validate central galaxy index (unique id, generated by sage)
                    centralind = (np.where((tree['Type'] == 0) & (tree['CentralGal'] >= 0)))[0]
                    centralgalind = tree[centralind]['CentralGal']
                    if not bool(np.all(tree['ID'][centralind] ==
                                       tree['ID'][centralgalind])):
                        print("tree[ID][centralind] = {0}".format(tree['ID'][centralind]))
                        print("tree[ID][centralgalind] = {0}".format(tree['ID'][centralgalind]))
                        print("centralind = {0}".format(centralind))
                        print("tree['snapnum'][centralind] = {0}".format(tree['snapnum'][centralind]))
                        print("tree['Type'][centralind] = {0}".format(tree['Type'][centralind]))
                        badind = tree['ID'][centralind] != tree['ID'][centralgalind]
                        print("badind = {0} len(bad) = {1}".format(badind, len(badind)))
                        print("tree[ID][c[b]] = {0}".format(tree['ID'][centralind[badind]]))
                        print("tree[ID][cg[b]] = {0}".format(tree['ID'][centralgalind[badind]]))
                              
                    assert bool(np.all(tree['ID'][centralind] ==
                                       tree['ID'][centralgalind])), \
                                       "Central Galaxy ID must equal GalaxyID for centrals"

                    numtrees_processed += 1
                    bar.update(numtrees_processed)

                    yield tree
