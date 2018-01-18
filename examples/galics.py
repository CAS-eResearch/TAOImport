"""Convert GALICS output to TAO.

A control script to be used with `taoconvert` to convert SAGE output
binary data into HDF5 input for TAO.
"""


import re
import os
import numpy as np
import tao
from collections import OrderedDict
from tqdm import tqdm, trange
import h5py
from IPython.core.debugger import Tracer

class GALICSConverter(tao.Converter):
    """Subclasses tao.Converter to perform GALICS output conversion."""

    def __init__(self, *args, **kwargs):
        src_fields_dict = OrderedDict([
                ("galaxyID", {
                        "type": np.int64,
                        "label": "galaxyID",
                        "order": 0,
                        "units": "None",
                        "group": "Galaxy Properties",
                        "description": "Galaxy ID" # unique in the whole simulation
                        }),
                ("g_treeID", {
                        "type": np.int64,
                        "label": "galaxy Tree ID",
                        "order": 1,
                        "units": "None",
                        "group": "Galaxy Properties",
                        "description": "Tree ID" # galaxy tree ID
                        }),
                ("g_haloID", {
                    "type": np.int64,
                    "label": "galaxy's halo ID",
                    "order": 2,
                    "units": "None",
                    "group": "Halo Properties",
                    "description": "Halo ID of the host halo"
                    }),
                ("g_DescendantID", {
                        "type": np.int64,
                        "label": "Descendant ID", # ID of the galaxy's Descendant
                        "order": 3,
                        "units": "None",
                        "group": "Galaxy Properties",
                        "description": "ID of the galaxy's descendant"
                        }),
                ("g_FirstProgenitorID", {
                    "type": np.int64,
                    "label": "FirstProgenitor ID", # ID of the galaxy's FirstProgenitor
                    "order": 4,
                    "units": "None",
                    "group": "Galaxy Properties",
                    "description": "ID of the galaxy's First Progenitor"
                    }),
                ("g_NextProgenitorID", {
                        "type": np.int64,
                        "label": "NextProgenitor ID", # ID of the galaxy's NextProgenitor
                        "order": 5,
                        "units": "None",
                        "group": "Galaxy Properties",
                        "description": "ID of the galaxy's Next Progenitor"
                        }),
                ("g_LastProgenitorID", {
                    "type": np.int64,
                    "label": "LastProgenitor ID", # ID of the galaxy's LastProgenitor
                    "order": 6,
                    "units": "None",
                    "group": "Galaxy Properties",
                    "description": "ID of the galaxy's Last Progenitor"
                    }),
                ("SnapNum", {
                    "type": np.int16,
                    "label": "Snapshot number",
                    "order": 7,
                    "units": "None",
                    "group": "Simulation",
                    "description": "Snapshot number"
                    }),
                ("halo_mvir", {
                        "type": np.float64,
                        "label": "Mvir",
                        "order": 8,
                        "units": "Msun",
                        "group": "Halo Properties",
                        "description": "Virial mass of the host halo"
                        }),
                ("halo_mfof", {
                        "type": np.float64,
                        "label": "Mfof",
                        "order": 9,
                        "units": "Msun",
                        "group": "Halo Properties",
                        "description": "FOF mass of the host halo"
                        }),
                ("halo_rvir", {
                    "type": np.float64,
                    "label": "Rvir",
                    "order": 10,
                    "units": "Mpc",
                    "group": "Halo Properties",
                    "description": "Virial radius of the host halo"
                    }),
                ("disc_mgal", {
                    "type": np.float64,
                    "label": "DiskTotalMass",
                    "order": 11,
                    "units": "Msun",
                    "group": "Galaxy Masses",
                    "description": "Total mass of the galaxy disk"
                    }),
                ("bulge_mgal", {
                    "type": np.float64,
                    "label": "BulgeTotalMass",
                    "order": 12,
                    "units": "Msun",
                    "group": "Galaxy Masses",
                    "description": "Total mass of the galaxy bulge"
                    }),
                 ("disc_mcoldgas", {
                        "type": np.float64,
                        "label": "Disk ColdGas Mass",
                        "order": 13,
                        "units": "Msun",
                        "group": "Galaxy Masses",
                        "description": "Disk ColdGas Mass"
                        }),
                ("bulge_mcoldgas", {
                        "type": np.float64,
                        "label": "Bulge ColdGas Mass",
                        "order": 14,
                        "units": "Msun",
                        "group": "Galaxy Masses",
                        "description": "Bulge ColdGas Mass"
                        }),
                ("disc_mstar", {
                        "type": np.float64,
                        "label": "Disk Stellar Mass",
                        "order": 15,
                        "units": "Msun",
                        "group": "Galaxy Masses",
                        "description": "Disk Stellar Mass"
                        }),
                ("bulge_mstar", {
                        "type": np.float64,
                        "label": "Bulge Stellar Mass",
                        "order": 16,
                        "units": "Msun",
                        "group": "Galaxy Masses",
                        "description": "Bulge Stellar Mass"
                        }),
                ("disc_mcold_metals", {
                        "type": np.float64,
                        "label": "Disk MetalsMass",
                        "order": 17,
                        "units": "Msun",
                        "group": "Galaxy Masses",
                        "description": "Disk Metals Mass"
                        }),
                ("bulge_mcold_metals", {
                    "type": np.float64,
                    "label": "Disk MetalsMass",
                    "order": 18,
                    "units": "Msun",
                    "group": "Galaxy Masses",
                    "description": "Bulge Metals Mass"
                    }),
                ("disc_scalelength", {
                        "type": np.float64,
                        "label": "DiskScaleLength",
                        "order": 19,
                        "units": "Mpc",
                        "group": "Galaxy Properties",
                        "description": "Scalelength of the disk"
                        }),
                ("bulge_scalelength", {
                        "type": np.float64,
                        "label": "BulgeScaleLength",
                        "order": 20,
                        "units": "Mpc",
                        "group": "Galaxy Properties",
                        "description": "Scalelength of the bulge"
                        }),
                ("sfr_disk", {
                        "type": np.float64,
                        "label": "SfrDisk",
                        "order": 21,
                        "units": "Msun/yr",
                        "group": "Galaxy Properties",
                        "description": "SFR of the disk"
                        }),
                ("sfr_bulge", {
                        "type": np.float64,
                        "label": "SfrBulge",
                        "order": 22,
                        "units": "Msun/yr",
                        "group": "Galaxy Properties",
                        "description": "SFR of the bulge"
                        }),
                ("nb_merg", {
                        "type": np.int16,
                        "label": "Number of mergers",
                        "order": 23,
                        "units": "None",
                        "group": "Galaxy Properties",
                        "description": "Number of (minor and major) mergers"
                        }),
                ("Delta_t", {
                        "type": np.float64,
                        "label": "Delta_t",
                        "order": 24,
                        "units": "Myr",
                        "group": "Galaxy Properties",
                        "description": "Duration of the SF episode"
                        }),
                ("x_pos", {
                        "type": np.float64,
                        "label": "X position",
                        "order": 25,
                        "units": "Mpc",
                        "group": "Positions & velocities",
                        "description": "X Position of the galaxy in the box" # fake value
                        }),
               ("y_pos", {
                        "type": np.float64,
                        "label": "Y position",
                        "order": 26,
                        "units": "Mpc",
                        "group": "Positions & velocities",
                        "description": "Y Position of the galaxy in the box" # fake value
                        }),
                ("z_pos", {
                        "type": np.float64,
                        "label": "Z position",
                        "order": 27,
                        "units": "Mpc",
                        "group": "Positions & velocities",
                        "description": "Z Position of the galaxy in the box" # fake value
                        }),
                ("x_vel", {
                        "type": np.float64,
                        "label": "X velocity",
                        "order": 28,
                        "units": "km/s",
                        "group": "Positions & velocities",
                        "description": "X velocity of the galaxy in the box" # fake value
                        }),
                ("y_vel", {
                        "type": np.float32,
                        "label": "Y velocity",
                        "order": 29,
                        "units": "km/s",
                        "group": "Positions & velocities",
                        "description": "Y velocity of the galaxy in the box" # fake value
                        }),
                ("z_vel", {
                        "type": np.float64,
                        "label": "Z velocity",
                        "order": 30,
                        "units": "km/s",
                        "group": "Positions & velocities",
                        "description": "Z velocity of the galaxy in the box" # fake value
                        }),
                ("GalaxyType", {
                        "type": np.int16,
                        "label": "Galaxy type",
                        "order": 31,
                        "units": "None",
                        "group": "Galaxy Properties",
                        "description": "0:central, 2: satellite" # fake value
                        })
                 ])

        self.src_fields_dict = src_fields_dict
        self.sim_file = ''
        self.snapshots = []
        super(GALICSConverter, self).__init__(*args, **kwargs)
    

    @classmethod
    def add_arguments(cls, parser):
        """Adds extra arguments required for GALICS conversion.

        Extra arguments required for conversion are:
          1. The directory containing GALICS output hdf5 file.
          2. The filename for the GALICS hdf5 file.
          3. The name of the simulation (dark matter/hydro).
          4. The name of the galaxy formation model (default 'Galics')
          5. The first snapshot to process (default 2)
          6. The last snapshot to process  (default 94)
          7. A boolean flag whether processing galaxies (default True, set to False to process halos)
          
        """
        parser.add_argument('--trees-dir', default='.',
                            help='location of GALICS trees')
        parser.add_argument('--galics-filename', default='Galaxies_snap', 
                            help='Name of the galaxies file (<snapnum>.h5 '\
                                'will be appended to this to create the '\
                                'individual files corresponding to each '\
                                'snapshot)')
        parser.add_argument('--sim-name', help='name of the dark matter or '
                            'hydro simulation')
        parser.add_argument('--model-name', default='Galics',
                            help='name of the SAM. Set to '\
                                'simulation name for a hydro sim')
        parser.add_argument('--firstsnap', default=2,
                            help='The first snapshot to process')
        parser.add_argument('--lastsnap', default=94,
                            help='The last snapshot to process')
        parser.add_argument('--processing-galaxies', default=True,
                            help='A boolean flag to indicate that '\
                                'galaxies are being processed. Set to False '\
                                'to process halos')
  

    def get_simfilename(self, snapnum):
        if not self.args.trees_dir:
            msg = 'Must specify trees directory containing GALICS hdf5 file'
            raise tao.ConversionError(msg)

        sim_file = '{0}/{1}{2:0d}.h5'.format(self.args.trees_dir,
                                    self.args.galics_file, snapnum)
        return sim_file
        
    def get_simulation_data(self):
        """Extract simulation data.

        Extracts the simulation data from the GALICS parameter file and
        returns a dictionary containing the values.
        """
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


    
    # get_snapshot_redshifts to be modified to read snap z
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
        """Returns a mapping from TAO fields to GALICS fields."""

        mapping = {'posx': 'x_pos',
                   'posy': 'y_pos',
                   'posz': 'z_pos',
                   'velx': 'x_vel',
                   'vely': 'y_vel',
                   'velz': 'z_vel',
                   'coldgas': 'disc_mcoldgas',
                   'metalscoldgas': 'disc_mcold_metals',
                   'diskscaleradius': 'disc_scalelength',
                   'objecttype': 'GalaxyType',
                   'dt': 'Delta_t',
                   }

        return mapping

    # Only prop to show in tao
    def get_extra_fields(self):
        """Returns a list of GALICS fields and types to include."""
        wanted_field_keys = [
            "galaxyID",
            "g_treeID",
            "g_haloID",
            "g_DescendantID",
            "g_FirstProgenitorID",
            "g_NextProgenitorID",
            "SnapNum",
            "halo_mvir",
            "halo_mfof",
            "halo_rvir",
            "disc_mgal",
            "bulge_mgal",
            "disc_mcoldgas",
            "bulge_mcoldgas",
            "disc_mstar",
            "bulge_mstar",
            "disc_mcold_metals",
            "bulge_mcold_metals",
            "disc_scalelength",
            "bulge_scalelength",
            "sfr_disk",
            "sfr_bulge",
            "nb_merg",
            "Delta_t",
            "x_pos",
            "y_pos",
            "z_pos",
            "x_vel",
            "y_vel",
            "z_vel",
            "GalaxyType",
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

        """ Read in the list of available snapshots from the GalicsSnaphots_list.dat file.

        Parameters
        ----------
        fname : str
            GalicsSnaphots_list.dat

        Returns
        -------
        snaplist : array
            snapshots list

        zlist : array
            redshifts

        """

        with open('GalicsSnaphots_list.dat', 'r') as f:
            lines = f.readlines()
        
        snaplist = []
        zlist    = []
        for line in lines:
            p = line.split()
            snaplist.append(int(p[0]))
            zlist.append(np.float64(p[1]))
               
        return np.array(snaplist, dtype=int), np.array(zlist, dtype=np.float64)


    def GalaxyType(self, tree):
        """
        Return a fake galaxy type (all=0) for GALICS galaxies
        
        """
        return np.zeros(len(tree), dtype=np.int16)



    def generate_galics_filename(inputdir, hdf5base, snapnum):
        return '{0}/{1}{2}.h5'.format(inputdir, hdf5base, snapnum)


    def check_and_open_all_tree_files(inputdir, hdf5base, snapshots):
        '''
        Create a dictionary of h5py file handles opened for reading.

        The dictionary key is the snapshot number and the returned value
        is the h5py descriptor. The input is a list of snapshot numbers
        to locate
        '''
        hf_files = dict()

        # Check that all the files exist
        for snapnum in snapshots:
            # Check that snapnum is of type integer
            test_snapnum = snapnum
            try:
                test_snapnum += 1
            except TypeError:
                message = 'Error: Snapshot number={0} should be an integer type'\
                    .format(snapnum)
                print(message)
                raise

            fname = generate_galics_filename(inputdir, hdf5base, snapnum)
            try:
                # File exists open the file, and append to dict
                # with the snapnum as key
                hf = h5py.File(fname, 'r')
                hf_files[snapnum] = hf
            except IOError:
                print("Could not locate the file at snapshot = {0}. "
                      "Generated filename was = {1}".format(snapnum, fname))
                raise

        return hf_files


    def validate_treeid_assumptions(hf_files, snapshots, tree_id_field):
        '''
        Validate all assumptions about the TREEID. The assumptions are:

        1. TREEID uniquely identifies a tree across all snapshots (i.e., if
        two halos/galaxies share the same TREEID, then they are in the same
        tree). Not sure how to validate this!

        2. Galaxies in a TREE are written in sequential order on a per snapshot
        basis. All galaxies from a particular tree are written out sequentially
        (i.e., there will not be any inter-mixing of galaxies belonging
        to different trees).

        3. The ordering of the trees is set by the TREEID order at the last
        snapshot (usually z=0). For instance, if a tree with TREEID=XXX appears
        before another tree with TREEID=YYYY, then *if* the TREEID=XXX tree is
        present at any previous snapshot, then that XXX tree will be before the
        TREEID=YYY. However, if TREEID=XXX ceases to exist, then TREEID=YYY will
        appear in such a way that the relative ordering of trees at the last
        snapshot is preserved.

        What other assumptions are being made? -> Thibault

        i) All GALAXYIDS should be unique.
        ii) GALAXYID of descendant should be descendantid
        iii) TREEIDS at z=0 should be unique
        iv) GALAXYTREEIDS should also be unique


        '''

        # No error checking for now
        return True

    def read_tree_ids(uniq_tree_ids, tree_id_field,
                      snapshots, hf_files):

        for snapnum in tqdm(snapshots):
            hf = hf_files[snapnum]

            # Read-in *ALL* the treeids at this snapshot
            treeids = hf['Galics_output'][tree_id_field][:]



    @profile
    def get_start_and_stop_indices_per_tree(uniq_tree_ids, tree_id_field,
                                            snapshots, hf_files):

        '''

        Returns 2D arrays of shape (ntrees, lastsnap + 1) containing the
        starting and stopping indices per tree per snapshot. The `start`
        (and `stop`) indices are returned in the same order as the supplied
        `unique tree ids`. 

        '''
        ntrees = len(uniq_tree_ids)
        lastsnap = max(snapshots)
        start_indices = np.empty((ntrees, lastsnap + 1), dtype=np.int)
        stop_indices = np.empty((ntrees, lastsnap + 1), dtype=np.int)
        ngalaxies_per_tree_per_snapshot = np.zeros((ntrees, lastsnap + 1),
                                                   dtype=np.int)

        start_indices[:] = -1
        stop_indices[:] = -1

        for snapnum in tqdm(snapshots):
            hf = hf_files[snapnum]

            # Read-in *ALL* the treeids at this snapshot
            treeids = hf['Galics_output'][tree_id_field][:]

            # Now match the treeids at this snapshot to the
            # unique treeids
            for treenum, tid in tqdm(enumerate(uniq_tree_ids), total=ntrees):
                ind = (np.where(treeids == tid))[0]

                # if there is no match, that means that the tree
                # does not exist at this snapshot
                ngals = len(ind)
                if len(ind) == 0:
                    continue

                message = "\nThe array indices for each tree should be sequential "\
                    "but TREEID = {0} at snapshot = {1}\nseems to be stored "\
                    "over non-sequential indices. Check if the input GALICS "\
                    "data are okay".format(tid, snapnum)

                min_ind = ind.min()
                max_ind = ind.max()
                if (max_ind - min_ind) != (ngals-1):
                    message += "\nThe difference between the max index and min "\
                        "index for galaxies belonging to the same tree must be\n"\
                        "the same length as the number of galaxies in that tree "\
                        "at that snapshot. However, I find max. index = {0} and\n"\
                        "min. index = {1} while the number of galaxies = {2}."\
                        .format(max_ind, min_ind, ngals)
                    message += "\nind = {0}\n".format(ind)
                    raise AssertionError(message)


                # Sort the array - might not be necessary
                # but a sorted array is required for any sequential test
                # sorting catches any repeated indices which might pass
                # the previous (max_ind - min_ind) == (ngals - 1) test
                ind.sort()

                # Check that the array is sequential
                # -> take a difference of consecutive elements in the
                # sorted array. This difference should be exactly equal
                # to 1.
                diff1d = np.ediff1d(ind)

                # There was a numpy bug where np.all did not return
                # boolean but respected the datatype of the input array
                # but I only want a PASS/FAIL
                assert bool(np.all(diff1d == 1)) == True, message

                # All checks have passed -> now store the start/stop/ngal indices
                start_indices[treenum, snapnum] = min_ind
                stop_indices[treenum, snapnum] = max_ind + 1 # Slice indices are not inclusive
                ngalaxies_per_tree_per_snapshot[treenum, snapnum] = ngals

        return start_indices, \
            stop_indices, \
            ngalaxies_per_tree_per_snapshot
    
    
    # leave it here but dont touch
    def copy_array_fields(self, src_tree, dest_tree, fieldname, shape):
        """
        Copies fields that are arrays into individual elements
        per array.

        for example, pos[3] gets copied to pos_0, pos_1, pos_2

        More generally, field[shape] gets copied into
        field_0, field_1, ..field_(shape-1)
        """

        if len(src_tree) != len(dest_tree):
            msg = 'The source and destination arrays must be of the same'\
                'size. Source size = {0} dest. size = {1}'.format(
                len(src_tree), len(dest_tree))
            raise tao.ConversionError(msg)
        
        # Get the field
        for (fld, dst) in zip(src_tree[fieldname], dest_tree):
            for axis in range(shape[0]):
                dest_fieldname = '{0}_{1}'.format(fieldname, axis)
                dst[dest_fieldname] = fld[axis]
        
    

    @profile
    def iterate_trees(self):
        """Iterate over GALICS trees."""
        # I need to "compute" the galaxy type here (fake values: all=0)
        computed_fields = {'GalaxyType': self.GalaxyType}
        
        computed_field_list = [('snapnum',  self.src_fields_dict['snapnum']['type']),
                               ('mergeIntoID', self.src_fields_dict['mergeIntoID']['type']),
                               ('mergeIntoSnapNum', self.src_fields_dict['mergeIntoSnapNum']['type']),
                               ('mergetype', self.src_fields_dict['mergetype']['type']),
                               ('dT', self.src_fields_dict['dT']['type']),
                               ('Descendant', self.src_fields_dict['Descendant']['type']),
                               ]
        
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
        #param file : cosmo, box size etc => see read_simulation
        #ascii snapshots+redshift list
        rev_sorted_ind = np.argsort(snaps)[::-1]
        snaps = snaps[rev_sorted_ind]
        redshift = redshifts[rev_sorted_ind]
        lt_times = lt_times[rev_sorted_ind]
        dt_values = np.ediff1d(lt_times, to_begin=lt_times[0])


        
        ntrees = self.get_ntrees()
        totntrees = sum(ntrees.values())

        array_fields = []
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
                    array_fields.append((name, shape))
                    for k in range(shape[0]):
                        ordered_type.append(('{0}_{1}'.format(name, k), typ))

        ordered_type.extend(computed_field_list)
        src_type = np.dtype(ordered_type)


        # First create the list snapshots, the '- 1' in the second parameter
        # is required otherwise the first snapshot will not be included.
        # This is intentionally ordered from the last snapshot backwards
        # because that's how the vertical tree structure is usually processed
        snapshots = np.arange(lastsnap, firstsnap - 1, step=-1, dtype=np.int)

        # Now check if all the relevant files exist and open them with h5py
        # Result is a dictionary of h5py file handles
        hf_files = check_and_open_all_tree_files(inputdir, hdf5base, snapshots)

        # Are we processing galaxies or halos -> set the Unique ID field
        # appropriately
        if processing_galaxies:
            unique_id_field = 'GALAXYID'
            tree_id_field = 'G_TREEID'
            desc_id_field = 'G_DESCENDANTID'
        else:
            unique_id_field = 'HALOID'
            tree_id_field = 'TREEID'
            desc_id_field = 'DESCENDANTID'

        # Files are open -> validate the tree assumptions are not being violated
        # This is a PASS/FAIL test.
        validate_treeid_assumptions(hf_files, snapshots, tree_id_field)

        # Get the list of TREEIDS at the last snapshot
        hf = hf_files[lastsnap]
        input_type = hf['Galics_output'].dtype
        treeids = hf['Galics_output'][tree_id_field][:]

        # Find the unique set of TREEIDS. These unique tree ids will
        # provide the basis for processing the data one tree at a time.
        uniq_tree_ids = np.unique(treeids)
        ntrees = uniq_tree_ids.size

        # We have an array of unique tree ids -> let's first compute the starting
        # and stopping indices for each tree at each snapshot
        # start_indices, \
        #     stop_indices, \
        #     ngalaxies_per_tree_per_snapshot = \
        #     get_start_and_stop_indices_per_tree(uniq_tree_ids, tree_id_field,
        #                                         snapshots, hf_files)

        read_tree_ids(uniq_tree_ids, tree_id_field, snapshots, hf_files)

        return

        # Get the number of galaxies per tree -> this is simply a sum (per tree)
        # over the number of galaxies per snapshot
        ngalaxies_per_tree = ngalaxies_per_tree_per_snapshot.sum(axis=1)

        # Check that snapshots is reverse sorted
        # Because the descendants are matched by DescendantID
        # -> the descendant galaxy *MUST* already be loaded. This
        # can only happen in the snapshots are processed in reverse
        # sorted order, i.e., last snapshot is processed first and then
        # we go back in time.
        message = "BUG: Snapshots must be processed from the last snapshot "\
            "to the first"
        assert bool(np.all(np.ediff1d(snapshots) < 0)), message


        # Verify that unique IDs are actually unique
        ids = hf['Galics_output'][unique_id_field]
        unique_ids = np.unique(ids)
        message = "Unique IDs are expected for the field = `{0}` "\
            "But the array size is = {1} while the number of unique "\
            "values = {2}".format(unique_id_field, len(ids),
                                  len(unique_ids))

        assert len(unique_ids) == len(unique_ids), message

        # Create the source datatype based on all the fields in the input
        # + any other computed fields etc that might be getting pulled through
        # INCORRECTLY set to input type for now:
        src_type = input_type

        # Tree validation passed -> process one tree at a time. 
        for treenum, tid in tqdm(enumerate(uniq_tree_ids), total=ntrees):
            # print("Working on treenum = {0}".format(treenum))

            # First create the numpy array to hold the vertical tree
            # The input data type must be that specified in the hdf5 file
            tree = np.empty(ngalaxies_per_tree[treenum], dtype=src_type)

            # For each tree, loop over all snapshots
            # First create the array with number of
            # galaxies per snapshot for this tree
            ngals_per_snap = ngalaxies_per_tree_per_snapshot[treenum, :]
            message = "Error: Number of elements in the array should be "\
                "equal to the last snapshot number + 1. This allows direct "\
                "indexing by the snapshot number. However, the shape is "\
                "{0} rather than {1}==(lastsnap + 1)".format(len(ngals_per_snap),
                                                             lastsnap + 1)
            assert len(ngals_per_snap) == (lastsnap + 1), message

            # The offset in the vertical tree -> the
            # starting index for storing at a snapshot
            offs = 0
            for snapnum in snapshots:
                hf = hf_files[snapnum]
                galaxies = hf['Galics_output']
                start = start_indices[treenum, snapnum]
                stop = stop_indices[treenum, snapnum]
                ngalaxies_this_snap = ngals_per_snap[snapnum]

                ## Gather the individual fields 
                source_sel = np.s_[start:stop]
                dest_sel = np.s_[offs:offs + ngalaxies_this_snap]
                gal_data = galaxies[source_sel]
                message = 'BUG: Range of array indices=[{0}, {1}] do not '\
                    'agree with the number of galaxies expected = {2}.\n'\
                    'Please check the function `get_start_and_stop_indices_per_tree`'\
                    .format(start, stop, ngalaxies_this_snap)
                assert (stop - start) == ngalaxies_this_snap, message

                descs = np.empty(ngalaxies_this_snap, dtype=np.int32)
                descs[:] = -1

                if snapnum < lastsnap:
                    descids = gal_data[desc_id_field]

                    # Find the objects (halos/galaxies) that do have
                    # descendants (usually descid is set to -1 to
                    # signify no descendant but I am checking for
                    # non-zero positive values as a conservative method)
                    good_desc = (np.where(descids >= 0))[0]
                    if len(good_desc) > 0:
                        # now match descendantids to the unique ids at the
                        # next snapshot
                        for s, d in zip(good_desc, descids[good_desc]):
                            ii = (np.where(future_snap_uniqueid == d))[0]
                            # Is there a match?
                            if len(ii) == 0:
                                message = "Error: Could not find descendant for "\
                                    "galaxy with ID = {0} at snapshot = {1}. "\
                                    "descendantID = {2} at the next snapshot"\
                                    .format(gal_data[s][unique_id_field],
                                            snapnum, d)
                                raise AssertionError(message)

                            # Is there exactly one match?
                            if len(ii) > 1:
                                message = "Error: Found multiple descendants for "\
                                    "galaxy with ID = {0} at snapshot = {1}. "\
                                    "descendantID = {2} at the next snapshot\n"\
                                    "Matched indices = {3}"\
                                    .format(gal_data[s][unique_id_field],
                                            snapnum, d, ii)
                                raise AssertionError(message)

                            descs[s] = ii

                # All the data have been read-in and calculated
                # assign to the existing arrays
                message = 'gal_data = {0}\ndest_sel={1}\nsource_sel={2}'\
                    .format(gal_data, dest_sel, source_sel)
                assert gal_data.shape == tree[dest_sel].shape, message
                tree[dest_sel] = gal_data
                # tree[dest_sel]['snapnum'] = snapnum
                # tree[dest_sel]['Descendant'] = descs

                # Store the Unique IDs to match for descendants
                # at the previous snapshot (which will be the
                # next iteration)
                future_snap_uniqueid = gal_data[unique_id_field]

                # Update the number of galaxies read in so far
                # for this tree
                offs += ngalaxies_this_snap

                # print("tree = {0}".format(tree))
                # print("Working on treenum = {0}....done".format(treenum))
                # yield tree


        # Close all the open 'snapshot' files
        for _, hf in hf_files.items():
            hf.close()

