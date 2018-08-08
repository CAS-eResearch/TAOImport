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
import warnings
from os.path import join as pjoin


class GALICSConverter(tao.Converter):
    """Subclasses tao.Converter to perform GALICS output conversion."""

    def __init__(self, *args, **kwargs):
        src_fields_dict = OrderedDict([
                ("GALAXYID", {
                        "type": np.int64,
                        "label": "galaxyID",
                        "order": 0,
                        "units": "None",
                        "group": "Galaxy Properties",
                        "description": "Galaxy ID" # unique in the whole simulation
                        }),
                ("G_TREEID", {
                        "type": np.int64,
                        "label": "galaxy Tree ID",
                        "order": 1,
                        "units": "None",
                        "group": "Galaxy Properties",
                        "description": "Tree ID" # galaxy tree ID
                        }),
                ("G_HALOID", {
                    "type": np.int64,
                    "label": "galaxy's halo ID",
                    "order": 2,
                    "units": "None",
                    "group": "Halo Properties",
                    "description": "Halo ID of the host halo"
                    }),
                ("G_DESCENDANTID", {
                        "type": np.int64,
                        "label": "Descendant ID", # ID of the galaxy's Descendant
                        "order": 3,
                        "units": "None",
                        "group": "Galaxy Properties",
                        "description": "ID of the galaxy's descendant"
                        }),
                ("G_FIRSTPROGENITORID", {
                    "type": np.int64,
                    "label": "FirstProgenitor ID", # ID of the galaxy's FirstProgenitor
                    "order": 4,
                    "units": "None",
                    "group": "Galaxy Properties",
                    "description": "ID of the galaxy's First Progenitor"
                    }),
                ("G_NEXTPROGENITORID", {
                        "type": np.int64,
                        "label": "NextProgenitor ID", # ID of the galaxy's NextProgenitor
                        "order": 5,
                        "units": "None",
                        "group": "Galaxy Properties",
                        "description": "ID of the galaxy's Next Progenitor"
                        }),
                ("G_LASTPROGENITORID", {
                    "type": np.int64,
                    "label": "LastProgenitor ID", # ID of the galaxy's LastProgenitor
                    "order": 6,
                    "units": "None",
                    "group": "Galaxy Properties",
                    "description": "ID of the galaxy's Last Progenitor"
                    }),
                ("SNAPNUM", {
                    "type": np.int16,
                    "label": "Snapshot number",
                    "order": 7,
                    "units": "None",
                    "group": "Simulation",
                    "description": "Snapshot number"
                    }),
                ("HALO_MVIR", {
                        "type": np.float64,
                        "label": "Mvir",
                        "order": 8,
                        "units": "Msun",
                        "group": "Halo Properties",
                        "description": "Virial mass of the host halo"
                        }),
                ("HALO_MFOF", {
                        "type": np.float64,
                        "label": "Mfof",
                        "order": 9,
                        "units": "Msun",
                        "group": "Halo Properties",
                        "description": "FOF mass of the host halo"
                        }),
                ("HALO_RVIR", {
                    "type": np.float64,
                    "label": "Rvir",
                    "order": 10,
                    "units": "Mpc",
                    "group": "Halo Properties",
                    "description": "Virial radius of the host halo"
                    }),
                ("DISC_MGAL", {
                    "type": np.float64,
                    "label": "DiskTotalMass",
                    "order": 11,
                    "units": "Msun",
                    "group": "Galaxy Masses",
                    "description": "Total mass of the galaxy disk"
                    }),
                ("BULGE_MGAL", {
                    "type": np.float64,
                    "label": "BulgeTotalMass",
                    "order": 12,
                    "units": "Msun",
                    "group": "Galaxy Masses",
                    "description": "Total mass of the galaxy bulge"
                    }),
                 ("DISC_MCOLDGAS", {
                        "type": np.float64,
                        "label": "Disk ColdGas Mass",
                        "order": 13,
                        "units": "Msun",
                        "group": "Galaxy Masses",
                        "description": "Disk ColdGas Mass"
                        }),
                ("BULGE_MCOLDGAS", {
                        "type": np.float64,
                        "label": "Bulge ColdGas Mass",
                        "order": 14,
                        "units": "Msun",
                        "group": "Galaxy Masses",
                        "description": "Bulge ColdGas Mass"
                        }),
                ("DISC_MSTAR", {
                        "type": np.float64,
                        "label": "Disk Stellar Mass",
                        "order": 15,
                        "units": "Msun",
                        "group": "Galaxy Masses",
                        "description": "Disk Stellar Mass"
                        }),
                ("BULGE_MSTAR", {
                        "type": np.float64,
                        "label": "Bulge Stellar Mass",
                        "order": 16,
                        "units": "Msun",
                        "group": "Galaxy Masses",
                        "description": "Bulge Stellar Mass"
                        }),
                ("DISC_MCOLD_METALS", {
                        "type": np.float64,
                        "label": "Disk MetalsMass",
                        "order": 17,
                        "units": "Msun",
                        "group": "Galaxy Masses",
                        "description": "Disk Metals Mass"
                        }),
                ("BULGE_MCOLD_METALS", {
                    "type": np.float64,
                    "label": "Disk MetalsMass",
                    "order": 18,
                    "units": "Msun",
                    "group": "Galaxy Masses",
                    "description": "Bulge Metals Mass"
                    }),
                ("DISC_SCALELENGTH", {
                        "type": np.float64,
                        "label": "DiskScaleLength",
                        "order": 19,
                        "units": "Mpc",
                        "group": "Galaxy Properties",
                        "description": "Scalelength of the disk"
                        }),
                ("BULGE_SCALELENGTH", {
                        "type": np.float64,
                        "label": "BulgeScaleLength",
                        "order": 20,
                        "units": "Mpc",
                        "group": "Galaxy Properties",
                        "description": "Scalelength of the bulge"
                        }),
                ("SFR_DISK", {
                        "type": np.float64,
                        "label": "SfrDisk",
                        "order": 21,
                        "units": "Msun/yr",
                        "group": "Galaxy Properties",
                        "description": "SFR of the disk"
                        }),
                ("SFR_BULGE", {
                        "type": np.float64,
                        "label": "SfrBulge",
                        "order": 22,
                        "units": "Msun/yr",
                        "group": "Galaxy Properties",
                        "description": "SFR of the bulge"
                        }),
                ("NB_MERG", {
                        "type": np.int16,
                        "label": "Number of mergers",
                        "order": 23,
                        "units": "None",
                        "group": "Galaxy Properties",
                        "description": "Number of (minor and major) mergers"
                        }),
                ("DELTA_T", {
                        "type": np.float64,
                        "label": "Delta_t",
                        "order": 24,
                        "units": "Myr",
                        "group": "Galaxy Properties",
                        "description": "Duration of the SF episode"
                        }),
                ("X_POS", {
                        "type": np.float64,
                        "label": "X position",
                        "order": 25,
                        "units": "Mpc",
                        "group": "Positions & velocities",
                        "description": "X Position of the galaxy in the box" # fake value
                        }),
               ("Y_POS", {
                        "type": np.float64,
                        "label": "Y position",
                        "order": 26,
                        "units": "Mpc",
                        "group": "Positions & velocities",
                        "description": "Y Position of the galaxy in the box" # fake value
                        }),
                ("Z_POS", {
                        "type": np.float64,
                        "label": "Z position",
                        "order": 27,
                        "units": "Mpc",
                        "group": "Positions & velocities",
                        "description": "Z Position of the galaxy in the box" # fake value
                        }),
                ("X_VEL", {
                        "type": np.float64,
                        "label": "X velocity",
                        "order": 28,
                        "units": "km/s",
                        "group": "Positions & velocities",
                        "description": "X velocity of the galaxy in the box" # fake value
                        }),
                ("Y_VEL", {
                        "type": np.float32,
                        "label": "Y velocity",
                        "order": 29,
                        "units": "km/s",
                        "group": "Positions & velocities",
                        "description": "Y velocity of the galaxy in the box" # fake value
                        }),
                ("Z_VEL", {
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
                        }),
                ("mergeIntoID", {
                        "type": np.int32,
                        "label": "Descendant Galaxy Index",
                        "description": "Index for the descendant galaxy "\
                            "after a merger",
                        "group": "Internal",
                        "order": 32,
                        }),
                ("mergeIntoSnapNum", {
                        "type": np.int32,
                        "label": "Descendant Snapshot",
                        "description": "Snapshot for the descendant galaxy",
                        "group": "Internal",
                        "order": 33,
                        }),
                ("mergetype", {
                        "type": np.int32,
                        "label": "Merger Type",
                        "description": "Merger type: "\
                            "0=none; 1=minor merger; 2=major merger; "\
                            "3=disk instability; 4=disrupt to ICS",
                        "group": "Internal",
                        "order": 34,
                        }),
                # ("dT", {
                #         "type": np.float32,
                #         "label": "Galaxy Age",
                #         "group": "Internal",
                #         "order": 35,
                #         }),
                ("Descendant", {
                        'description': 'Tree-local index of the descendant ',
                        'type': np.int32,
                        'group': "Internal",
                        'order': 36,
                        }),
                 ])

        self.src_fields_dict = src_fields_dict
        self.cosmo_params = None
        self.hubble = None
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
                            'hydro simulation', default='Lyon Simulations')
        parser.add_argument('--model-name', default='Galics',
                            help='name of the SAM. Set to '\
                                'simulation name for a hydro sim')
        parser.add_argument('--firstsnap', default=2,
                            help='The first snapshot to process (inclusive)')
        parser.add_argument('--lastsnap', default=7,
                            help='The last snapshot to process (inclusive)')
        parser.add_argument('--processing-galaxies', default=True,
                            help='A boolean flag to indicate that '\
                                'galaxies are being processed. Set to False '\
                                'to process halos')
  

    def get_simfilename(self, snapnum):
        if not self.args.trees_dir:
            msg = 'Must specify trees directory containing GALICS hdf5 file'
            raise tao.ConversionError(msg)
        
        sim_file = '{0}/{1}{2:0d}.h5'.format(self.args.trees_dir,
                                             self.args.galics_filename,
                                             snapnum)
        return sim_file


    def read_input_params(self,
                          fn='../galicsInputFiles/simulation_data.dat'):
        filename = pjoin(self.args.trees_dir, fn)
        print("filename in read input params = {0}".format(filename))
        with open(filename, 'r') as f:
            props_dict = dict()
            for line in f:
                line = line.strip()
                if line.startswith("#"): continue

                line = line.partition('#')[0]
                key, value = line.split('=')

                key = key.strip()
                value = value.strip()

                props_dict[key] = np.float64(value)

        return props_dict

        
    def get_simulation_data(self):
        """Extract simulation data.

        Extracts the simulation data from the GALICS parameter file and
        returns a dictionary containing the values. Called by tao.Converter
        """
        params_dict = self.read_input_params()
        self.cosmo_params = params_dict
        hubble = params_dict['Hubble_h']
        if hubble < 1.0:
            hubble = hubble * 100.0
        msg = 'Hubble parameter must be in physical units (not little h)'
        assert hubble > 1.0, msg
        self.hubble = hubble
        sim_data = dict()
        sim_data['box_size'] = params_dict['BoxSize']
        sim_data['hubble'] = hubble
        sim_data['omega_m'] = params_dict['OmegaM']
        sim_data['omega_l'] = params_dict['OmegaLambda']
        print("sim_data = {0}".format(sim_data))
        return sim_data

    
    # get_snapshot_redshifts to be modified to read snap z
    # Called by tao converter
    def get_snapshot_redshifts(self):
        """Parse and convert the expansion factors.

        Uses the expansion factors to calculate snapshot redshifts. Returns
        a list of redshifts in order of snapshots.
        """

        snaps, redshifts, lt_times = self.read_snaplist()
        if len(redshifts) == 0:
            msg = "Could not parse any redshift values"\
                .format(sim_file)
            raise tao.ConversionError(msg)
        print("Found {0} redshifts".format(len(redshifts)))

        return redshifts

    def get_mapping_table(self):
        """Returns a mapping from TAO fields to GALICS fields."""

        mapping = {'posx': 'X_POS',
                   'posy': 'Y_POS',
                   'posz': 'Z_POS',
                   'velx': 'X_VEL',
                   'vely': 'Y_VEL',
                   'velz': 'Z_VEL',
                   'coldgas': 'DISC_MCOLDGAS',
                   'metalscoldgas': 'DISC_MCOLD_METALS',
                   'diskscaleradius': 'DISC_SCALELENGTH',
                   'objecttype': 'GalaxyType',
                   'dt': 'DELTA_T',
                   'snapnum':'SNAPNUM',
                   'descendant':'Descendant',
                   'sfrdisk': 'SFR_DISK',
                   'sfrbulge': 'SFR_BULGE',
                   'sfrdiskz': 'DISC_MCOLD_METALS',
                   'sfrbulgez': 'BULGE_MCOLD_METALS',
                   }

        return mapping


    # Only prop to show in tao
    def get_extra_fields(self):
        """Returns a list of GALICS fields and types to include."""
        wanted_field_keys = [
            "GALAXYID",
            "G_TREEID",
            "G_HALOID",
            "G_DESCENDANTID",
            "G_FIRSTPROGENITORID",
            "G_NEXTPROGENITORID",
            "HALO_MVIR",
            "HALO_MFOF",
            "HALO_RVIR",
            "DISC_MGAL",
            "BULGE_MGAL",
            "BULGE_MCOLDGAS",
            "DISC_MSTAR",
            "BULGE_MSTAR",
            "BULGE_MCOLD_METALS",
            "BULGE_SCALELENGTH",
            "SFR_DISK",
            "SFR_BULGE",
            "NB_MERG",
            "DELTA_T",
            "mergeIntoID",
            "mergetype",
            "mergeIntoSnapNum",
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
    

    def read_snaplist(self,
                      fn="../galicsInputFiles/galics_with_lttimes.dat"):
        
        """ Read in the list of available snapshots from the GalicsSnaphots_list.dat file.

        Parameters
        ----------
        fname : string
                default="../galicsInputFiles/GalicsSnaphots_list.dat"

        Returns
        -------
        snaplist : array
            snapshots list

        zlist : array
            redshifts

        """
        from os.path import join as pjoin
        fname = pjoin(self.args.trees_dir,
                      fn)
        
        snaplist = []
        zlist    = []
        lt_times = []
        with open(fname, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'): continue
                
                p = line.split()
                snaplist.append(p[0])
                zlist.append(p[1])
                lt_times.append(p[2])

                
        return np.array(snaplist, dtype=np.int), \
            np.array(zlist, dtype=np.float64),\
            np.array(lt_times, dtype=np.float64)


    def GalaxyType(self, tree):
        """
        Return a fake galaxy type (all=0) for GALICS galaxies
        
        """
        return np.zeros(len(tree), dtype=np.int16)



    def generate_galics_filename(self, inputdir, hdf5base, snapnum):
        
        return '{0}/{1}{2}.h5'.format(inputdir, hdf5base, snapnum)


    def check_and_open_all_tree_files(self, inputdir, hdf5base, snapshots):
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

            fname = self.generate_galics_filename(inputdir, hdf5base, snapnum)
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


    def validate_treeid_assumptions(self, hf_files, snapshots, tree_id_field):
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

    @profile
    def find_start_stop_from_treeids(self, treeids, snapnum,
                                     start_stop_indices):
        # Create the arrays to hold in 
        uniq_tids_this_snap, uniq_index, uniq_counts = np.unique(treeids,
                                                                 return_index=True,
                                                                 return_counts=True)
        ## Check that there isn't a bug in np.unique
        assert uniq_counts.sum() == len(treeids), "Bug in numpy unique implementation"

        desc="Calculating start and stop indices"
        for tid, ui, uc in tqdm(zip(uniq_tids_this_snap, uniq_index, uniq_counts), desc=desc):
            ## MS: for a list x, ``x.count(x[0]) == len(x)`` might be the
            ## fastest check for *all* identical elements rather than
            ## the ``set`` implementation used here

            # The set implementation
            # yy = set(treeids[ui:ui+uc] - tid)
            # if len(yy) !=1 or 0 not in yy:

            # the np.all implementation
            # if not np.all(treeids[ui:ui+uc] == tid):

            # the np.unique implementation
            # if not np.unique(treeids[ui:ui+uc]) == [tid]:
            
            # the list.count implementation
            yy = list(treeids[ui:ui+uc])
            if yy.count(tid) != uc:
                msg = "\nThe array indices for each tree should be "\
                    "stored contiguously at each snapshot. But "\
                    "TREEID = {0} at snapshot = {1}\nseems to be stored "\
                    "over non-sequential indices. Check if the input GALICS "\
                    "data are okay. yy = {2} yy.count(yy[0]) = {3} uc = {4}\n"\
                    .format(tid, snapnum, yy, yy.count(yy[0]), uc)
                raise tao.validators.ValidationError(msg)

            # For this tree at this snapshot, store the start
            # and stop indices as a tuple. The key to the dictionary
            # is also a tuple of treeid + snapshot number
            try:
                _ =start_stop_indices[(tid, snapnum)]
                msg = "\nError: treeid = {1} already has been processed at "\
                    "snapshot = {0}. Please check in the input GALICS data "\
                    "are okay\n".format(tid, snapnum)
                raise tao.validators.ValidationError(msg)
            except KeyError:
                start_stop_indices[(tid, snapnum)] = (ui, ui+uc)

        return uniq_tids_this_snap
    
    @profile
    def find_unique_trees(self, tree_id_field,
                          snapshots, hf_files):
        r"""
        Read the tree ID fields to find the number and
        treeids for unique trees
        
        """
        lastsnap = max(snapshots)
        msg = "Error: Bug in code. The snapshots have to be processed "\
            "in reverse order. Snapshots at later times need to be "\
            "processed before earlier snapshots (i.e., the snapshots "\
            "array has to be sorted in decreasing order). Currently, "\
            "the snapshots array contains: {0}".format(snapshots)
        if lastsnap != snapshots[0]:
            raise tao.validators.ValidationError(msg)


        start_stop_indices = dict()
        uniq_treeids = list()
        
        # Loop over the snapshots 
        desc = "Calculating indices per snapshot for unique trees"
        for snapnum in tqdm(snapshots, total=lastsnap+1, desc=desc):
            hf = hf_files[snapnum]
            treeids = hf['Galics_output'][tree_id_field][:]
            uniq_tids_this_snap = self.find_start_stop_from_treeids(treeids, snapnum,
                                                                    start_stop_indices)


            if snapnum == lastsnap:
                uniq_treeids = list(uniq_tids_this_snap)
                final_snap = [lastsnap] * len(uniq_tids_this_snap)
                continue
            
            # Sorted 1D array of values in `ar1` that are not in `ar2`
            new_tids = np.setdiff1d(uniq_tids_this_snap,
                                    uniq_treeids, assume_unique=True)
            if len(new_tids) > 0:
                uniq_treeids.extend(new_tids)
                new_snaps = [snapnum] * len(new_tids)
                final_snap.extend(new_snaps)


        # Now that we know the number of unique trees, we can calculate the
        # number of galaxies in each tree at each snapshot
        ntrees = len(uniq_treeids)
        ngalaxies_per_tree_per_snapshot = np.zeros((ntrees, lastsnap + 1),
                                                   dtype=np.int)
        firstsnap = min(snapshots)
        desc="Creating ngalaxies per tree array"
        for treenum, tid in tqdm(enumerate(uniq_treeids), desc=desc, total=ntrees):
            final_snap_this_tree = final_snap[treenum]
            for snapnum in range(firstsnap, final_snap_this_tree + 1):
                try:
                    start, stop = start_stop_indices[(tid, snapnum)] 
                    ngalaxies_per_tree_per_snapshot[treenum, snapnum] = stop - start
                except KeyError:
                    # There should be at least one dictionary hit (i.e., the tree must
                    # exist somewhere). However, since we are *always* starting from
                    # `firstsnap`, there will be cases where the tree does not exist
                    # at `snapnum`
                    msg = "\nError: treeid = {0} should exist at final snapshot = {1}\n"\
                        "Please file bug report on the git repo.".format(tid, snapnum)
                    if snapnum == lastsnap:
                        raise tao.validators.ValidationError(msg)
                    pass

                                 
        return uniq_treeids,\
            final_snap, \
            start_stop_indices, \
            ngalaxies_per_tree_per_snapshot
        
    
    def get_start_and_stop_indices_per_tree(self, ntrees, uniq_treeids,
                                            tree_id_field,
                                            snapshots, hf_files):

        r"""

        Returns 2D arrays of shape (ntrees, lastsnap + 1) containing the
        starting and stopping indices per tree per snapshot. The `start`
        (and `stop`) indices are returned in order in which unique tree
        ids are present in the file.

        """
        lastsnap = max(snapshots)
        msg = "Error: Bug in code. The snapshots have to be processed "\
            "in reverse order. Snapshots at later times need to be "\
            "processed before earlier snapshots (i.e., the snapshots "\
            "array has to be sorted in decreasing order). Currently, "\
            "the snapshots array contains: {0}".format(snapshots)
        if lastsnap != snapshots[0]:
            raise tao.validators.ValidationError(msg)

        start_indices = np.empty((ntrees, lastsnap + 1), dtype=np.int)
        stop_indices = np.empty((ntrees, lastsnap + 1), dtype=np.int)
        ngalaxies_per_tree_per_snapshot = np.zeros((ntrees, lastsnap + 1),
                                                   dtype=np.int)

        start_indices[:] = -1
        stop_indices[:] = -1


        # For validation, maintain a set of treeids processed during the
        # entire simulation. Note that this is meant to catch the case
        # when a tree present in uniq_treeids is not processed by this
        # function. The other case, where a "new" tree is being processed
        # (i.e., a treeid not present in `uniq_treeids`) will cause a KeyError
        # and crash the code. 
        seen_treeids_allsnaps = set()

        # Create a dictionary that converts from treeid to arrayindex
        # This is simply the ordering of treeids at the lastsnapshot
        # Requires an OrderedDict to preserve the file order for
        # treeids
        ntrees = len(uniq_treeids)
        treeid_to_array_index = OrderedDict(zip(uniq_treeids,
                                                np.arange(ntrees))
                                            )
        desc="Finding tree indices"    
        for snapnum in tqdm(snapshots, total=lastsnap + 1, desc=desc):
            hf = hf_files[snapnum]

            seen_tids_this_snap = set()
            start_off = 0
            treeids = hf['Galics_output'][tree_id_field]
            totngal_this_snap = treeids.shape[0]

            ## Re-think this while loop -> this should be possible to
            ## do with np.unique(treeids, return_inverse=True, return_counts=True)

            desc = "Working on snapshot {0}".format(snapnum)
            pbar = tqdm(total=totngal_this_snap, desc=desc)
            while start_off < totngal_this_snap:
                # print("Working on tree = {0} ntrees = {1} ngal = {2} "\
                #        "start offset = {3}"
                #       .format(treeindex, ntrees,
                #               totngal_this_snap, start_off))

                # start processing new tree
                curr_treeid = treeids[start_off]

                # Make sure that we have not seen this treeid
                # at this snapshot already! (Validate requirement
                # that treeids are stored contiguously at a given
                # snapshot
                msg = "\nThe array indices for each tree should be "\
                    "stored contiguously at each snapshot. But "\
                    "TREEID = {0} at snapshot = {1}\nseems to be stored "\
                    "over non-sequential indices. Check if the input GALICS "\
                    "data are okay.\n"\
                    .format(curr_treeid, snapnum)
                if curr_treeid in seen_tids_this_snap:
                    raise tao.validators.ValidationError(msg)
                
                end_off = start_off + 1
                while (end_off < totngal_this_snap) and \
                        (treeids[end_off] == curr_treeid):
                    end_off += 1

                try:
                    treeindex = treeid_to_array_index[curr_treeid]
                except KeyError:
                    msg = "\nCould not find the array index for "\
                        "treeid = {0} at snapshot = {1}. Check for a bug "\
                        "in function ``".format(curr_treeid, snapnum)
                    raise tao.validators.ValidationError(msg)
            
                msg = "Error: Bug in code. Tree index = {0} for "\
                    "treeid = {1} should be smaller than the "\
                    "total number of trees = {2}. Please file "\
                    "an issue on the TAOImport github repo.\n"\
                    .format(treeindex, curr_treeid, ntrees)
                if treeindex >= ntrees:
                    raise tao.validators.ValidationError(msg)
                
                # Assign the values
                start_indices[treeindex, snapnum] = start_off
                stop_indices[treeindex, snapnum] = end_off
                ngalaxies_per_tree_per_snapshot[treeindex, snapnum] = end_off - start_off 

                pbar.update(end_off - start_off)

                # Now update the start offset
                start_off = end_off

                # the `set` seen_tids_this_snap is reset at every snapshot
                # and is used to make sure that a treeid appears
                # contiguously within one snapshot
                seen_tids_this_snap.add(curr_treeid)

                # The set `seen_treeids_allsnaps` is present to check
                # that exactly `ntrees` were processed
                if curr_treeid not in seen_treeids_allsnaps:
                    seen_treeids_allsnaps.add(curr_treeid)
                
            pbar.close()
            
        # Check that the number of trees processed is the
        # same as the number of unique trees (passed as the `ntrees`
        # parameter). 
        msg = "Error: The number of unique treeids = {0} found in "\
            "the last snapshot does not equal the "\
            "`ntrees` parameter = {1} passed into the function"\
            .format(len(seen_treeids_allsnaps), ntrees)
        if len(seen_treeids_allsnaps) != ntrees:
            raise tao.validators.ValidationError(msg)
                    
        # Create the treeid_file_order variable containing the treeids
        # in the order in which they appear in the file
        # Check that treeid_to_array_index is an OrderedDict
        msg = "Error: The variable containing the treeids must be an "\
            "OrderedDict, found {0} instead. Change the initialization "\
            "for `treeid_to_array_index` variable to fix this issue"\
            .format(type(treeid_to_array_index))
        if not isinstance(treeid_to_array_index, collections.OrderedDict):
            raise tao.validators.ValidationError(msg)
        

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
        
    

    # @profile
    def iterate_trees(self):
        """Iterate over GALICS trees."""

        # I need to "compute" the galaxy type here (fake values: all=0)
        fields_computed_with_function = {'GalaxyType': self.GalaxyType}
        
        computed_field_list = [('mergeIntoID', self.src_fields_dict['mergeIntoID']['type']),
                               ('mergeIntoSnapNum', self.src_fields_dict['mergeIntoSnapNum']['type']),
                               ('mergetype', self.src_fields_dict['mergetype']['type']),
                               # ('dT', self.src_fields_dict['dT']['type']),
                               ('Descendant', self.src_fields_dict['Descendant']['type']),
                               ]

        lastsnap = self.args.lastsnap
        firstsnap = self.args.firstsnap
        sim_file = self.get_simfilename(lastsnap)
        snaps, redshifts, lt_times = self.read_snaplist()
        rev_sorted_ind = np.argsort(snaps)[::-1]
        snaps = snaps[rev_sorted_ind]
        redshift = redshifts[rev_sorted_ind]
        lt_times = lt_times[rev_sorted_ind]
        dt_values = np.ediff1d(lt_times, to_begin=lt_times[0])

        ## Figure out the input data-type
        allkeys = [k.lower() for k in self.src_fields_dict.keys()]
        for f in fields_computed_with_function:
            if f.lower() not in allkeys:
                assert "Computed field = {0} must still be defined "\
                    "in the module level field_dict".format(f)

            field_dtype_dict = self.src_fields_dict[f]
            computed_field_list.append((f, field_dtype_dict['type']))

        ## Since this is an hdf5 file, we can directly query the
        ## data-type    
        array_fields = []
        with h5py.File(sim_file, "r") as fin:
            file_dtype = fin['Galics_output'].dtype
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

        # for cf in computed_field_list:
        #     if cf not in ordered_type.keys():
        #         ordered_type.extend(cf)
                        
        ordered_type.extend(computed_field_list)
                        
        
        print("ordered_type is {0}".format(ordered_type))
        src_type = np.dtype(ordered_type)
        print("Input src_type is {0}".format(src_type))

        ## Check that computed fields are being pulled through, either via
        ## mapping table or get_extra_fields
        ## MS 13/2/2018: This check is failing for `dT` -> this is potentially a
        ## convention issue because Tibo has Delta_t. Must check and confirm
        ## with Tibo and Darren as to what might be appropriate. 
        mapping_table = (self.get_mapping_table()).keys()
        extra_fields = (self.get_extra_fields()).keys()
        all_fields_carried_through = mapping_table
        all_fields_carried_through.extend(extra_fields)
        for (cf, _) in computed_field_list:
            if cf not in all_fields_carried_through:
                msg = "Computed field = `{0}` is not being carried through to "\
                    "TAO. Please add `{0}` into either the `get_mapping_table` "\
                    "function or `get_extra_fields`.\nFor reference:\n"\
                    "mapping table is: `{1}`\n"\
                    "Extra fields is: `{2}`\n"\
                    "all_fields_carried_through = `{3}`"\
                    .format(cf, mapping_table,
                            extra_fields,
                            all_fields_carried_through)
                warnings.warn(msg, Warning)
        

        # First create the list snapshots, the '- 1' in the second parameter
        # is required otherwise the first snapshot will not be included.
        # This is intentionally ordered from the last snapshot backwards
        # because that's how the vertical tree structure is usually processed
        ## MS: 28th Feb 2018: Snapshots do not necessarily have to start at 0
        snapshots = np.arange(lastsnap, firstsnap - 1, step=-1, dtype=np.int)
        # print("snaps = {0}".format(snaps))
        # snapshots = np.array(snaps, dtype=np.int)
        # print("snapshots arange = {0}".format(np.arange(lastsnap, firstsnap - 1, step=-1, dtype=np.int)))
        
        # Now check if all the relevant files exist and open them with h5py
        # Result is a dictionary of h5py file handles
        hf_files = self.check_and_open_all_tree_files(self.args.trees_dir,
                                                      self.args.galics_filename,
                                                      snapshots)

        # Are we processing galaxies or halos -> set the Unique ID field
        # appropriately
        if self.args.processing_galaxies:
            unique_id_field = 'GALAXYID'
            tree_id_field = 'G_TREEID'
            desc_id_field = 'G_DESCENDANTID'
        else:
            unique_id_field = 'HALOID'
            tree_id_field = 'TREEID'
            desc_id_field = 'DESCENDANTID'

        # Files are open -> validate the tree assumptions are not being violated
        # This is a PASS/FAIL test.
        self.validate_treeid_assumptions(hf_files, snapshots, tree_id_field)

        # Get the list of TREEIDS at the last snapshot
        hf = hf_files[lastsnap]
        input_type = hf['Galics_output'].dtype
        treeids = hf['Galics_output'][tree_id_field][:]

        # Find the unique set of TREEIDS. These unique tree ids will
        # provide the basis for processing the data one tree at a time.
        uniq_tree_ids = np.unique(treeids)

        # The other horizontal tree, MERAXES, has ntrees as an array
        # with each element denoting the number of trees per file (within
        # MERAXES, each MPI core writes out one file. GALICS is different --
        # there is only one file per snapshot)

        # Since some trees might disappear before reaching the final
        # snapshot, there will be at least `ntrees`
        ntrees = uniq_tree_ids.size

        tree_id_file_order, final_snap_treeids, start_stop_indices, ngalaxies_per_tree_per_snapshot = self.find_unique_trees(tree_id_field,
                                                                                                                             snapshots,
                                                                                                                             hf_files)
        
        # This is true number of trees we have to process
        ntrees = len(tree_id_file_order)

        # # We have an array of unique tree ids -> let's first compute the starting
        # # and stopping indices for each tree at each snapshot
        # start_indices, stop_indices, ngalaxies_per_tree_per_snapshot = \
        #     self.get_start_and_stop_indices_per_tree(ntrees,
        #                                              tree_id_file_order,
        #                                              tree_id_field,
        #                                              snapshots, hf_files)
            
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

        assert len(unique_ids) == len(ids), message

        # Tree validation passed -> process one tree at a time. 
        for treenum, tid in tqdm(enumerate(tree_id_file_order), total=ntrees):
            # print("Working on treenum = {0}".format(treenum))

            # First create the numpy array to hold the vertical tree
            # The input data type must be that specified in the hdf5 file
            tree_size = ngalaxies_per_tree[treenum]
            tree = np.empty(tree_size, dtype=src_type)

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

                # Does the tree exist at this snapshot?
                # the array index (`treenum`) is exactly
                # in-sync with `tree_id_file_order`
                if final_snap_treeids[treenum] < snapnum:
                    continue
                
                hf = hf_files[snapnum]
                galaxies = hf['Galics_output']
                try:
                    start, stop = start_stop_indices[(tid, snapnum)]
                except KeyError:
                    # tree does not exist at this snapshot -> continue
                    continue
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
                            
                            # ii only refers to the snapshot local index, but we need
                            # an index that spans the entire range of the tree.
                            # The indices for ids at the previous snapshot will
                            # start at `prev_offs` (the value of `offs` at the
                            # previous snapshot). Therefore, the effective index for
                            # the descendant is given below:
                            descs[s] = ii + prev_offs

                # All the data have been read-in and calculated
                # assign to the existing arrays
                message = 'gal_data = {0}\ndest_sel={1}\nsource_sel={2}'\
                    .format(gal_data, dest_sel, source_sel)
                assert gal_data.shape == tree[dest_sel].shape, message
                tree[dest_sel] = gal_data
                tree[dest_sel]['SNAPNUM'] = snapnum
                tree[dest_sel]['Descendant'] = descs

                # Store the Unique IDs to match for descendants
                # at the previous snapshot (which will be the
                # next iteration)
                future_snap_uniqueid = gal_data[unique_id_field]
                prev_offs = offs

                # Update the number of galaxies read in so far
                # for this tree
                offs += ngalaxies_this_snap

            # Done reading in the entire tree
            if offs != tree_size:
                msg = "For tree = {0}, expected to find total number of "\
                    "halos = {1} but during loading found = {2} instead"\
                    .format(treenum, tree_size, offs)
                raise AssertionError(msg)
            
            ## Convert positions to co-moving Mpc/h
            for fld in ['X_POS', 'Y_POS', 'Z_POS']:
                tree[fld][:] = tree[fld][:] * (self.hubble * 0.01)
            
            # One tree has been completely loaded (vertical tree now)
            for fieldname, conv_func in fields_computed_with_function.items():
                tree[fieldname] = conv_func(tree)


            ## The following are copied from meraxes -> adapt to
            ## GALICS convention
            # Now assign the remaining fields 
            tree['mergeIntoID'] = -1
            tree['mergeIntoSnapNum'] = -1
            tree['mergetype'] = 0
            for gal in tree:
                gid, d = gal['GALAXYID'], gal['Descendant']
                # no valid descendant, nothing to verify
                if d == -1:
                    continue

                if gid == tree['GALAXYID'][d]:
                    # The galaxy continues as itself
                    gal['mergeIntoID'] = -1
                    gal['mergeIntoSnapNum'] = -1
                    gal['mergetype'] = 0
                    continue

                else:

                    ind = (np.where((tree['SNAPNUM'] > gal['SNAPNUM']) &
                                   (tree['GALAXYID'] == gid)))[0]
                    if len(ind) > 0:
                        msg = 'Error: Galaxy with ID = {0} '\
                             'at snapshot = {1} has descendant ID = '\
                             '{2} (which is different) but this galaxy '\
                             'exists at future snapshots. Num Tree matches '\
                             'in the future snaps = {3}. snaps = {4} with '\
                             'iD = {5}'.format(gid, gal['SNAPNUM'],
                                               tree['GALAXYID'][d], len(ind),
                                               tree['SNAPNUM'][ind],
                                               tree['GALAXYID'][ind])
                        Tracer()()
                        raise tao.ConversionError(msg)

                    gal['mergeIntoID'] = tree['GALAXYID'][d]
                    gal['mergeIntoSnapNum'] = tree['SNAPNUM'][d]
                    gal['mergetype'] = 2


            # # Populate the field with galaxy ages (required by TAO SED module) 
            # tree['dT'] = dt_values[tree['SNAPNUM']]

            # First validate some fields.
            for f in ['GalaxyType', 'GALAXYID', 'SNAPNUM']:
                if min(tree[f]) < 0:
                    msg = "ERROR; min(tree[{0}]) = {1} should be non-zero "\
                        .format(f, min(tree[f]))

                    ind = (np.where(tree[f] < 0))[0]
                    msg += "tree[f] = {0}".format(tree[ind][f])
                    msg += "tree[snapnum] = {0}".format(tree[ind]['SNAPNUM'])
                    raise ValueError(msg)

            # # Validate central galaxy index (unique id, generated by sage)
            # centralind = (np.where((tree['GalaxyType'] == 0) & (tree['CentralGal'] >= 0)))[0]
            # centralgalind = tree[centralind]['CentralGal']
            # if not bool(np.all(tree['GALAXYID'][centralind] ==
            #                    tree['GALAXYID'][centralgalind])):
            #     print("tree[ID][centralind] = {0}".format(tree['GALAXYID'][centralind]))
            #     print("tree[ID][centralgalind] = {0}".format(tree['GALAXYID'][centralgalind]))
            #     print("centralind = {0}".format(centralind))
            #     print("tree['SNAPNUM'][centralind] = {0}".format(tree['SNAPNUM'][centralind]))
            #     print("tree['GalaxyType'][centralind] = {0}".format(tree['GalaxyType'][centralind]))
            #     badind = tree['GALAXYID'][centralind] != tree['GALAXYID'][centralgalind]
            #     print("badind = {0} len(bad) = {1}".format(badind, len(badind)))
            #     print("tree[ID][c[b]] = {0}".format(tree['GALAXYID'][centralind[badind]]))
            #     print("tree[ID][cg[b]] = {0}".format(tree['GALAXYID'][centralgalind[badind]]))

            # assert bool(np.all(tree['GALAXYID'][centralind] ==
            #                    tree['GALAXYID'][centralgalind])), \
            #                    "Central Galaxy ID must equal GalaxyID for centrals"

                
            # print("tree = {0}".format(tree))
            # print("Working on treenum = {0}....done".format(treenum))
            yield tree


        # Close all the open 'snapshot' files
        for _, hf in hf_files.items():
            hf.close()

