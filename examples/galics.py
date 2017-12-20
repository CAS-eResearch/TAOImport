"""Convert GALICS output to TAO.

A control script to be used with `taoconvert` to convert SAGE output
binary data into HDF5 input for TAO.
"""


import re
import os
import numpy as np
import tao
from collections import OrderedDict
from tqdm import tqdm
from tqdm import trange
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
          4. The name of the galaxy formation model (simulation name in case of hydro)
        """

        parser.add_argument('--trees-dir', default='.',
                            help='location of GALICS trees')
        parser.add_argument('--galics-file', default='Galaxies_snap', 
                            help='Name of the galaxies file (<snapnum>.h5 will be appended to this to create the individual files corresponding to each snapshot)')
        parser.add_argument('--sim-name', help='name of the dark matter or '
                            'hydro simulation')
        parser.add_argument('--model-name', help='name of the SAM. Set to '
                            'simulation name for a hydro sim')

  

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

        fname = open('GalicsSnaphots_list.dat', 'r')
        lines = fname.readlines()
        fname.close()
        
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
        computed_fields = {'GalaxyType': self.GalaxyType,
                           }
        
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
        ! param file : cosmo, box size etc => see read_simulation
        ! ascii snapshots+redshift list
        rev_sorted_ind = np.argsort(snaps)[::-1]
        snaps = snaps[rev_sorted_ind]
        redshift = redshifts[rev_sorted_ind]
        lt_times = lt_times[rev_sorted_ind]
        dt_values = np.ediff1d(lt_times, to_begin=lt_times[0])



        !! 
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

        numtrees_processed = 0
        print("totntrees = {0}".format(totntrees))

        with h5py.File(sim_file, "r") as fin:

            for icore in range(ncores):
                ntrees_this_core = ntrees[icore]
                print("Working on {0} trees on core = {1}".format(ntrees_this_core, icore))
                fin_galaxies_per_snap = dict()
                descendant_fin_per_snap = dict()
                for snap in snaps:
                    fin_galaxies_per_snap[snap] = fin['Snap{0:03d}/Core{1:d}/Galaxies'.
                                                      format(snap, icore)]
                    if snap != max(snaps):
                        descendant_fin_per_snap[snap] = fin['Snap{0:03d}/Core{1:d}/DescendantIndices'.
                                                            format(snap, icore)]

                tree_fids, tree_counts, tree_offsets, tree_first_snap, ngalaxies_per_snap = self.get_tree_counts_and_offset(icore)
                converted_ngalaxies_per_snap = np.zeros(max(snaps) + 1, dtype=np.int64)

                nforests = len(tree_fids)

                # trange is shortcut for tqdm( [x]range(length))
                for iforest in trange(nforests, total=nforests):
                    forest = tree_fids[iforest]
                    
                    # number of galaxies per snapshot for this forest
                    ngalaxies = tree_counts[iforest]

                    # array offset per snapshot for this forest within the
                    # hdf5 dataset
                    offsets = tree_offsets[iforest]

                    # total number of galaxies in the forest
                    # (the sum is over snapshots)
                    tree_size = ngalaxies.sum()
                    if tree_size == 0:
                        msg = "Number of galaxies in forest # {0} with "\
                            "ForestID = {1} is 0. Bug in code"\
                            .format(iforest, forest)
                        print(msg)
                        Tracer()()
                    
                    # print("Working on forest = {0} on core = {1}. Tree size = {2}".format(forest, icore, tree_size))
                    tree = np.empty(tree_size, dtype=src_type)
                    
                    offs = 0
                    first_snap = tree_first_snap[iforest]
                    good_snaps = np.arange(snaps[0], first_snap-1, -1)
                    ngalaxies_future_snap = 0
                    future_snap = -1
                    for snap in good_snaps:
                        ngalaxies_this_snap = tree_counts[iforest, snap]
                        if ngalaxies_this_snap == 0:
                            continue
                        
                        converted_ngalaxies_per_snap[snap] += ngalaxies_this_snap
                        
                        # print("Reading from 'Snap{0:03d}/Core{1:d}/Galaxies' ngalaxies_this_snap = {2}".
                        #       format(snap, icore, ngalaxies_this_snap))
                        galaxies = fin_galaxies_per_snap[snap]
                        start_offset = tree_offsets[iforest, snap] #(vertical_tree_offsets[forest])[snap]
                        #print("snap = {3} forest = {0} ngalaxies = {2} start_offset = {1}".format(forest, start_offset, ngalaxies_this_snap, snap))
                        source_sel = np.s_[start_offset: start_offset + ngalaxies_this_snap]
                        dest_sel = np.s_[offs:offs + ngalaxies_this_snap]
                        gal_data = galaxies[source_sel]
                        if snap != max(good_snaps):
                            descendants = descendant_fin_per_snap[snap]
                            descs = descendants[source_sel]
                            
                            # Fix the descendant offset
                            descs[descs > -1] += (offs - prev_offset -
                                                  ngalaxies_future_snap)
                        else:
                            descs = np.empty(ngalaxies_this_snap, dtype=np.int32)
                            descs[:] = -1

                        
                        if len(gal_data) != ngalaxies_this_snap:
                            Tracer()()

                        tree[dest_sel] = gal_data
                        tree[dest_sel]['snapnum'] = snap
                        tree[dest_sel]['Descendant'] = descs

                        for (fieldname, shape) in array_fields:
                            self.copy_array_fields(gal_data, tree[dest_sel],
                                                   fieldname, shape)

                        this_centrals = tree['CentralGal'][dest_sel]
                        centralgalind = (np.where(this_centrals >= 0))[0]
                        prev_offset = tree_offsets[iforest, snap]
                        if len(centralgalind) > 0:
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
                        
                        # Check and set  centralgal offsets
                        tree[dest_sel]['CentralGal'] = this_centrals
                        if len(centralgalind) > 0:
                            if (min(this_centrals[centralgalind]) < offs) or \
                                    (max(this_centrals) >= offs + ngalaxies_this_snap):
                                msg = 'Error: Centrals at snap = {0} must be within '\
                                    'offs = {1} and offs + chunksize = {2}. '\
                                    'Central lies in range [{3}, {4}]. \n'\
                                    'Galaxies["CentralGal"] is in range [{5}, {6}] \n'\
                                    'this_centrals = {7}'\
                                    .format(snap, offs, offs+ngalaxies_this_snap,
                                            min(this_centrals[centralgalind]), max(this_centrals),
                                            min(galaxies['CentralGal']), max(galaxies['CentralGal']),
                                            this_centrals)
                                    
                                raise ValueError(msg)

                        # The processing is done with the latest snapshot first
                        # and then backwards in time. Thus, ngalaxies_future_snap
                        # contains the number of galaxies in snapshot that has
                        # already been processed
                        future_snap = snap
                        ngalaxies_future_snap = ngalaxies_this_snap
                        future_snap_index_offset = (offs - prev_offset)
                            
                        offs += ngalaxies_this_snap
                        if offs > tree_size:
                            msg = 'For tree = {0}, the start offset can at most be '\
                                'the tree size = {1}. However, offset = {2} has '\
                                'occurred. Bug in code'.format(itree, tree_size,
                                                               offs)
                            raise ValueError(msg)
              
                                      
                    # The entire forest has been loaded
                    if offs != tree_size:
                        msg = "For tree = {0}, expected to find total number of "\
                            "halos = {1} but during loading found = {2} instead"\
                            .format(itree, tree_size, offs)
                        raise AssertionError(msg)

                    # Validate that only one forest has been loaded.
                    min_forestid = min(tree['ForestID'])
                    max_forestid = max(tree['ForestID'])
                    if min_forestid != max_forestid or min_forestid != forest:
                        msg = 'Error during loading a single forest as a '\
                            'vertical tree. Expected to find one single '\
                            'forestID. Instead min(forestID) = {0} and '\
                            'max(forestID) = {1}'.format(min_forestid,
                                                         max_forestid)
                        raise AssertionError(msg)
                    
                    # Fix NAN's in MWMSA (mass-weighted mean stellar age)
                    nan_ind = np.isnan(tree['MWMSA'])
                    tree['MWMSA'][nan_ind] = 0.0
                            
                    # One tree has been completely loaded (vertical tree now)
                    for fieldname, conv_func in computed_fields.items():
                        tree[fieldname] = conv_func(tree)


                    # Since I need to use the descendants now, I have to validate
                    # the indices. For *any* galaxy, the descendant should be 
                    # -1, or a valid index in the next snapshot [0, ngalaxies[nextsnap])
                    # When a valid index is present, is found the galaxy id must
                    # either equal current galaxyid. If the equality is not satisfied
                    # then a merger has occurred and *all* future snapshots including
                    # `nextsnap`  *can not* contain current galaxyid.

                    # Validate Descendants and Populate the "merger" fields required
                    # for the SED module. Needs to be in a converter_function
                    # but 4 fields are being updated by one function. Ideally
                    # the converter function itself should be the key and
                    # the values should the list of fields
                        
                    tree['mergeIntoID'] = -1
                    tree['mergeIntoSnapNum'] = -1
                    tree['mergetype'] = 0
                    for gal in tree:
                        gid, d = gal['ID'], gal['Descendant']
                        # no valid descendant, nothing to verify
                        if d == -1:
                            continue

                        if gid == tree['ID'][d]:
                            # The galaxy continues as itself
                            gal['mergeIntoID'] = -1
                            gal['mergeIntoSnapNum'] = -1
                            gal['mergetype'] = 0
                            continue
                        
                        else:
                            
                            ind = (np.where((tree['snapnum'] > gal['snapnum']) &
                                           (tree['ID'] == gid)))[0]
                            if len(ind) > 0:
                                msg = 'Error: Galaxy with ID = {0} '\
                                     'at snapshot = {1} has descendant ID = '\
                                     '{2} (which is different) but this galaxy '\
                                     'exists at future snapshots. Num Tree matches '\
                                     'in the future snaps = {3}. snaps = {4} with '\
                                     'iD = {5}'.format(gid, gal['snapnum'],
                                                       tree['ID'][d], len(ind),
                                                       tree['snapnum'][ind],
                                                       tree['ID'][ind])
                                Tracer()()
                                raise tao.ConversionError(msg)
                            
                            gal['mergeIntoID'] = tree['ID'][d]
                            gal['mergeIntoSnapNum'] = tree['snapnum'][d]
                            gal['mergetype'] = 2
                    
                    
                    # Populate the field with galaxy ages (required by TAO SED module) 
                    tree['dT'] = dt_values[tree['snapnum']]

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


                    yield tree

                # Now validate that *ALL* galaxies on this core
                # were transferred
                # print("ngalaxies_per_snap = {0}\n".format(ngalaxies_per_snap))
                if not bool(np.all(ngalaxies_per_snap ==
                                   converted_ngalaxies_per_snap)):
                    msg = "Error: Did not convert *all* galaxies for core "\
                        "= {0}.\nExpected to convert = {1} galaxies per "\
                        "snapshot.\nInstead converted = {2} galaxies per "\
                        "snapshot".format(icore, ngalaxies_per_snap,
                                          converted_ngalaxies_per_snap)
                    Tracer()()
                    raise tao.ConversionError(msg)
                    
                        
