from __future__ import print_function
import numpy as np
import time
from .library import library
from .Exporter import Exporter
from .Mapping import Mapping
from .xml import get_settings_xml
# from IPython.core.debugger import Tracer
from collections import OrderedDict
import os

from datetime import datetime

class ConversionError(Exception):
    pass

class Converter(object):

    def __init__(self, modules, args):
        self.modules = modules
        self.args = args
        self.MPI = None
        table = self.get_mapping_table()
        fields = self.get_extra_fields()
        self.mapping = Mapping(self, table, fields)
        self.make_datatype()
        for mod in self.modules:
            mod.mapping = self.mapping

    def combine_and_append_keys(self, old_dict, new_dict):
        # lc -> lower-case
        lc_old_keys = [k.lower() for k in old_dict.keys()]
        lc_new_keys = [k.lower() for k in new_dict.keys()]
        updated_dict = old_dict

        # Take the new values
        for lc_new_key, key in zip(lc_new_keys, new_dict.keys()):
            # if lc_new_key not in lc_old_keys:
            #     print "adding a new key = '{0}' with val = '{1}' to dict"\
            #         .format(key, new_dict[key])
            # else:
            #     print "updating key = '{0}' with val = '{1}'. Old value = '{2}' "\
            #         .format(key, new_dict[key], updated_dict[key])

            # Update the dictionary
            updated_dict[key] = new_dict[key]

        return updated_dict

    def make_datatype(self):
        all_fields = []
        seen_fields = set()
        metadata = OrderedDict()
        import pprint
        pp = pprint.PrettyPrinter(indent=4)

        # print "BEGINNING self.mapping.fields = {0}".format(self.mapping.fields)
        for mod in self.modules:
            # print "inside converter make_dataypes - mod = {0}".format(mod)
            if mod.disabled:
                continue
            
            new_fields, meta_field = mod.get_numpy_fields()

            for nf in new_fields:
                lower_case_field = nf[0].lower()
                if lower_case_field not in seen_fields:
                    all_fields.append(nf)
                    seen_fields.add(lower_case_field)
                    metadata[lower_case_field] = meta_field[nf[0]]
                    # print "nf = {0}, d = {1}".format(nf, meta_field[nf[0]])
                else:
                    # Update the dictionary with (possible) newer info
                    old_d = metadata[lower_case_field]
                    new_d = meta_field[nf[0]]
                    # print "Calling combine keys for mod = {0} field = {1} "\
                    #     .format(mod, nf[0])

                    # print "old dict = {0}\nnew_dict = {1}".format(old_d, new_d)
                    updated_d = self.combine_and_append_keys(old_d, new_d)
                    metadata[lower_case_field] = updated_d

        mapped_keys_lower = [k.lower() for k in self.mapping.table.keys()]
        for mf, d in self.mapping.fields.iteritems():
            lower_case_field = mf.lower()
            assert lower_case_field not in mapped_keys_lower,\
                "mapped field = {0} should not *also* be present in the list "\
                "of fields to directly transfer. Please remove this field "\
                "from either get_mapping_table() or get_extra_fields(). "\
                .format(lower_case_field)

            if lower_case_field not in seen_fields:
                all_fields.append((mf, d['type']))
                seen_fields.add(lower_case_field)
                metadata[lower_case_field] = d
                # print "mf = {0} d = {1}".format(mf, d)
            else:
                # print "mf = {0} already present. Now printing d".format(mf)
                old_d = metadata[lower_case_field]
                # print "Calling combine keys for mod = {0} field = {1} "\
                #     .format(mod, mf)

                # print "old dict = {0}\nnew_dict = {1}".format(old_d, d)

                updated_d = self.combine_and_append_keys(old_d, d)
                metadata[lower_case_field] = updated_d

        # Now just update the dictionaries for the items in the mapping table
        for key, val in self.mapping.table.iteritems():
            lower_case_field = key.lower()
            old_d = metadata[lower_case_field]
            try:
                new_d = self.src_fields_dict[val]
            except KeyError:
                try:
                    new_d = self.src_fields_dict[val.lower()]
                except:
                    Tracer()()

            # print "Calling combine keys for mapping table field = {0} val = {1}"\
            #     .format(key, val)

            # print "old dict = {0}\nnew_dict = {1}".format(old_d, new_d)
            updated_d = self.combine_and_append_keys(old_d, new_d)
            metadata[lower_case_field] = updated_d

        # print "all_fields = {0}".format(all_fields)
        try:
            self.galaxy_type = np.dtype(all_fields)
        except:
            Tracer()()

        self.metadata = metadata

    def convert(self):
        sim = self.get_simulation_data()
        redshifts = self.get_snapshot_redshifts()

        if sim is None:
            msg = '\n'.join([
                    'No simulation information has been specified in your conversion script. ',
                    'Please add a dictionary of simulation information composed of the Hubble ',
                    'constant, OmegaM, OmegaL, and the box size.'
                    ])
            print(msg)
            sys.exit(1)
            
        if 'hubble' not in sim:
            msg = '\n'.join([
                    'No Hubble value found in simulation data.'
                    ])
            print(msg)
            sys.exit(1)
        if 'omega_m' not in sim:
            msg = '\n'.join([
                    'No OmegaM value found in simulation data.'
                    ])
            print(msg)
            sys.exit(1)
            
        if 'omega_l' not in sim:
            msg = '\n'.join([
                    'No OmegaL value found in simulation data.'
                    ])
            print(msg)
            sys.exit(1)
        if 'box_size' not in sim:
            msg = '\n'.join([
                    'No box size value found in simulation data.'
                    ])
            print(msg)
            sys.exit(1)

        if redshifts is None:
            msg = '\n'.join([
                    'No snapshot redshift data found.',
                    ])
            print(msg)
            sys.exit(1)

        library['dataset-version'] = datetime.today().strftime("%Y-%m-%d")

        # argparse changes '-' to '_'. Hence, the field is
        # "dataset_version" rather than "dataset-version"
        if self.args.dataset_version:
            library['dataset-version'] = self.args.dataset_version
            
        # Cache some information for other parts of the system.
        library['box_size'] = sim['box_size']
        library['redshifts'] = redshifts
        library['n_snapshots'] = len(redshifts)
        library['metadata']  = self.metadata
        library['hubble'] = sim['hubble']

        if not self.args.sim_name:
            msg = 'Must specify "sim-name": name for the Simulation '\
                '(dark matter/hydro)'
            raise ConversionError(msg)
        library['sim-name']  = self.args.sim_name

        if not self.args.model_name:
            msg = 'Must specify a "model-name": name for the SAM galaxy '\
                'formation model (use simulation name for hydro)'
            raise ConversionError(msg)

        library['model-name'] = self.args.model_name

        # Write the xml file (only on rank=0 for MPI jobs)
        outfilename = self.args.output + '-settings.xml'
        rank = None
        comm = None
        if self.MPI is not None:
            comm = self.MPI.COMM_WORLD
            rank = comm.rank
            
        root_process = (not self.MPI) or (rank == 0)
        if root_process:
            with open(outfilename, 'w') as f:
                f.write(get_settings_xml(self.galaxy_type, redshifts,
                                         self.metadata))
            
        outfilename = self.generate_hdf5_filename(rank)
        with Exporter(outfilename, self) as exp:
            exp.set_cosmology(sim['hubble'], sim['omega_m'], sim['omega_l'])
            exp.set_box_size(sim['box_size'])
            exp.set_redshifts(redshifts)
            for tree in self.iterate_trees():
                exp.add_tree(tree)

        # If this is an MPI job, then the globalindex/globaldescendants have
        # to be fixed (those indices *must* be unique across all files)
        if comm is not None and comm.size > 1:
            comm.Barrier()
            self.finalize_hdf5_parallel_mode(comm)

        # Now write the common headers for both serial and parallel-mode.
        # (write only on root -- rank = 0 for MPI or the only task in serial)
        if root_process:
            self.write_common_hdf5_headers(comm, verbose=False)
            
    def write_common_hdf5_headers(self, comm=None, verbose=False):
        
        import h5py

        ncores = 1
        if comm is not None:
            ncores = comm.size
    
        # Not specifying the rank -> generating the master filename
        outfilename = self.generate_hdf5_filename() 
        mode = 'r+' if comm is None else 'w'

        with h5py.File(outfilename, mode) as hf:
            hf.attrs['nfiles'] = ncores
            hf.attrs['runtype'] = 'serial' if comm is None else 'parallel'
            alloutputfiles = []
            if comm is None:
                # in serial mode -> current file is the
                # only output file
                alloutputfiles.append(outfilename)
            else:
                # in MPI mode, each MPI task has written out its own
                # converted file.
                for rank in xrange(ncores):
                    alloutputfiles.append(self.generate_hdf5_filename(rank))
                    
            hf.attrs['filenames'] = alloutputfiles

        outfilename = self.generate_hdf5_filename()
        with h5py.File(outfilename, 'r+') as hf:
            cmdline_args = OrderedDict(vars(self.args))
            for k,v in cmdline_args.items():
                if v is not None:
                    hf.attrs[str(k)] = v

        return
    


    def generate_hdf5_filename(self, rank=None):
        import os

        # Store the absolute path
        outfilename = os.path.abspath(self.args.output)
        if rank is not None:
            if outfilename[-1] != '_':
                outfilename += '_'

            outfilename += '{0:d}'.format(rank)

        outfilename += '.h5'
        return outfilename



    def finalize_hdf5_parallel_mode(self, comm=None, verbose=False):

        if comm is None or comm.size == 1:
            return

        def _gather_and_cumul_sum_parallel(inputs, comm,
                                           verbose=False,
                                           field_desc=''):
            
            assert comm is not None
            rank = comm.rank
            ncores = comm.size
            comm.Barrier()

            if verbose:
                for r in xrange(ncores):
                    if r == rank:
                        print("On rank = {0} {1} = {2}"\
                                  .format(r, field_desc, inputs))
                    comm.Barrier()

            # Checks:

            # Check #1: The input is of integer type
            msg = "On rank = {0}, expected integer input but found = {1} "\
                "instead of type = {2}"\
                .format(rank, inputs, type(inputs))
            try:
                inputs.dtype
                assert np.issubdtype(inputs.dtype, np.integer), msg
            except AttributeError:
                assert isinstance( inputs, ( int, long ) ), msg

            recvbuf = np.zeros(ncores, dtype=np.int64)
            comm.Allgather([inputs, MPI.INT64_T],
                           [recvbuf, MPI.INT64_T])
            recvbuf = recvbuf.cumsum()
            if rank == 0:
                offset = 0
            else:
                offset = recvbuf[rank-1]

            if verbose:
                for r in xrange(ncores):
                    if r == rank:
                        print("On rank = {0} {1} offset = {2}"\
                                  .format(r, field_desc, offset))
                    comm.Barrier()
                    
            return offset
            
        
        import h5py

        # The list of fields that need to be fixed
        tree_fields_to_fix = [('treeindex', 'tree'),
                              ('tree_displs', 'galaxy')]
        
        # Was an MPI task -> need to compute the unique
        # globalindex across *all* files
        MPI = self.MPI
        rank = comm.rank
        ncores = comm.size
        
        outfilename = self.generate_hdf5_filename(rank)
        with h5py.File(outfilename, 'r') as hf:
            gal = hf['galaxies']

            # number of galaxies per core is to fix `global*` fields
            ngalaxies_this_core = np.array(gal.shape[0], dtype=np.int64)

            # number of trees per core is to fix `treeindex` field
            tree = hf['tree_counts']
            ntrees_this_core = np.array(tree.shape[0], dtype=np.int64)


        # Find the number of galaxies written up to the previous core
        galaxy_offset = _gather_and_cumul_sum_parallel(ngalaxies_this_core,
                                                       comm,
                                                       verbose=verbose,
                                                       field_desc='ngalaxies')

        # Now find the number of trees written up to the previous core
        tree_offset = _gather_and_cumul_sum_parallel(ntrees_this_core,
                                                     comm,
                                                     verbose=verbose,
                                                     field_desc='ntrees')

        # what are the fields that are 'global' ?
        # Essentially, these fields should be unique across
        # all files. Currently, these fields are only unique
        # across the local file (corresponding to the rank)
        # By adding the galaxy_offset (== number of galaxies written upto
        # this local file), we can make all of the 'global' fields
        # unique across all files.
        globalfields = []
        for mod in self.modules:
            for generator in mod.generators:
                if hasattr(generator, 'fields'):

                    # field is a tuple with name and dtype
                    for f, _  in generator.fields:

                        # Probably should also check that the field is
                        # of integer type
                        if 'global' in f.lower() and f not in globalfields:
                            globalfields.append(f)

        # Check that globalfields is unique, otherwise the galaxy_offset
        # will be applied twice.
        unique_len = len(set(globalfields))
        if unique_len != len(globalfields):
            msg = 'Error: Found duplicate fields while applying offsets '\
                'in globalfields. Unique items = {0} globalfields = {1}.'\
                .format(set(globalfields), globalfields)
            msg += '\nEasiest fix is to replace globalfields list in '\
                '`TAOImport/tao/Converter.py` with `list(set(globalfields))`'
            
            raise AssertionError(msg)

        # No offsets to fix on the root process, hence return immediately
        # The advantage is less tabbing in the following sub-section :)
        # - MS: 27/02/2017
        if rank == 0:
            return
        
        # Now fix the global* indices on cores that do need to
        # apply an offset
        with h5py.File(outfilename, 'a') as hf:
            galaxies = hf['galaxies']
            for f in globalfields:
                try:
                    val = galaxies[f][:]


                    # -- *Checks* -- 

                    # Check # 1:
                    # Check that we are not accidentally offsetting a different
                    # field that does not correspond to the generator. Unlikely
                    # to occur but always a good idea. 
                    msg = 'For field = {0}, maximum = {1} must at most '\
                        'equal to the number of galaxies on this core = '\
                        '{2} (rank = {3})'\
                        .format(f, val.max(), ngalaxies_this_core, rank)
                    assert val.max() < ngalaxies_this_core, msg

                    # Check # 2:
                    # Another check would be
                    # to make sure that the val.shape > ntrees_this_core
                    msg = 'For field = {0}, the length of the array = {1} '\
                        'should be (much) greater than the number of trees '\
                        ' = {2} on this core (rank = {3}) '\
                        '(since each tree should contain a few galaxies)' \
                        .format(f, val.shape[0], ntrees_this_core, rank)
                    assert val.shape[0] > ntrees_this_core, msg
                    
                    val[val >= 0] += galaxy_offset
                    galaxies[f] = val

                except:
                    raise

            # Now let's fix the fields that need to be offset by the number
            # of trees written out by *all* previous cores. The field
            # itself could be either directly specified in the file or 
            # within the galaxy dtype
            for f, typ in tree_fields_to_fix:

                if 'tree' in typ.lower():
                    offset = tree_offset
                elif 'galaxy' in typ.lower():
                    offset = galaxy_offset
                else:
                    msg = "Offset can only be of 'tree' or 'galaxy' type "\
                        "but for field = {0}, the offset type is specified as"\
                        " {1}".format(f, typ)
                    raise ValueError(msg)
                
                
                try:
                    val = galaxies[f][:]
                    val += offset
                    galaxies[f] = val

                # MS: I find this an annoying "feature" of numpy
                # If the field does not exist, the exception should
                # be a KeyError and *NOT* a ValueError. I also think the
                # numpy developers are also aware of this since I seem
                # to recall seeing an issue saying that the behaviour
                # will be changed in the future. Hence, catching both
                # the errors here. - MS 27th Feb, 2017
                except (ValueError, KeyError):
                    try:
                        # These are `tree_displs` fields which are
                        # embedded into the hdf5 file directly (rather
                        # than a galaxy property).
                        # Clarifying note: `tree_counts` is a per-tree
                        # quantity *also* embedded but being a per-tree
                        # quantity, `tree_counts` does not need any special
                        # fixes for the parallel case. -MS, 21st Mar, 2017
                        val = hf[f][:]
                        val += offset
                        hf[f][:] = val[:]
                        # Why not this line? - MS 14th Mar, 2017
                        # hf[f] = val
                    except (ValueError, KeyError):

                        # The treeindex field only exists in SAGE-like datasets
                        # Technically, I could check the converter cls and
                        # parse for 'SAGE' but that would require assuming a
                        # naming convention. Compromise - warn if 'SAGE'
                        # is not present in the converter name
                        if not 'SAGE' in (converter.__name__).upper():
                            print("Warning: Could not find field = `{0}' in "
                                  "the hdf5 file. (Normal behaviour for "
                                  "`treeindex' field while converting "
                                  "non-SAGE datasets)".format(f))
                        pass
        return
        
    def convert_tree(self, src_tree):
        tstart = time.time()
        dst_tree = np.empty_like(src_tree,
                                 dtype=self.galaxy_type)
        t0 = time.time()
        # Do conversion first.
        fields = {}
        for mod in self.modules:
            mod.convert_tree(src_tree, fields)

        mod_time = time.time() - t0

        t0 = time.time()
        # Next check for any issues.
        for mod in self.modules:
            mod.validate_fields(fields)

        val_time = time.time() - t0

        t0 = time.time()

        # Now perform generation.
        for mod in self.modules:
            mod.generate_fields(fields)

        gen_time = time.time() - t0
        t0 = time.time()
        # Now we can merge the fields into the tree.
        self._merge_fields(fields, dst_tree)
        # iter_time=time.time() - t0

        # t0 = time.time()
        # Perform a direct transfer of fields from within the
        # mapping that are flagged.
        self._transfer_fields(src_tree, dst_tree)

        copy_time = time.time() - t0

        t0 = time.time()
        # Now run any post conversion routines.
        for mod in self.modules:
            mod.post_conversion(dst_tree)

        post_conv_time = time.time() - t0
        total_time = time.time() - tstart

        # print "{:10.6f} {:10.6f}({:4.1f}%)  {:10.6f}({:4.1f}%) {:10.6f}({:4.1f}%) {:10.6f}({:4.1f}%) {:10.6f}({:4.1f}%) {:10.6f}({:4.1f}%)".format(total_time,
        #                                                                                                                               mod_time,mod_time/total_time*100.0,
        #                                                                                                                               val_time,val_time/total_time*100.0,
        #                                                                                                                               gen_time,gen_time/total_time*100.0,
        #                                                                                                                               iter_time,iter_time/total_time*100.0,
        #                                                                                                                               copy_time,copy_time/total_time*100.0,
        #                                                                                                                               post_conv_time,post_conv_time/total_time*100.0
        #                                                                                                                               )

        return dst_tree

    def _merge_fields(self, fields, dst_tree):
        for name, values in fields.iteritems():
            x = dst_tree[name]
            x[:] = values

    def _transfer_fields(self, src_tree, dst_tree):
        for field, _ in self.mapping.fields.iteritems():
            dst_tree[field] = src_tree[field]
