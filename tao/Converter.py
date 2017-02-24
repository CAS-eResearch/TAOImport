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
        if self.MPI is not None:
            self.finalize_hdf5()
                
    def generate_hdf5_filename(self, rank=None):
        outfilename = self.args.output
        if rank is not None:
            if outfilename[-1] != '_':
                outfilename += '_'

            outfilename += '{0:d}'.format(rank)

        outfilename += '.h5'
        return outfilename

    def finalize_hdf5(self):

        if self.MPI is None:
            return

        # Was an MPI task -> need to compute the unique
        # globalindex across *all* files

        MPI = self.MPI
        comm = MPI.COMM_WORLD
        comm.Barrier()
        rank = comm.rank
        ncores = comm.size
        
        import h5py
        outfilename = self.generate_hdf5_filename(rank)
        with h5py.File(outfilename, 'r') as hf:
            gal = hf['galaxies']
            ngalaxies_this_core = np.array(gal.shape[0], dtype=np.int64)

        comm.Barrier()
        offset = np.int(0)
        recvbuf = np.zeros(ncores, dtype=np.int64)
        comm.Allgather([ngalaxies_this_core, MPI.INT64_T],
                       [recvbuf, MPI.INT64_T])
        recvbuf = recvbuf.cumsum()

        if rank == 0:
            offset = 0
        else:
            offset = recvbuf[rank-1]
        

        # print("offset = {0} on rank = {1} file = {2}"
        #       .format(offset, rank, outfilename))

        # what are the fields that are 'global' ?
        # Essentially, these fields should be unique across
        # all files. Currently, these fields are only unique
        # across the local file (corresponding to the rank)
        # By adding the offset (== number of galaxies written upto
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

        # Check that globalfields is unique, otherwise the offset
        # will be applied twice.
        unique_len = len(set(globalfields))
        if unique_len != len(globalfields):
            msg = 'Error: Found duplicate fields while applying offsets '\
                'in globalfields. Unique items = {0} globalfields = {1}.'\
                .format(set(globalfields), globalfields)
            msg += '\nEasiest fix is to replace globalfields list in '\
                '`TAOImport/tao/Converter.py` with `list(set(globalfields))`'
            
            raise AssertionError(msg)

        # Now fix the global* indices 
        if rank > 0:
            with h5py.File(outfilename, 'a') as hf:
                galaxies = hf['galaxies']
                for f in globalfields:
                    try:
                        # print("Before offsetting `field' {0} "
                        #       " on rank = {1}: val = {2}"
                        #       .format(f, rank, galaxies[f][0]))

                        val = galaxies[f][:]
                        val[val >= 0] += offset
                        galaxies[f] = val

                        # print("After offsetting `field' {0} "
                        #       " on rank = {1} : val = {2} (offset = {3})"
                        #       .format(f, rank, galaxies[f][0], offset))


                    except:
                        raise
            
            # Only the non-root processes have to fix the offsets
            # and once the MPI processes have fixed those indices,
            # their job is done
            if rank > 0:
                return


        # Now let's create a hdf5 master file that will hold sym-links to the
        # individual hdf5 files created by the MPI processes. Only done on the
        # root processor.
        assert rank == 0
        
        # Not specifying the rank -> generating the master filename
        outfilename = self.generate_hdf5_filename() 

        with h5py.File(outfilename, 'w') as hf:
            hf.attrs['nfiles'] = ncores
            alloutputfiles = []
            for rank in xrange(ncores):
                alloutputfiles.append(self.generate_hdf5_filename(rank))

            hf.attrs['filenames'] = alloutputfiles
        
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
