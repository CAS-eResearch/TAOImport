import numpy as np
import time
from .library import library
from .Exporter import Exporter
from .Mapping import Mapping
from .xml import get_settings_xml
from IPython.core.debugger import Tracer
from collections import OrderedDict
import os

from datetime import datetime

class ConversionError(Exception):
    pass

class Converter(object):

    def __init__(self, modules, args):
        self.modules = modules
        self.args = args
        dictname = 'src_fields_dict'
        if hasattr(self, dictname):
            initial_order = 0
            src_fields_dict = self.src_fields_dict
            
            # Check if there are any arrays that are present
            for f, d in src_fields_dict.items():
                try:
                    shape = d['shape']

                    # This field (referenced by field name "f")
                    # is an array type. Needs to be replaced by
                    # "f_{dim}" for each of the dimensions in "shape".
                    # (For instance, pos[3] will get changed to pos_0,
                    # pos_1, pos_2)

                    # The print function is fake - it is the regular python2
                    # print 
                    print("Fixing array field {0} with shape={2}. Old "
                          "dictionary = {1}".format(f, d, shape))
                    src_fields_dict.pop(f)
                    d['shape']=1
                    prev_label = d['label']
                    for axis in range(shape):
                        new_field_name = '{0}_{1}'.format(f, axis)
                        try:
                            d['label'] = '{0} (along {1})'.format(prev_label,
                                                                  axis)
                        except KeyError:
                            d['label'] = new_field_name
                            
                        src_fields_dict[new_field_name] = d
                        print('New dictionary: {0}:{1} (along {2})'.
                              format(new_field_name, d, axis))
                    
                except KeyError:
                    # No shape - therefore a single ranked array
                    pass
            
        # print("self.src_fields_dict = {0}".format(self.src_fields_dict))
        
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
                except Exception as e:
                    print("error {0}".format(e))
                    Tracer()()

            print "Calling combine keys for mapping table field = {0} val = {1}"\
                .format(key, val)

            print "old dict = {0}\nnew_dict = {1}".format(old_d, new_d)
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
            print '\n'.join([
                'No simulation information has been specified in your conversion script. ',
                'Please add a dictionary of simulation information composed of the Hubble ',
                'constant, OmegaM, OmegaL, and the box size.'
            ])
            sys.exit(1)
        if 'hubble' not in sim:
            print '\n'.join([
                'No Hubble value found in simulation data.'
            ])
            sys.exit(1)
        if 'omega_m' not in sim:
            print '\n'.join([
                'No OmegaM value found in simulation data.'
            ])
            sys.exit(1)
        if 'omega_l' not in sim:
            print '\n'.join([
                'No OmegaL value found in simulation data.'
            ])
            sys.exit(1)
        if 'box_size' not in sim:
            print '\n'.join([
                'No box size value found in simulation data.'
            ])
            sys.exit(1)

        if redshifts is None:
            print '\n'.join([
                'No snapshot redshift data found.',
            ])
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

        outfilename = self.args.output + '-settings.xml'
        with open(outfilename, 'w') as f:
            f.write(get_settings_xml(self.galaxy_type, redshifts,
                                     self.metadata))
        with Exporter(self.args.output, self) as exp:
            exp.set_cosmology(sim['hubble'], sim['omega_m'], sim['omega_l'])
            exp.set_box_size(sim['box_size'])
            exp.set_redshifts(redshifts)
            for tree in self.iterate_trees():
                exp.add_tree(tree)

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
            raise ConversionError(msg)
        
        # Get the field
        for (fld, dst) in zip(src_tree[fieldname], dest_tree):
            for axis in range(shape[0]):
                dest_fieldname = '{0}_{1}'.format(fieldname, axis)
                try:
                    dst[dest_fieldname] = fld[axis]
                except:
                    print()
                    raise
        

    def _merge_fields(self, fields, dst_tree):
        for name, values in fields.iteritems():
            x = dst_tree[name]
            x[:] = values

    def _transfer_fields(self, src_tree, dst_tree):
        for field, _ in self.mapping.fields.iteritems():
            dst_tree[field] = src_tree[field]
