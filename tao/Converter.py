import numpy as np
import time
from .library import library
from .Exporter import Exporter
from .Mapping import Mapping
from .xml import get_settings_xml
from IPython.core.debugger import Tracer
from collections import OrderedDict
import os

class ConversionError(Exception):
    pass

class Converter(object):

    def __init__(self, modules, args):
        self.modules = modules
        self.args = args
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

        # Cache some information for other parts of the system.
        library['box_size'] = sim['box_size']
        library['redshifts'] = redshifts
        library['n_snapshots'] = len(redshifts)
        library['metadata']  = self.metadata
        library['hubble'] = sim['hubble']

        if not self.args.sim_name:
            raise ConversionError('Must specify "sim-name": name for the Simulation (dark matter/hydro)')
        library['sim-name']  = self.args.sim_name

        if not self.args.model_name:
            raise ConversionError('Must specify a "model-name": name for the SAM galaxy formation model (use simulation name for hydro)')

        library['model-name'] = self.args.model_name

        
        with open(os.path.dirname(self.args.output)+'/settings.xml', 'w') as f:
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
        dst_tree = np.empty_like(src_tree, dtype = self.galaxy_type)
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

        iter_time=time.time() - t0

        t0 = time.time() 
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
        # print "fields in _merge_fields = {0}".format(fields)
        for name, values in fields.iteritems():
            x = dst_tree[name]
            x[:] = values

    def _transfer_fields(self, src_tree, dst_tree):
        # print "fields in _transfer_fields = {0} ".format(self.mapping.fields)
        # print "src_tree.dtype = {0}".format(src_tree.dtype)
        # print "dst_tree.dtype = {0}".format(dst_tree.dtype)
        for field, _ in self.mapping.fields.iteritems():
            dst_tree[field] = src_tree[field]

