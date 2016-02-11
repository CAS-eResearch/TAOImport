import numpy as np
import time
from .library import library
from .Exporter import Exporter
from .Mapping import Mapping
from .xml import get_settings_xml

class ConversionError(Exception):
    pass

class Converter(object):

    def __init__(self, modules, args):
        self.modules = modules
        self.args = args
        self.mapping = Mapping(self, self.get_mapping_table(), self.get_extra_fields())
        self.make_datatype()
        for mod in self.modules:
            mod.mapping = self.mapping

    def make_datatype(self):
        all_fields = []
        seen_fields = set()
        for mod in self.modules:
            new_fields = mod.get_numpy_fields()
            for nf in new_fields:
                if nf[0].lower() not in seen_fields:
                    all_fields.append(nf)
                    seen_fields.add(nf[0].lower())
        for mf in self.mapping.fields:
            if mf[0].lower() not in seen_fields:
                all_fields.append(mf)
                seen_fields.add(mf[0].lower())
        self.galaxy_type = np.dtype(all_fields)

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

        with open('settings.xml', 'w') as file:
            file.write(get_settings_xml(self.galaxy_type))
        
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
        for name, values in fields.iteritems():
            x = dst_tree[name]
            x[:] = values

    def _transfer_fields(self, src_tree, dst_tree):
        for field, dtype in self.mapping.fields:
            dst_tree[field] = src_tree[field]

