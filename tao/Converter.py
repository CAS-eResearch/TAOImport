import numpy as np
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

        with Exporter(self.args.output, self) as exp:
            exp.set_cosmology(sim['hubble'], sim['omega_m'], sim['omega_l'])
            exp.set_box_size(sim['box_size'])
            exp.set_redshifts(redshifts)
            for tree in self.iterate_trees():
                exp.add_tree(tree)

        with open('settings.xml', 'w') as file:
            file.write(get_settings_xml(self.galaxy_type))

    def convert_tree(self, src_tree):
        dst_tree = np.empty(len(src_tree), self.galaxy_type)

        # Do conversion first.
        fields = {}
        for mod in self.modules:
            mod.convert_tree(src_tree, fields)

        # Next check for any issues.
        for mod in self.modules:
            mod.validate_fields(fields)

        # Now perform generation.
        for mod in self.modules:
            mod.generate_fields(fields)

        # Now we can merge the fields into the tree.
        for name, values in fields.iteritems():
            x = dst_tree[name]
            x[:] = values

        # Perform a direct transfer of fields from within the
        # mapping that are flagged.
        for field, dtype in self.mapping.fields:
            dst_tree[field] = src_tree[field]

        # Now run any post conversion routines.
        for mod in self.modules:
            mod.post_conversion(dst_tree)

        return dst_tree

    # def depth_first_order(self, tree):
    #     dfi = [0]
    #     parents = {}
    #     order = np.empty(len(tree))

    #     def _recurse(idx):
    #         order[idx] = dfi[0]
    #         dfi[0] += 1
    #         tree['subsize'][idx] = 1
    #         if idx in parents:
    #             for par in parents[idx]:
    #                 tree['subsize'][idx] += _recurse(par)
    #         return tree['subsize'][idx]

    #     # Find the roots and parents.
    #     roots = []
    #     for ii in range(len(tree)):
    #         desc = tree['descendant'][ii]
    #         if desc == -1:
    #             roots.append(ii)
    #         else:
    #             parents.setdefault(desc, []).append(ii)

    #     # Recurse to find the new ordering.
    #     for ii in roots:
    #         _recurse(ii)

    #     # Remap everything.
    #     for ii in range(len(tree)):
    #         tree['local_index'][ii] = order[tree['local_index'][ii]]
    #         if tree['descendant'][ii] != -1:
    #             tree['descendant'][ii] = order[tree['descendant'][ii]]
    #             tree['global_descendant'][ii] = tree['global_index'][tree['descendant'][ii]]
