#!/usr/bin/env python

import argparse, os, sys, pprint
import numpy as np, tao
from tao.find_modules import find_modules

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert semi-analytic data into TAO format')
    parser.add_argument('-s', '--script', default='taoconv.py', help='script defining conversion (default: taoconv.py)')
    parser.add_argument('-o', '--output', default='output', help='output name')
    parser.add_argument('-i', '--info', help='show information about a field')
    modules = find_modules()
    for mod in modules:
        mod.add_arguments(parser)
    args = parser.parse_args()

    if args.info:
        for mod in modules:
            if args.info in mod.fields:
                pprint.pprint({args.info: mod.fields[args.info]})
                sys.exit(0)
        print 'Unknown field: ' + args.info
        sys.exit(1)

    launch_dir = os.getcwd()
    script = os.path.join(launch_dir, args.script)
    if not os.path.exists(script):
        print 'Unable to find script: "%s"'%args.script
        sys.exit(1)

    locals_dict = {}
    globals_dict = {'np': np, 'tao': tao}
    execfile(script, globals_dict, locals_dict)

    sim = locals_dict.get('simulation', None)
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

    redshifts = locals_dict.get('snapshot_redshifts', None)
    if redshifts is None:
        print '\n'.join([
            'No snapshot redshift data found.',
        ])
        sys.exit(1)

    tao.library['box_size'] = sim['box_size']

    with tao.Exporter(args.output, modules, locals_dict.get('mapping', None), arguments=args) as exp:
        exp.set_cosmology(sim['hubble'], sim['omega_m'], sim['omega_l'])
        exp.set_box_size(sim['box_size'])
        exp.set_redshifts(redshifts)
        for tree in locals_dict['iterate_trees']():
            exp.add_tree(tree)
