from collections import OrderedDict
from Module import Module
from .validators import *
from .generators import *
from .library import library

class LightCone(Module):
    fields = OrderedDict([
        ('objecttype', {
            'description': 'Type of galaxy',
            'choices': [0, 1, 2],
            'type': np.int16,
            'group': "Galaxy Properties",
        }),
        ('posx', {
            'description': 'The x position of the galaxy',
            'units': 'Mpc/h',
            'type': np.float32,
            'group': "Positions & Velocities",
        }),
        ('posy', {
            'description': 'The y position of the galaxy',
            'units': 'Mpc/h',
            'type': np.float32,
            'group': "Positions & Velocities",
        }),
        ('posz', {
            'description': 'The z position of the galaxy',
            'units': 'Mpc/h',
            'type': np.float32,
            'group': "Positions & Velocities"
        }),
        ('velx', {
            'description': 'The x velocity of the galaxy',
            'units': 'km/s',
            'type': np.float32,
            'group': "Positions & Velocities"
        }),
        ('vely', {
            'description': 'The y velocity of the galaxy',
            'units': 'km/s',
            'type': np.float32,
            'group': "Positions & Velocities",
        }),
        ('velz', {
            'description': 'The z velocity of the galaxy',
            'units': 'km/s',
            'type': np.float32,
            'group': "Positions & Velocities",
        }),
        ('snapnum', {
            'description': 'The simulation snapshot number',
            'type': np.int32,
            'group': "Simulation",
        }),
    ])
    generators = [
        GlobalIndices(),
    ]
    validators = [
        Required('posx', 'posy', 'posz',
                 'velx', 'vely', 'velz',
                 'snapnum'),
        OverLittleH('posx', 'posy', 'posz'),

        # any reasonable pos/vel probably should have a width of 1e-2.
        # I am sure, in the future, there will be examples to the contrary.
        # MS - 31/10/2016 11 am
        # Found it. MS - 31/10/2016 4 pm. Switched from 1e-2 to 1e-3.
        # Found it. MS - 1/11/2016 4 pm. Switched from 1e-3 to 1e-4
        # Found it. MS - 2/11/2016 6 am. Switched from 1e-4 to 1e-6. min=6
        # Found it. MS - 13/07/2018 8 pm. Switched from 1e-6 to 1e-4, min=10
        ## But that error with Millennium was strange!
        # Working on ntrees = 32054 in group = 471
        # Traceback (most recent call last):
        #   File "/home/msinha/.local/bin/taoconvert", line 4, in <module>
        #     __import__('pkg_resources').run_script('taoconvert==0.1', 'taoconvert')
        #   File "build/bdist.linux-x86_64/egg/pkg_resources/__init__.py", line 750, in run_script
        #   File "build/bdist.linux-x86_64/egg/pkg_resources/__init__.py", line 1527, in run_script
        #   File "/home/msinha/.local/lib/python2.7/site-packages/taoconvert-0.1-py2.7.egg/EGG-INFO/scripts/taoconvert", line 87, in <module>
        #     converter.convert()
        #   File "/home/msinha/.local/lib/python2.7/site-packages/taoconvert-0.1-py2.7.egg/tao/Converter.py", line 208, in convert
        #     exp.add_tree(tree)
        #   File "/home/msinha/.local/lib/python2.7/site-packages/taoconvert-0.1-py2.7.egg/tao/Exporter.py", line 62, in add_tree
        #     dst_tree = self.converter.convert_tree(tree)
        #   File "/home/msinha/.local/lib/python2.7/site-packages/taoconvert-0.1-py2.7.egg/tao/Converter.py", line 225, in convert_tree
        #     mod.validate_fields(fields)
        #   File "/home/msinha/.local/lib/python2.7/site-packages/taoconvert-0.1-py2.7.egg/tao/Module.py", line 71, in validate_fields
        #     validator.validate_fields(fields)
        #   File "/home/msinha/.local/lib/python2.7/site-packages/taoconvert-0.1-py2.7.egg/tao/validators.py", line 101, in validate_fields
        #     raise ValidationError(msg)
        # tao.validators.ValidationError: At least 6 values are within min. width = 1e-06 for field "posx".. Found values of min,max=[220.76648,220.76648]. Size = 6
        # Exception in thread Thread-472 (most likely raised during interpreter shutdown):
        # Traceback (most recent call last):

        NonZeroDistribution(1e-4, 10, 'posx', 'posy', 'posz',
                            'velx', 'vely', 'velz'),
        WithinRange(0.0, library['box_size'], 'posx', 'posy', 'posz'),
        WithinCRange(0, library['n_snapshots'], 'snapnum')
    ]
