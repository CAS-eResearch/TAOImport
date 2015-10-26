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
            'type': np.int32,
        }),
        ('posx', {
            'description': 'The x position of the galaxy',
            'units': 'Mpc/h',
            'type': np.float32,
        }),
        ('posy', {
            'description': 'The y position of the galaxy',
            'units': 'Mpc/h',
            'type': np.float32,
        }),
        ('posz', {
            'description': 'The z position of the galaxy',
            'units': 'Mpc/h',
            'type': np.float32,
        }),
        ('velx', {
            'description': 'The x velocity of the galaxy',
            'units': 'km/s',
            'type': np.float32,
        }),
        ('vely', {
            'description': 'The y velocity of the galaxy',
            'units': 'km/s',
            'type': np.float32,
        }),
        ('velz', {
            'description': 'The z velocity of the galaxy',
            'units': 'km/s',
            'type': np.float32,
        }),
        ('snapnum', {
            'description': 'The simulation snapshot number',
            'type': np.int32,
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
        WithinRange(0.0, library['box_size'], 'posx', 'posy', 'posz'),
        WithinCRange(0, library['n_snapshots'], 'snapnum')
    ]
