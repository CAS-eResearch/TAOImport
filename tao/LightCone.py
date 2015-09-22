from collections import OrderedDict
from Module import Module
from .validators import *
from .generators import *
from .library import library

class LightCone(Module):
    fields = OrderedDict([
        ('type', {
            'description': 'Type of galaxy',
            'choices': [0, 1, 2],
            'type': np.uint32,
        }),
        ('position_x', {
            'description': 'The x position of the galaxy',
            'units': 'Mpc/h',
            'type': np.float32,
        }),
        ('position_y', {
            'description': 'The y position of the galaxy',
            'units': 'Mpc/h',
            'type': np.float32,
        }),
        ('position_z', {
            'description': 'The z position of the galaxy',
            'units': 'Mpc/h',
            'type': np.float32,
        }),
        ('velocity_x', {
            'description': 'The x velocity of the galaxy',
            'units': 'km/s',
            'type': np.float32,
        }),
        ('velocity_y', {
            'description': 'The y velocity of the galaxy',
            'units': 'km/s',
            'type': np.float32,
        }),
        ('velocity_z', {
            'description': 'The z velocity of the galaxy',
            'units': 'km/s',
            'type': np.float32,
        }),
        ('snapshot', {
            'description': 'The simulation snapshot number',
            'type': np.uint32,
        }),
    ])
    generators = [
        GlobalIndices(),
    ]
    validators = [
        Required('position_x', 'position_y', 'position_z',
                 'velocity_x', 'velocity_y', 'velocity_z',
                 'snapshot'),
        OverLittleH('position_x', 'position_y', 'position_z'),
        WithinRange(0.0, library['box_size'], 'position_x', 'position_y', 'position_z'),
        WithinCRange(0, library['n_snapshots'], 'snapshot')
    ]
