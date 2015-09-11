from Module import Module
from .validators import *
from .generators import *
from .library import library

class LightCone(Module):
    fields = {
        'position_x': {
            'description': 'The x position of the galaxy',
            'units': 'Mpc/h',
        },
        'position_y': {
            'description': 'The y position of the galaxy',
            'units': 'Mpc/h',
        },
        'position_z': {
            'description': 'The z position of the galaxy',
            'units': 'Mpc/h',
        },
        'velocity_x': {
            'description': 'The x velocity of the galaxy',
            'units': 'km/s',
        },
        'velocity_y': {
            'description': 'The y velocity of the galaxy',
            'units': 'km/s',
        },
        'velocity_z': {
            'description': 'The z velocity of the galaxy',
            'units': 'km/s',
        },
        'snapshot': {
            'description': 'The simulation snapshot number',
        }
    }
    generators = [
        GlobalIndices(),
    ]
    validators = [
        Required('position_x', 'position_y', 'position_z',
                 'velocity_x', 'velocity_y', 'velocity_z',
                 'snapshot'),
        OverLittleH('position_x', 'position_y', 'position_z'),
        WithinRange(0.0, library['box_size'], 'position_x', 'position_y', 'position_z'),
    ]
