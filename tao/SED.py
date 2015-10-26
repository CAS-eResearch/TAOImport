from collections import OrderedDict
import numpy as np
from Module import Module
from LightCone import LightCone
from .validators import *
from .generators import *

class SED(Module):
    dependencies = [LightCone]
    fields = OrderedDict([
        ('descendant', {
            'description': 'Tree-local index of the descendant galaxy',
            'type': np.int32,
        }),
        ('mergetype', {
            'description': 'Flag indicating type of merger',
            'choices': {
                0: 'None',
                1: 'Major merger',
                2: 'Minor merger',
                3: 'Disruption',
                4: 'ICS',
            },
            'type': np.int32,
        }),
        ('dt', {
            'description': 'Time-step between this galaxy and parent',
            'units': 'Myears/h',
            'type': np.float32,
        }),
        ('sfrdisk', {
            'description': 'Disk star formation rate',
            'units': 'Msun/year',
            'type': np.float32,
        }),
        ('sfrbulge', {
            'description': 'Bulge star formation rate',
            'units': 'Msun/year',
            'type': np.float32,
        }),
        ('sfrdiskz', {
            'description': 'Disk metallicity star formation rate',
            'units': 'Msun/year',
            'type': np.float32,
        }),
        ('sfrbulgez', {
            'description': 'Bulge metallicity star formation rate',
            'units': 'Msun/year',
            'type': np.float32,
        }),
    ])
    generators = [
        TreeIndices(),
        TreeLocalIndices(),
        GlobalDescendants(),
        DepthFirstOrdering(),
    ]
    validators = [
        Required('descendant', 'mergetype', 'dt',
                 'sfrdisk', 'sfrbulge',
                 'sfrdiskz', 'sfrbulgez'),
        TreeLocalIndex('descendant'),
        Choice([0, 1, 2, 3, 4], 'mergetype'),
        NonZero('dt'),
        Positive('sfrdisk', 'sfrbulge', 'sfrdiskz', 'sfrbulgez', 'dt'),
    ]

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--disable-sed', action='store_true', help='disable SED module')

    def parse_arguments(self, args):
        self.disabled = args.disable_sed
