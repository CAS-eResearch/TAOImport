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
        ('merge_type', {
            'description': 'Flag indicating type of merger',
            'choices': {
                0: 'None',
                1: 'Major merger',
                2: 'Minor merger',
                3: 'Disruption',
                4: 'ICS',
            },
            'type': np.uint32,
        }),
        ('dt', {
            'description': 'Time-step between this galaxy and parent',
            'units': 'Myears/h',
            'type': np.float32,
        }),
        ('sfr_disk', {
            'description': 'Disk star formation rate',
            'units': 'Msun/year',
            'type': np.float32,
        }),
        ('sfr_bulge', {
            'description': 'Bulge star formation rate',
            'units': 'Msun/year',
            'type': np.float32,
        }),
        ('sfr_disk_z', {
            'description': 'Disk metallicity star formation rate',
            'units': 'Msun/year',
            'type': np.float32,
        }),
        ('sfr_bulge_z', {
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
        Required('descendant', 'merge_type', 'dt',
                 'sfr_disk', 'sfr_bulge',
                 'sfr_disk_z', 'sfr_bulge_z'),
        TreeLocalIndex('descendant'),
        Choice([0, 1, 2, 3, 4], 'merge_type'),
        NonZero('dt'),
        Positive('sfr_disk', 'sfr_bulge', 'sfr_disk_z', 'sfr_bulge_z', 'dt'),
    ]

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--disable-sed', action='store_true', help='disable SED module')

    def parse_arguments(self, args):
        self.disabled = args.disable_sed
