from Module import Module
from .validators import *
from .generators import *

class SED(Module):
    fields = {
        'descendant': {
            'description': 'Tree-local index of the descendant galaxy',
        },
        'merge_type': {
            'description': 'Flag indicating type of merger',
            'choices': {
                0: 'Major merger',
                1: 'Minor merger',
                2: 'Disruption',
            },
        },
        'dt': {
            'description': 'Time-step between this galaxy and parent',
            'units': 'Myears/h',
        },
        'sfr_disk': {
            'description': 'Disk star formation rate',
            'units': 'Msun/year',
        },
        'sfr_bulge': {
            'description': 'Bulge star formation rate',
            'units': 'Msun/year',
        },
        'sfr_disk_z': {
            'description': 'Disk metallicity star formation rate',
            'units': 'Msun/year',
        },
        'sfr_bulge_z': {
            'description': 'Bulge metallicity star formation rate',
            'units': 'Msun/year',
        },
    }
    generators = [
        TreeIndices(),
        TreeLocalIndices(),
        GlobalDescendants(),
    ]
    validators = [
        Required('descendant', 'merge_type', 'dt',
                 'sfr_disk', 'sfr_bulge',
                 'sfr_disk_z', 'sfr_bulge_z'),
    ]

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--disable-sed', action='store_true', help='disable SED module')

    def parse_arguments(self, args):
        self.disabled = args.disable_sed
