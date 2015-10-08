from collections import OrderedDict
import numpy as np
from Module import Module
from SED import SED
from .validators import *

class Dust(Module):
    dependencies = [SED]
    fields = OrderedDict([
        ('coldgas', {
            'description': 'Cold gas mass',
            'units': 'Msun/h',
            'type': np.float32,
        }),
        ('metalscoldgas', {
            'description': 'Cold gas metallicity mass',
            'units': 'Msun/h',
            'type': np.float32,
        }),
        ('diskscaleradius', {
            'description': 'Disk scale radius',
            'units': 'Mpc/h',
            'type': np.float32,
        }),
    ])
    validators = [
        Required('coldgas', 'metalscoldgas', 'diskscaleradius'),
        Positive('coldgas', 'metalscoldgas', 'diskscaleradius'),
    ]

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--disable-dust', action='store_true', help='disable dust module')

    def parse_arguments(self, args):
        self.disabled = args.disable_dust
