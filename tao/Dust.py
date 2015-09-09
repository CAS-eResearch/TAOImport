from Module import Module
from .validators import *

class Dust(Module):
    fields = {
        'cold_gas': {
            'description': 'Cold gas mass',
            'units': 'Msun/h',
        },
        'metals_cold_gas': {
            'description': 'Cold gas metallicity mass',
            'units': 'Msun/h',
        },
        'disk_scale_radius': {
            'description': 'Disk scale radius',
            'units': 'Mpc/h',
        },
    }
    validators = [
        Required('cold_gas', 'metals_cold_gas', 'disk_scale_radius'),
    ]

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--disable-dust', action='store_true', help='disable dust module')

    def parse_arguments(self, args):
        self.disabled = args.disable_dust
