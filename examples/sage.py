# Add any arguments you may want.
parser.add_argument('--trees-dir', default='.', help='location of SAGE trees')

# Simulation data. This includes:
#  1. "box_size", the simulation domain in "Mpc/h",
#  2. "hubble", the Hubble constant used,
#  3. "omega_m", ?
#  4. "omega_l", ?
simulation = {
    'box_size': 62.5,
    'hubble': 0.71,
    'omega_m': 0.25,
    'omega_l': 0.75,
}

# Any iterable containing the redshifts of each snapshot
# in increasing order of snapshot number.
snapshot_redshifts = [127, 94, 32, 0]

# Need to do some work to get the descendant value.
class SAGEMapping(tao.Mapping):
    def map_descendant(self, tree):
        pass # TODO

mapping = tao.Mapping({
    'position_x': 'x',
    'position_y': 'y',
    'position_z': 'z',
    'velocity_x': 'vx',
    'velocity_y': 'vy',
    'velocity_z': 'vz',
    'snapshot': 'snapnum',
    'merge_type': 'merge',
    # TODO
}, [

    # Need to add in all the SAGE fields that are not a
    # part of the standard set.
    ('spin_x', np.float32),
    ('spin_y', np.float32),
    ('spin_z', np.float32),
    # TODO
])

# A generator function which should yield each tree in the
# source dataset represented as a compound NumPy array.
def iterate_trees(args):

    # You'll probably want to define the datatype of the source
    # trees somewhere here.
    src_type = np.dtype([
        ('x', 'f'), ('y', 'f'), ('z', 'f'),
        ('vx', 'f'), ('vy', 'f'), ('vz', 'f'),
        ('desc', 'i'), ('snapshot', 'i'),
        ('cg', 'f'), ('cgz', 'f'), ('dsr', 'f'),
        ('mt', 'i'), ('dt', 'f'), ('sfr_disk', 'f'),
        ('sfr_bulge', 'f'), ('sfr_disk_z', 'f'),
        ('sfr_bulge_z', 'f'), ('my_field', 'f'),
    ])

    entries = [e for e in os.listdir(args.trees_dir) if os.path.isfile(e)]
    redshift_strings = get_redshift_strings(entries)

    # Of course you'll need to replace this with something
    # that loads the source trees one at a time.
    tree_size = 100
    tree = np.empty(tree_size, dtype=src_type)
    yield tree
    tree_size = 50
    tree = np.empty(tree_size, dtype=src_type)
    yield tree
