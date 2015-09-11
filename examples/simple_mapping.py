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

# A "tao.Mapping" instance initialised with a dictionary
# to use to map TAO required fields to the field in the
# source dataset. The second argument is a list of tuples,
# containing a field name and a field type. These fields
# will be passed through from the source data.
mapping = tao.Mapping({
    'position_x': 'x',
    'position_y': 'y',
    'position_z': 'z',
    'velocity_x': 'vx',
    'velocity_y': 'vy',
    'velocity_z': 'vz',
    'descendant': 'desc',
    'cold_gas': 'cg',
    'metals_cold_gas': 'cgz',
    'disk_scale_radius': 'dsr',
    'merge_type': 'mt',
}, [
    ('my_field', np.float32),
])

# A generator function which should yield each tree in the
# source dataset represented as a compound NumPy array.
def iterate_trees():

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

    # Of course you'll need to replace this with something
    # that loads the source trees one at a time.
    tree_size = 100
    tree = np.empty(tree_size, dtype=src_type)
    yield tree
    tree_size = 50
    tree = np.empty(tree_size, dtype=src_type)
    yield tree
