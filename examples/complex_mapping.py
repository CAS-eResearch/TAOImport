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

# In this case we inherit the Mapping class to do some
# more complicated mappings.
class MyMapping(tao.Mapping):

    # Any method of the form "map_<field>" where <field>
    # is the name of a TAO field will be called to produce
    # the mapped values. The 'tree' argument is the current
    # source tree. Use it and any of its contents to
    # generate a new array with the necessary values.
    def map_position_x(self, tree):
        x = tree['x']
        return 2*x

# A "tao.Mapping" instance initialised with a dictionary
# to use to map TAO required fields to the field in the
# source dataset.
mapping = tao.Mapping({
    'position_x': 'x',
    'position_y': 'y',
    'position_z': 'z',
    'velocity_x': 'vx',
    'velocity_y': 'vy',
    'velocity_z': 'vz',
    'descendant': 'desc',
})

# A generator function which should yield each tree in the
# source dataset represented as a compound NumPy array.
def iterate_trees():

    # You'll probably want to define the datatype of the source
    # trees somewhere here.
    src_type = np.dtype([
        ('x', 'f'), ('y', 'f'), ('z', 'f'),
        ('vx', 'f'), ('vy', 'f'), ('vz', 'f'),
        ('desc', 'i'), ('snapshot', 'i'),
    ])

    # Of course you'll need to replace this with something
    # that loads the source trees one at a time.
    tree_size = 100
    trees = np.empty(tree_size, dtype=src_type)
    yield trees
