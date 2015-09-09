import h5py
import numpy as np

datatype = np.dtype([
    ('snapshot', np.uint32),
    ('type',     np.uint32),
    ('global_index', np.uint64),
    ('local_index', np.uint32),
    ('tree_index', np.uint32),
    ('descendant', np.int32),
    ('global_descendant', np.uint64),
    ('position_x', 'f'), ('position_y', 'f'), ('position_z', 'f'),
    ('velocity_x', 'f'), ('velocity_y', 'f'), ('velocity_z', 'f'),
])
