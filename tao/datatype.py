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
    ('merge_type', np.uint32),
    ('position_x', np.float32), ('position_y', np.float32), ('position_z', np.float32),
    ('velocity_x', np.float32), ('velocity_y', np.float32), ('velocity_z', np.float32),
    ('dt', np.float32),
    ('sfr_disk', np.float32), ('sfr_bulge', np.float32),
    ('sfr_disk_z', np.float32), ('sfr_bulge_z', np.float32),
    ('cold_gas', np.float32),
    ('metals_cold_gas', np.float32),
    ('disk_scale_radius', np.float32),
])
