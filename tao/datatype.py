import h5py
import numpy as np

datatype = np.dtype([
    ('snapshot', np.int32),
    ('type',     np.int32),
    ('position_x', np.float32), ('position_y', np.float32), ('position_z', np.float32),
    ('velocity_x', np.float32), ('velocity_y', np.float32), ('velocity_z', np.float32),
])
