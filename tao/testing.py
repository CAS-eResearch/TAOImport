import numpy as np

def random_tree(src_type):
    size = np.random.randint(1, 10)
    tree = np.empty(size, src_type)
    for name in src_type.names:
        tree[name] = np.random.rand(size)
    return tree
