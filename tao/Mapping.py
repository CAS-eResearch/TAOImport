class Mapping(object):

    def __init__(self, table=None):
        self.table = table if table else {}

    def map(self, tree, name):
        atname = 'map_' + name
        if hasattr(self, atname):
            return getattr(self, atname)(tree)
        else:
            mapped_name = self.table.get(name, None)
            if mapped_name:
                return tree[mapped_name]
            elif name in tree.dtype.names:
                return tree[name]
            else:
                return None
