class Mapping(object):

    def __init__(self, converter, table=None, fields=[]):
        self.converter = converter
        self.table = table if table else {}
        self.fields = fields

    def map(self, tree, name):
        atname = 'map_' + name
        if hasattr(self.converter, atname):
            return getattr(self.converter, atname)(tree)
        else:
            mapped_name = self.table.get(name, None)
            if mapped_name:
                return tree[mapped_name]
            elif name in tree.dtype.names:
                return tree[name]
            else:
                return None
