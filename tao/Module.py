from Mapping import Mapping

def set_module(mod, obj):
    obj.module = mod
    return obj

class Module(object):

    def __init__(self, arguments=None):
        self.mapping = None
        self.validators = [set_module(self, v) for v in getattr(self, 'validators', [])]
        self.generators = [set_module(self, g) for g in getattr(self, 'generators', [])]
        self.disabled = False
        self.parse_arguments(arguments)

    def __str__(self):
        return self.__class__.__name__

    @classmethod
    def add_arguments(cls, parser):
        pass

    def parse_arguments(self, args):
        pass

    def convert_tree(self, src_tree, fields):
        if self.disabled:
            return
        for name in self.fields.iterkeys():
            data = self.mapping.map(src_tree, name)
            if data is not None:
                fields[name] = data

    def generate_fields(self, fields):
        if self.disabled:
            return
        for generator in self.generators:
            generator.generate_fields(fields)

    def validate_fields(self, fields):
        if self.disabled:
            return
        for validator in self.validators:
            validator.validate_fields(fields)
