class ValidationError(Exception):
    pass

class Validator(object):
    pass

class FieldValidator(Validator):

    def __init__(self, *fields):
        self.fields = fields
        self.module = None

    def validate_fields(self, fields):
        for fld in self.fields:
            self.validate_field(fld, fields)

class Required(FieldValidator):

    def validate_field(self, field, fields):
        if field not in fields:
            raise ValidationError(
                'Missing required field "%s" in module "%s".'%(field, self.module)
            )

class OverLittleH(FieldValidator):

    def validate_field(self, field, fields):
        pass
