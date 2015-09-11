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
            msg = 'Missing required field "%s" in module "%s".'%(field, self.module)
            msg += ' For more details, please run: "convert -i %s"'%field
            raise ValidationError(msg)

class OverLittleH(FieldValidator):

    def validate_field(self, field, fields):
        pass

class TreeLocalIndex(FieldValidator):

    def validate_fields(self, fields):
        for fld in self.fields:
            if fld not in fields:
                continue
            data = fields[fld]
            for val in data:
                if val < -1 or val >= len(data):
                    msg = 'Invalid tree-local index in "%s".'%fld
                    msg += ' Valid index range is [-1, %d), but found value of %d.'%(len(data), val)
                    raise ValidationError(msg)

class Positive(FieldValidator):

    def validate_fields(self, fields):
        for fld in self.fields:
            if fld not in fields:
                continue
            data = fields[fld]
            for val in data:
                if val < 0:
                    msg = 'Invalid value in "%s".'%fld
                    msg += ' Should be positive, but found value of %d.'%val
                    raise ValidationError(msg)

class NonZero(FieldValidator):

    def validate_fields(self, fields):
        for fld in self.fields:
            if fld not in fields:
                continue
            data = fields[fld]
            for val in data:
                if val == 0:
                    msg = 'Invalid value in "%s".'%fld
                    msg += ' Should be non-zero.'
                    raise ValidationError(msg)

class WithinRange(FieldValidator):

    def __init__(self, lower, upper, *fields):
        super(WithinRange, self).__init__(*fields)
        self.lower = lower
        self.upper = upper

    def validate_fields(self, fields):
        for fld in self.fields:
            if fld not in fields:
                continue
            data = fields[fld]
            for val in data:
                if val < self.lower or val > self.upper:
                    msg = 'Invalid value in "%s".'%fld
                    msg += ' Should be within range [%s, %s], but found value of %s.'%(self.lower, self.upper, val)
                    raise ValidationError(msg)

class Choice(FieldValidator):

    def __init__(self, choices, *fields):
        super(Choice, self).__init__(*fields)
        self.choices = set(choices)

    def validate_fields(self, fields):
        for fld in self.fields:
            if fld not in fields:
                continue
            data = fields[fld]
            for val in data:
                if val not in self.choices:
                    msg = 'Invalid choice in "%s".'%fld
                    msg += ' Valid choices are %s, but found value of %d.'%(self.choices, val)
                    raise ValidationError(msg)
