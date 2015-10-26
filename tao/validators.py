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
            if min(data) < -1 or max(data) >= len(data):
                msg = 'Invalid tree-local index in "%s".'%fld
                msg += ' Valid index range is [-1, %d), but found value of min,max=[%d,%d].'%(len(data), min(data),max(data))
                raise ValidationError(msg)
                

class Positive(FieldValidator):

    def validate_fields(self, fields):
        for fld in self.fields:
            if fld not in fields:
                continue

            data = fields[fld]
            if min(data) < 0:
                msg = 'Invalid value in "%s".'%fld
                msg += ' Should be positive, but found value of %s.'%min(data)
                raise ValidationError(msg)

        
class NonZero(FieldValidator):

    def validate_fields(self, fields):
        for fld in self.fields:
            if fld not in fields:
                continue

            data = set(fields[fld])
            if 0 in data:
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
            if min(data) < self.lower or max(data) > self.upper:
                    msg = 'Invalid values in "%s".'%fld
                    msg += ' Should be within range [%s, %s], but found value of min,max=[%s,%s].'%(self.lower, self.upper, min(data),max(data))
                    raise ValidationError(msg)
                
        
class WithinCRange(FieldValidator):

    def __init__(self, lower, upper, *fields):
        super(WithinCRange, self).__init__(*fields)
        self.lower = lower
        self.upper = upper

    def validate_fields(self, fields):
        for fld in self.fields:
            if fld not in fields:
                continue

            data = fields[fld]
            if min(data) < self.lower or max(data) > (self.upper-1):
                    msg = 'Invalid value in "%s".'%fld
                    msg += ' Should be within range [%s, %s], but found value of min,max= [%s,%s].'%(self.lower, self.upper, min(data), max(data))
                    raise ValidationError(msg)

            
class Choice(FieldValidator):

    def __init__(self, choices, *fields):
        super(Choice, self).__init__(*fields)
        self.choices = set(choices)

    def validate_fields(self, fields):
        for fld in self.fields:
            if fld not in fields:
                continue

            diff = set(fields[fld]) - self.choices
            if len(diff) > 0:
                msg = 'Invalid choice in "%s".'%fld
                msg += ' Valid choices are %s, but found %s.'%(self.choices, diff)
                raise ValidationError(msg)
                
