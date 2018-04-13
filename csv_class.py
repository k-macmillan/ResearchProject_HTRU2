

class CSV:
    """Allows standard formatting of any csv file"""
    def __init__(self, name='', col_names=[], label_name='', col_defaults=[[]], num_examples={'train': 0, 'test': 0}, classes=0):
        """Default initializer for the class"""
        self.name = name
        self.col_names = col_names
        self.label_name = label_name
        self.col_defaults = col_defaults
        self.num_examples = num_examples
        self.classes = classes
        self.accuracy = 0.0
