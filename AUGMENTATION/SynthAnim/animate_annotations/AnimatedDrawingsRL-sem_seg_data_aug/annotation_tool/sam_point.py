
class SAMPoint():
    def __init__(self, x: int, y: int, label: bool):
        self.x = x
        self.y = y
        self.label = label  # label (bool: is inside figure?)

        self.oval_id = None
