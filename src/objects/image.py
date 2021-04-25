

class Satellite_Image():
    def __init__(self, path, x, y):
        self.path = path
        self.x = x
        self.y = y
        self.brightness_score = None
        self.contrast_score = None
        self.batch_number = None
        self.classifier = ""
