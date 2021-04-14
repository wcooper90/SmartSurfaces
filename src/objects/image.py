

class Image():
    def __init__(self, path):
        self.path = path
        self.brightness_score = None
        self.contrast_score = None
        self.batch_number = None
        self.classifier = ""
