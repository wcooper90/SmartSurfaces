import sys

class LandChunk():

    def __init__(self, image):
        self.image = image
        self.contoured = None
        self.contours = 0
        self.albedo = sys.max
        self.trees = sys.max
        self.LSroofs = sys.max
        self.HSroofs = sys.max
        self.percentGreen = sys.max

    def calculate_albedo(self):
        return 0

    def find_contours(self):
        return 0

    def calculate_green(self):
        return 0

    def calculate_trees(self):
        return 0

    def calculate_LSroofs(self):
        return 0

    def calculate_HSroofs(self):
        return 0
