

class Image_Curator():
    def __init__ (self, geojson_path):
        self.status = "complete"
        # 2 dimensional array to represent the batches of images
        self.image_coordinates = []
        self.batches = []

    # interpret geojson file, find image coordinates
    def geojson_reader(self):
        pass

    # batch image coordinates
    def batcher(self):
        pass


    # call create_images with the image coordinates
    def curate_images(self):
        pass



    # run through images to produce an average brightness, contrast, classification
    # run through images again to assign standarization values to each image object
    def evaluate_images(self):

        pass


    # call image altering functions 
    def standardize_images(self):
        pass


    # control flow for this object, returns image objects in batches
    def control_flow(self):
        pass
