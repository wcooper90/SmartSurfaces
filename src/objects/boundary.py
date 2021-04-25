class Boundary():
    """
    Important Link!!!
    https://nominatim.openstreetmap.org/search.php?q=cherry%20hill&polygon_geojson=1&format=jsonv2&debug=1
    """
    def __init__(self, city_name):
        self.city_name = city_name
        self.boundary = None

        # just boundary for Baltimore
        self.boundary = [[-76.7112977,39.3719562],[-76.7112726,39.3543824],[-76.7112337,39.3271971],[-76.7111829,39.2916607],[-76.7111659,39.2778381],[-76.6888115,39.2680858],[-76.6597751,39.2554169],[-76.6178331,39.2371116],[-76.6116152,39.2343976],[-76.5836752,39.2081197],[-76.582525,39.2077508],[-76.5805555,39.2071192],[-76.5497285,39.1972328],[-76.5464898,39.1992526],[-76.5461562,39.1994606],[-76.5298609,39.2096221],[-76.5298503,39.2189088],[-76.5298259,39.2402128],
                        [-76.5297584,39.2991345],[-76.5297583,39.2992049],[-76.5297304,39.3239064],[-76.5297299,39.3243919],[-76.5297172,39.3347547],[-76.529677,39.3708243],[-76.5296761,39.3716015],[-76.5296757,39.3719713],[-76.569831,39.37204],[-76.5820693,39.3719669],[-76.6526604,39.3719623],[-76.6529027,39.371961],[-76.6529981,39.371961],[-76.7112977,39.3719562]]
        self.point_flipper()


    def find_boundary(self):
        pass

    def point_flipper(self):
        points = []

        # turns points into tuples, reverses coordinate order
        for point in self.boundary:
            points.append((point[1], point[0]))

        self.boundary = points



    def coordinate_to_pixel_converter(self):
        pass
