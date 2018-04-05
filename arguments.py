

class Arguments():

    def __init__(self):

        # documentation mode
        self.documentation = False

        # data generation
        self.n = 10
        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        self.min_l = 5
        self.max_l = 10
        self.max_lines = 5
        self.lower_case = False
        self.shape = (400, 300)
        self.text_box = (250, 250)
        self.pos = None
        self.max_angle = 15
        self.font = 'Arial'
        self.colorspace = 'RGB'
        self.container = False
        self.path = 'data'

        # data loading and processing
        self.input_shape = 28
        self.load_path = 'test_data'
        self.cut_bottom = True
        self.cut_top = True
        # sauvola
        self.window = 31
        self.k = 0.1
        self.r = 128


        # blob extraction
        self.blob_t = 0.52

        # line extraction
        self.line_t = 0.002

        # rotation correction
        self.n_angles = 50
        self.n_bins = 99
        self.angle = 20.

        # MSER extraction
        self.min_area = 25
        self.max_area = 50000
        self.delta = 15
        self.normalize = True
        self.invert = True

        #component evaluation
        #neighbors
        self.distance = True
        self.dims = True
        self.color = True
        self.t_A = 5
        self.t_asp = 5
        self.t_color = 20

if __name__ == '__main__':

    args = Arguments()