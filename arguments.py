

class Arguments():

    def __init__(self):

        # documentation mode
        self.documentation = False

        # data generation
        self.n = 100000
        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        self.min_l = 1
        self.max_l = 1
        self.max_lines = 1
        self.lower_case = True
        self.shape = (32, 32)
        self.text_box = (32, 32)
        self.pos = None
        self.max_angle = 1
        self.font = 'Arial.ttf'
        self.font_size_range = (30, 35)
        self.font_size = None
        self.font_randomise = True
        self.colorspace = 'L'
        self.outputformat = 'container'
        self.path = 'data'

        # data loading and processing
        self.input_shape = 28
        self.load_path = 'data'
        self.cut_bottom = True
        self.cut_top = True

        # blob extraction
        self.blob_t = 0.52

        # line extraction
        self.line_t = 0.002

        # rotation correction
        self.n_angles = 50
        self.n_bins = 99
        self.angle = 20.


if __name__ == '__main__':

    args = Arguments()
