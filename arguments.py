

class Arguments():

    def __init__(self):

        # documentation mode
        self.documentation = False

        # data generation
        self.n = 50
        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        self.min_l = 1
        self.max_l = 1
        self.max_lines = 1
        self.lower_case = False
        self.shape = (30, 30)
        self.text_box = (30, 30)
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


if __name__ == '__main__':

    args = Arguments()