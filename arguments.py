

class Arguments():

    def __init__(self):

        # documentation mode
        self.documentation = False

        # data generation
        self.n = 5000
        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        self.char_dict = {}
        self.int_dict = {}
        for i, char in enumerate(self.alphabet):
            self.char_dict[char] = i
            self.int_dict[i] = char
        self.min_l = 5
        self.max_l = 10
        self.max_lines = 5
        self.lower_case = False
        self.shape = (800, 600)
        self.text_box = (1500, 1500)
        self.pos = (0.9, 0.9) #(0.8, 0.3)
        self.max_angle = 15
        self.font = 'arial'
        self.font_size = 50
        self.colorspace = 'RGB'
        self.container = False
        self.path = 'data'

        # data loading and processing
        self.input_shape = 32
        self.load_path = 'data'
        self.cut_bottom = True
        self.cut_top = True

        # blob extraction
        self.blob_t = .52#0.52
        self.min_pixels = 50

        # line extraction
        self.line_t = 0.002

        # rotation correction
        self.n_angles = 50
        self.n_bins = 99
        self.angle = 20.


if __name__ == '__main__':

    args = Arguments()