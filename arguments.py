class Arguments():

    def __init__(self):

        # documentation mode
        self.documentation = False

        # data generation
        self.n = 100
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
        self.font = 'all'
        self.font_size = 50
        self.colorspace = 'RGB'
        self.container = False
        self.image_path = 'data'
        self.train_path = 'data'
        self.safe_override = False

        # data loading and processing
        self.method = 'threshold'  # 'threshold' or 'mser'

        self.input_shape = 32
        self.load_path = 'data'
        self.cut_bottom = True
        self.cut_top = True

        # sauvola
        self.window = 31
        self.k = 0.1
        self.r = 128


        # blob extraction
        self.blob_t = .52#0.52
        self.min_pixels = 50

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


        # NN training
        self.batch_size = 300
        self.epochs = 500
        self.lr = 0.003
        self.momentum = 0.5
        self.test_size = 0.2
        self.seed = 123
        self.log_interval = 1#100
        self.shuffle = True
        self.model_path = 'model_weights'


if __name__ == '__main__':

    args = Arguments()