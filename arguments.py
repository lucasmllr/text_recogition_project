

class Arguments():
    '''class holding all parameters for the segmentation and classification process.

    Attributes:

        documentation (Bool): if true some additional information is printed when running the algorithm

        n (int): number of generated images
        alphabet (string): all possible characters
        min_l (int): min length of a line
        max_l (int): max length of a line
        max_lines (int): max number of lines
        lower_case (Bool): whether to use only lower case
        shape (tuple of ints): image size
        text_box (tuple of ints): size of text box in the image
        pos (tuple of float or None): if tuple of floats it corresponds to the maximum fraction of shape the top left of
                                        corner of the text box is moved, if none no variation is applied
        max_angle (float or int): max abs rotation angle of text box
        font (string): font of the text
        colorspace (string): colorspace of the images
        container (Bool): if true images are not saved as jpg but stored in a numpy array
        path (string): path to save the generated images to

        input_shape (int): input shape to the classification model
        load_path (string): path to load data from
        cut_bottom (bool): whether to cut below a threshold
        cut_top (bool): whether to cut above the threshold

        window (int): window size for which a local threshold is calculated in the sauvola method
        k (int): model parameter for sauvola thresholding
        r (int): model parameter for sauvola thresholding

        blob_t (float): threshold in range [0, 1] to binarize a normalized image
        line_t (float): threshold to be applied to a normalized historgram resulting from a projection onto the y-axis
                        in order to separate lines

        n_angles (int): number of angles evaluated in range [-angle, angle] to correct for rotatino of the text
        n_bins (int): number of bins placed on a centered 'sensor array' with the length of the image diagonal
        angle (float): abs of min and max angle evaluated for rotation correction

        min_area (int): min accepted area, i.e. number of included pixels for an MSER
        max_area (int): max accepted area for an MSER
        delta (int): model parameter for evaluation of stability value of a region

        distance (bool): whether to include eucliedean distance of top left corners of bboxes of components into evaluation
                        for neighbor criterion
        dims (bool): whether to include area and aspect ratio into evaluation of neighbor criterion
        color (bool): whether to include mean clolor into evaluation of neighbor criterion
        t_A (float or int): threshold on factor between areas of two components' bboxes to be considered neighbors
        t_asp (float or int): threshold on factor between aspect ratios of two components' bboxes to be considered neighbors
        t_color (float or int): threshold on abs mean color difference of two components to be considered neighbors
        C_d (float or int): model parameter for dynamic threshold of distance criterion
    '''

    def __init__(self):

        # documentation mode
        self.documentation = False

        # data generation
        self.n = 10
        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        self.min_l = 2
        self.max_l = 2
        self.max_lines = 1
        self.lower_case = False
        self.shape = (50, 50)
        self.text_box = (50, 50)
        self.pos = None
        self.max_angle = 5
        self.font = 'Arial'
        self.colorspace = 'RGB'
        self.container = False
        self.path = 'plot_data'

        # data loading and processing
        self.input_shape = 28
        self.load_path = 'test_data'
        self.cut_bottom = True
        self.cut_top = True
        # sauvola
        self.window = 51
        self.k = 0.1
        self.r = 128

        # blob extraction
        self.blob_t = 0.5

        # line extraction
        self.line_t = 0.002

        # rotation correction
        self.n_angles = 50
        self.n_bins = 99
        self.angle = 20.

        # MSER extraction
        self.min_area = 15
        self.max_area = 100000
        self.delta = 15
        self.normalize = True
        self.invert = True

        # component evaluation
        # neighbors
        self.distance = True
        self.dims = True
        self.color = True
        self.t_A = 10
        self.t_asp = 5
        self.t_color = 20
        self.C_d = 10

if __name__ == '__main__':

    args = Arguments()