

class Arguments():

    def __init__(self):

        # data generation
        self.n = 10
        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        self.min_l = 5
        self.max_l = 10
        self.max_lines = 5
        self.lower_case = False
        self.shape = (400, 300)
        self.text_box = (250, 250)
        self.pos = (0.8, 0.3)
        self.max_angle = 15
        self.font = 'Arial'
        self.colorspace = 'RGB'
        self.container = False
        self.path = 'data'


if __name__ == '__main__':

    args = Arguments()