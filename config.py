import tensorflow as tf

class Config():

    def __init__(self,
                 seq_len=800,
                 num_classes=10,
                 vocab_size=30000,
                 embedding_size = 300,
                 dropout_rate = 0.8,
                 x_train=None,
                 y_train=None,
                 x_test=None,
                 y_test=None,
                 pretrain_vec=None):

        # self.args = args
        self.init_lr =0.9
        self.batch_size = 32
        self.epochs = 10
        self.l2= 0.002
        self.routing = True

        self.sequence_length = seq_len
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.dropout_ratio = 0.8
        self.pretrain_vec = pretrain_vec
        self.filter_size = 3
        self.num_filters = 16