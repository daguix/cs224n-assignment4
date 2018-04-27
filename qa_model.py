from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from datetime import datetime
import logging
import sys #Progbar
import os

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.gen_math_ops import _batch_mat_mul as batch_matmul

from evaluate import exact_match_score, f1_score

tf.app.flags.DEFINE_integer("epochs", 7, "Number of epochs to train.")
tf.app.flags.DEFINE_float  ("cross_id_bias", -1, "ID coefficient to init attention multiplier matrix. (Use -1 to avoid using matrix altogether)")
tf.app.flags.DEFINE_integer("max_q", 80, "Max context length")
tf.app.flags.DEFINE_integer("max_c", 300, "Max question length")
tf.app.flags.DEFINE_boolean("log_losses", False, "Collect batch losses for plotting.")
tf.app.flags.DEFINE_string ("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_float  ("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float  ("learning_rate_decay", 0.9999, "Learning rate.")
tf.app.flags.DEFINE_integer("rnn_size", 100, "Size of each RNN layer.")
tf.app.flags.DEFINE_integer("hidden_size", 0, "Size of Decoder's hidden layer (defaults to embedding size.")
tf.app.flags.DEFINE_boolean("train_embeddings", False, "Train embedding vectors")
tf.app.flags.DEFINE_boolean("concat_encoding", True, "Concat encoding with LSTM decoding for classification")
tf.app.flags.DEFINE_float  ("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_boolean("evaluate_epoch", True, "Run full EM/F1 evaluation at the end of each epoch.")
tf.app.flags.DEFINE_boolean("save_epoch", True, "Auto-save the model at the end of each epoch.")
tf.app.flags.DEFINE_boolean("AD", True, "Use most context-relevant question words in encoding")
tf.app.flags.DEFINE_boolean("AQ", True, "Use most question-relevant context words in encoding")
tf.app.flags.DEFINE_string ("start_decoder", "GRU", "Start token decoding layer (BiLSTM, LSTM, BiGRU, GRU '')")
tf.app.flags.DEFINE_string ("end_decoder", "GRU", "End token decoding layer (BiLSTM, LSTM, BiGRU, GRU '')")
tf.app.flags.DEFINE_integer("start_layers", 2, "Number of start token decoding RNN layers 0+")
tf.app.flags.DEFINE_integer("end_layers", 2, "Number of end token decoding layers 0+")
tf.app.flags.DEFINE_string ("encoder_rnn", "BiLSTM", "End token decoding layer (BiLSTM, BiGRU, '')")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_float  ("max_gradient_norm", 5.0, "Clip gradients to this norm.")


logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, exact=None):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """
        values = values or []
        exact = exact or []

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if isinstance(self.sum_values[k], list):
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=None):
        self.update(self.seen_so_far+n, values)


def get_minibatches(data, minibatch_size, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:

        for inputs_minibatch in get_minibatches(inputs, minibatch_size):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in get_minibatches([inputs, labels], minibatch_size):
            ...

    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatch_size: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.

    """
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)


def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def mask_softmax(logits, axis, length):
    shape = logits.get_shape().as_list()
    assert len(shape) == 3 #TODO Generalize???

    mask = tf.to_float(tf.sequence_mask(length, maxlen=shape[axis]))

    assert axis == 1 or axis == 2 #TODO Generalize???
    mask = tf.expand_dims(mask, 3-axis)

    return logits-1e30*(1-mask)

def softmax_partial(logits, axis, length):
    logits = mask_softmax(logits, axis, length)
    softmax = tf.nn.softmax(logits, dim= axis if axis != 2 else -1)

    return softmax

class Config:
    embed_size = 0
    rnn_size = 0

    def __init__(self, embedding_size): #TODO from FLAGS
        self.embed_size = embedding_size
        self.rnn_size = FLAGS.rnn_size or self.embed_size

class Encoder(object):
    def __init__(self, config, pretrained_embeddings):

        self.max_c = FLAGS.max_c
        self.max_q = FLAGS.max_q

        self.pretrained_embeddings = pretrained_embeddings
        self.embed_size = config.embed_size
        assert self.embed_size == pretrained_embeddings.shape[1]
        
        self.rnn_size = config.rnn_size

        # Defining placeholders.
        self.q_ids_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.max_q], name='q_ids')
        self.c_ids_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, self.max_c], name='c_ids')

    def encode(self, c_len_placeholder, q_len_placeholder): #TODO???, inputs, masks, encoder_state_input):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        if FLAGS.train_embeddings:
            embeddings = tf.Variable(self.pretrained_embeddings, name='trainable_embeddings', dtype=tf.float32)
        else:
            embeddings = tf.constant(self.pretrained_embeddings, name='pretrained_embeddings', dtype=tf.float32)

        q_vectors = tf.nn.embedding_lookup(params=embeddings, ids=self.q_ids_placeholder)
        assert q_vectors.get_shape().as_list() == [None, self.max_q, self.embed_size]

        c_vectors = tf.nn.embedding_lookup(params=embeddings, ids=self.c_ids_placeholder)
        assert c_vectors.get_shape().as_list() == [None, self.max_c, self.embed_size]

        #From now on following terminology from https://arxiv.org/pdf/1611.01604.pdf

        l = 2*self.rnn_size #TODO sentinel vector
        mplus1 = self.max_c
        nplus1 = self.max_q

        encoding_size = l

        xavier_initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("Encoder_rnn") as scope:

            if FLAGS.encoder_rnn == 'BiLSTM':
                cell = tf.nn.rnn_cell.LSTMCell(num_units=self.rnn_size, initializer=xavier_initializer)

            if FLAGS.encoder_rnn == 'BiGRU':
                cell = tf.nn.rnn_cell.GRUCell(num_units=self.rnn_size)

            c_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell, cell, c_vectors, dtype=tf.float32, sequence_length=c_len_placeholder, scope=scope)

            scope.reuse_variables()

            q_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell, cell, q_vectors, dtype=tf.float32, sequence_length=q_len_placeholder, scope=scope)

        D = tf.concat_v2(c_outputs,2)
        assert D.get_shape().as_list() == [None, mplus1, l]

        Q = tf.concat_v2(q_outputs, 2)
        assert Q.get_shape().as_list() == [None, nplus1, l]

        if FLAGS.cross_id_bias >= 0:
            U = tf.Variable(name="U", initial_value = FLAGS.cross_id_bias*np.identity(l)+tf.random_uniform((l, l), -.01, .01), dtype=tf.float32)
            Q = tf.reshape(tf.matmul(tf.reshape(Q,[-1, l]), U), [-1, nplus1, l]) #TODO tensordot

            tf.summary.histogram('Cross_Attn_U', U)

        L = batch_matmul(Q,D, adj_y=True)
        assert L.get_shape().as_list() == [None, nplus1, mplus1]
        tf.summary.histogram('Attn_L', L)

        encoding = D
        encoding_size = l

        if FLAGS.AD:
            if FLAGS.AQ:
                A_Q = softmax_partial(L, 2, c_len_placeholder)
                assert A_Q.get_shape().as_list() == L.get_shape().as_list()
                tf.summary.histogram('A_Q', A_Q)

                C_Q = batch_matmul(A_Q, D)
                assert C_Q.get_shape().as_list() == [None, nplus1, l]
                tf.summary.histogram('C_Q', C_Q)

                Q = tf.concat_v2([Q, C_Q], 2)
                encoding_size += l

            A_DT = softmax_partial(L, 1, q_len_placeholder)
            assert A_DT.get_shape().as_list() == L.get_shape().as_list()
            tf.summary.histogram('A_DT', A_DT)

            C_D = batch_matmul(A_DT, Q, adj_x=True)
            assert C_D.get_shape().as_list() == [None, mplus1, encoding_size]
            tf.summary.histogram('C_D', C_D)

            encoding = tf.concat_v2([encoding, C_D], 2)
            encoding_size += l

        assert encoding.get_shape().as_list() == [None, mplus1, encoding_size]
        return encoding



class Decoder(object):
    def __init__(self, config):
        self.rnn_size = config.rnn_size

    def decode_pos(self, rnn_layers, rnn, encoded, endcoded_len, dropout):
        max_encoded   = encoded.get_shape().as_list()[1]
        encoding_size = encoded.get_shape().as_list()[2]

        xavier_initializer = tf.contrib.layers.xavier_initializer()
        keep_prob = 1-dropout

        decoded = encoded
        decoded_size = encoded.get_shape().as_list()[2]

        for layer in range(rnn_layers):
            with tf.variable_scope('Decoder_rnn' + str(layer)):
                decoded = tf.nn.dropout(decoded, keep_prob)

                if rnn == 'BiLSTM':
                    cell = tf.nn.rnn_cell.LSTMCell(num_units=self.rnn_size, initializer=xavier_initializer)

                    outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                        cell, cell, decoded, dtype=tf.float32, sequence_length=endcoded_len)

                    decoded = tf.concat_v2(outputs, 2)
                    decoded_size = self.rnn_size*2

                if rnn == 'LSTM':
                    cell = tf.nn.rnn_cell.LSTMCell(num_units=self.rnn_size, initializer=xavier_initializer)

                    outputs, _ = tf.nn.dynamic_rnn(
                        cell, decoded, dtype=tf.float32, sequence_length=endcoded_len)

                    decoded = outputs
                    decoded_size = self.rnn_size

                if rnn == 'BiGRU':
                    cell = tf.nn.rnn_cell.GRUCell(num_units=self.rnn_size)

                    outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                        cell, cell, decoded, dtype=tf.float32, sequence_length=endcoded_len)

                    decoded = tf.concat_v2(outputs, 2)
                    decoded_size = self.rnn_size*2

                if rnn == 'GRU':
                    cell = tf.nn.rnn_cell.GRUCell(num_units=self.rnn_size)

                    outputs, _ = tf.nn.dynamic_rnn(
                        cell, decoded, dtype=tf.float32, sequence_length=endcoded_len)

                    decoded = outputs
                    decoded_size = self.rnn_size

        if rnn and FLAGS.concat_encoding:
            encoded = tf.nn.dropout(encoded, keep_prob)

            decoded = tf.concat_v2([decoded, encoded],2)
            decoded_size += encoding_size

        assert decoded.get_shape().as_list() == [None, max_encoded, decoded_size]
        decoded = tf.reshape(decoded, [-1, decoded_size])

        if FLAGS.hidden_size:
            hidden_size = FLAGS.hidden_size

            bh = tf.get_variable("bh", shape=[hidden_size], initializer=tf.constant_initializer(0.01), dtype=tf.float32)
            Uh = tf.get_variable("Uh", shape=[decoded_size, hidden_size],
                            initializer=xavier_initializer, dtype=tf.float32)

            decoded = tf.nn.dropout(decoded, keep_prob)

            decoded = tf.nn.relu(bh + tf.matmul(decoded, Uh)) # tf.tensordot(decoded, Uh, [[-1], [0]]))
            decoded_size = hidden_size

        n_classifiers = 1

        bs = tf.get_variable("bs", shape=[n_classifiers], initializer=tf.constant_initializer(0), dtype=tf.float32)
        Us = tf.get_variable("Us", shape=[decoded_size, n_classifiers],
                            initializer=xavier_initializer, dtype=tf.float32)

        decoded = tf.nn.dropout(decoded, keep_prob)

        pred   = bs + tf.matmul(decoded,  Us)
        assert pred.get_shape().as_list() == [None, n_classifiers]

        return tf.reshape(pred, [-1, max_encoded, n_classifiers])

    def decode(self, encoded, encoded_len, dropout):
        with tf.variable_scope("Decode_start"):
            start = self.decode_pos(FLAGS.start_layers, FLAGS.start_decoder, encoded, encoded_len, dropout)

        with tf.variable_scope("Decode_end"):
            end = self.decode_pos(FLAGS.end_layers, FLAGS.end_decoder, encoded, encoded_len, dropout)

        return tf.concat_v2([start, end], 2)

class QASystem(object):
    def __init__(self, train_dir, pretrained_embeddings):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """

        # Save your model parameters/checkpoints here

        self.config = Config(pretrained_embeddings.shape[1])

        self.encoder = Encoder(self.config, pretrained_embeddings)
        self.decoder = Decoder(self.config)

        self.train_dir = train_dir

        # ==== set up placeholder tokens ========
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, name='dropout')

        self.q_len_placeholder = tf.placeholder(dtype=tf.int32, shape=[None], name='q_len')
        self.c_len_placeholder = tf.placeholder(dtype=tf.int32, shape=[None], name='c_len')

        self.span_placeholder  = tf.placeholder(dtype=tf.int32, shape=[None,2], name='span')

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        step = tf.Variable(0, trainable=False)
        rate = tf.train.exponential_decay(FLAGS.learning_rate, step, 1, FLAGS.learning_rate_decay)
        optimizer = get_optimizer(FLAGS.optimizer)(rate)

        grads, vars = zip(*optimizer.compute_gradients(self.loss))
        if FLAGS.max_gradient_norm:
            cgrads, self.grad_norm = tf.clip_by_global_norm(grads, FLAGS.max_gradient_norm)
        else:
            cgrads = grads
            self.grad_norm = tf.global_norm(grads)

        self.train_op = optimizer.apply_gradients(zip(cgrads, vars))

        # for n in tf.get_default_graph().as_graph_def().node:
        #    print(n.name)

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        encoding   = self.encoder.encode(self.c_len_placeholder, self.q_len_placeholder)
        max_encoded = encoding.get_shape().as_list()[1]

        self.prediction = self.decoder.decode(encoding, self.q_len_placeholder, self.dropout_placeholder)
        assert self.prediction.get_shape().as_list() == [None, max_encoded, 2]

        variable_summaries(self.prediction)

        self.prediction = mask_softmax(self.prediction, 1, self.c_len_placeholder)
        self.softmax = tf.nn.softmax(self.prediction, dim=1)
        assert self.softmax.get_shape().as_list() == [None, max_encoded, 2]

        multiword = tf.argmax( self.softmax, 1 )
        assert multiword.get_shape().as_list() == [None, 2]

        sngleword = tf.argmax( tf.reduce_prod( self.softmax, 2, keep_dims=True ), 1)
        assert sngleword.get_shape().as_list() == [None, 1]

        start = tf.slice(multiword, [0,0], [-1,1])
        end   = tf.slice(multiword, [0,1], [-1,1])
        conflicts = tf.greater(start, end)

        # Wherever start conflict with end choose best single word answer
        self.result = tf.to_int32( tf.where( tf.tile(conflicts, [1,2]), tf.tile(sngleword, [1,2]), multiword ))
        assert self.result.get_shape().as_list() == [None,2]

        tf.summary.histogram('Answer_Len', tf.slice(self.result, [0,1], [-1,1])-tf.slice(self.result, [0,0], [-1,1])+1)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        max_encoded = self.softmax.get_shape().as_list()[1]

        with vs.variable_scope("loss"):
            assert self.prediction.get_shape().as_list() == [None, max_encoded, 2]
            logits = tf.transpose(self.prediction, perm=[0, 2, 1])
            assert logits.get_shape().as_list() == [None, 2, max_encoded]

            assert self.span_placeholder.get_shape().as_list() == [None, 2]
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.span_placeholder)
            self.loss = tf.reduce_mean(losses)

            tf.summary.scalar('cross_entroy_loss',self.loss)

        with vs.variable_scope("em"):
            eq = tf.equal( self.result, self.span_placeholder )
            assert eq.get_shape().as_list() == [None, 2]

            both = tf.reduce_min( tf.to_float(eq), 1)
            assert both.get_shape().as_list() == [None]

            self.em = tf.reduce_mean(both)
            tf.summary.scalar('EM_Accuracy', self.em)

            #setup tensorboard
            self.summary = tf.summary.merge_all()


    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            pass

    def optimize(self, session, train):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        c_ids, c_len, q_ids, q_len, span = train

        input_feed = {
            self.encoder.c_ids_placeholder: c_ids,
            self.encoder.q_ids_placeholder: q_ids,

            self.c_len_placeholder: c_len,
            self.q_len_placeholder: q_len,

            self.dropout_placeholder: FLAGS.dropout,

            self.span_placeholder : span,
        }

        output_feed = [self.train_op, self.loss, self.em, self.summary] #TODO??? gradient and param norm???

        outputs = session.run(output_feed, input_feed)

        return outputs[1:] # All but the optimizer

    def test(self, session, valid):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        c_ids, c_len, q_ids, q_len, span = valid

        input_feed = {
            self.encoder.c_ids_placeholder: c_ids,
            self.encoder.q_ids_placeholder: q_ids,

            self.c_len_placeholder: c_len,
            self.q_len_placeholder: q_len,

            self.dropout_placeholder: 0,

            self.span_placeholder : span,
        }

        output_feed = [self.loss, self.em] #TODO??? gradient and param norm???

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        c_ids, c_len, q_ids, q_len = test_x

        input_feed = {
            self.encoder.c_ids_placeholder: [c_ids],
            self.encoder.q_ids_placeholder: [q_ids],

            self.c_len_placeholder: [c_len],
            self.q_len_placeholder: [q_len],

            self.dropout_placeholder: 0,
        }

        output_feed = [self.result]

        outputs = session.run(output_feed, input_feed)
        return outputs

    def answer(self, session, test_x):
        #TODO, once answer returns a_s and a_e, then we need to fix this function.
        a = self.decode(session, test_x)
        a_s = a[0][0][0]
        a_e = a[0][0][1]
        # a_s = np.argmax(yp, axis=1)
        # a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, session, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        loss_validation = 0

        batch_count = (len(valid_dataset[0]) + FLAGS.batch_size-1) // FLAGS.batch_size
        prog = Progbar(target=batch_count)
        losses = []
        # run over the minibatch size for validation dataset
        for i, batch in enumerate(get_minibatches(valid_dataset, FLAGS.batch_size)):
            loss_validation, em_validation = self.test(session, batch)

            losses.append([loss_validation, em_validation])
            prog.update(i + 1, [("Validation loss", loss_validation), ("Validation em", em_validation*100)])

        mean = np.mean(losses, axis = 0)
        logging.info("Validation logged mean: loss : %f, EM = %f %%", mean[0], mean[1]*100)

        return losses

    def evaluate_answer(self, session, dataset, samples = 100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param log: whether we print to std out stream
        :return:
        """
        samples = min(samples, len(dataset[0]))

        c_ids, c_len, q_ids, q_len, span = dataset

        f1 = 0.
        em = 0.

        for index in range(samples):
            a_s, a_e = self.answer(session, (c_ids[index], c_len[index], q_ids[index], q_len[index]))
            answers = c_ids[index][a_s: a_e+1]
            p_s, p_e = span[index]
            true_answer = c_ids[index][p_s: p_e+1]

            answers = " ".join(str(a) for a in answers)
            true_answer = " ".join(str(ta) for ta in true_answer)

            f1 += f1_score(answers, true_answer)
            em += exact_match_score(' '.join(str(a) for a in answers), ' '.join(str(ta) for ta in true_answer))
            #logging.info("answers %s, true_answer %s" % (answers, true_answer))

        f1 /=samples
        em /=samples

        if log:
            logging.info("F1: {:.2%}, EM: {:.2%}, for {} samples".format(f1, em, samples))

        return f1, em

    def preprocess_sequence_data(self, dataset):
        max_c = FLAGS.max_c
        max_q = FLAGS.max_q

        stop = next(( idx for idx, xi in enumerate(dataset) if len(xi[0])>max_c), len(dataset))
        assert len(dataset[stop-1][0])<=max_c

        c_ids = np.array([xi[0]+[0]*(max_c-len(xi[0])) for xi in dataset[:stop]], dtype=np.int32)
        q_ids = np.array([xi[1]+[0]*(max_q-len(xi[1])) for xi in dataset[:stop]], dtype=np.int32)

        span = np.array([xi[2] for xi in dataset[:stop]], dtype=np.int32)

        c_len  = np.array([len(xi[0]) for xi in dataset[:stop]], dtype=np.int32)
        q_len  = np.array([len(xi[1]) for xi in dataset[:stop]], dtype=np.int32)

        data_size = c_ids.shape[0]

        assert q_ids.shape[0] == data_size
        assert  c_ids.shape == (data_size, max_c)
        assert  q_len.shape == (data_size,)
        assert  c_len.shape == (data_size,)
        assert   span.shape == (data_size,2)

        return [c_ids, c_len, q_ids, q_len, span]


    def run_epoch(self, session, epoch, writer, train, dev):
        batch_count = (len(train[0]) + FLAGS.batch_size-1) // FLAGS.batch_size
        prog = Progbar(target=batch_count)
        losses = []
        # run over the minibatch size
        for i, batch in enumerate(get_minibatches(train, FLAGS.batch_size)):
            loss_train, em_train, summary = self.optimize(session, batch)

            writer.add_summary(summary, epoch*batch_count+i)

            loss_dev, em_dev = self.test(session, dev)

            losses.append([loss_train, loss_dev])
            prog.update(i + 1, [("train loss", loss_train), ("train em", em_train*100), ("dev loss", loss_dev), ("dev em", em_dev*100)])

        mean = np.mean(losses, axis = 0)
        logging.info("Logged mean epoch losses: train : %f dev : %f ", mean[0], mean[1])

        return losses

    def train(self, session, dataset):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in self.train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()

        results_path = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
        model_path = results_path + "model.weights/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        saver2 = tf.train.Saver()
        saver = tf.train.Saver()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        corpus = self.preprocess_sequence_data(dataset)

        train, valid = list( get_minibatches(corpus, len(corpus[0])*9//10))
        dev =  get_minibatches(valid, 16).next()

        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', session.graph)
        val_writer   = tf.summary.FileWriter(FLAGS.log_dir + '/val')

        # run the number of epochs for the train
        losses = []
        losses_validation = []
        best_f1 = 0

        for epoch in range(FLAGS.epochs):
            logging.info("Epoch %d out of %d", epoch + 1, FLAGS.epochs)
            loss = self.run_epoch(session, epoch, train_writer, train, dev)

            if FLAGS.evaluate_epoch:
                logging.info("Starting Answer Evaluation")
                f1, em = self.evaluate_answer(session, valid, 100, log=True)

                logging.info("Starting data validation step")
                loss_validation = self.validate(session, valid)

            if f1 > best_f1:
                logging.info("New Best F1 Score!!! %f", f1*100)
                best_f1 = f1
                if FLAGS.save_epoch:
                    logging.info("Checkpoint Saved %s %s" %(self.train_dir, model_path))
                    saver.save(session, self.train_dir+"/model.weights")
                    saver2.save(session, model_path+"model.weights") #save to current working directory as well just to be safe!

            if FLAGS.log_losses:
                losses.append(loss)
                losses_validation.append(loss_validation)

        #TODO plot or return losses

        logging.info("Best F1 Score = %f" % (best_f1*100))
        saver.save(session, self.train_dir+"/model.weights")

        best_f1_path = results_path[:-1]+"_"+str(round(best_f1,2))+"_f1"
        os.rename(model_path[:-1],  best_f1_path)
        os.rename("log/log.txt",    best_f1_path+"/log.txt")
        os.rename("log/flags.json", best_f1_path+"/flags.json")



class SquareTest(tf.test.TestCase):

    def testSquare(self):
        with self.test_session():
          c = tf.constant([[[2., 2., 2., 2.],
                            [5., 5., 5., 5.],
                            [-100,-100,-100, -100]],
                           [[2., 2., 2., 2.],
                            [5., 5., 5., 5.],
                            [-100,-100,-100, -100]]])

          result = tf.nn.softmax(c, -1).eval()

          self.assertAllClose(result, [[[.25,.25,.25,.25],
                                        [.25,.25,.25,.25],
                                        [.25,.25,.25,.25]],
                                       [[.25,.25,.25,.25],
                                        [.25,.25,.25,.25],
                                        [.25,.25,.25,.25]]])

          result = tf.nn.softmax(c, 1).eval()
          """
          self.assertAllClose(result, [[[  3.35350138e-04,   3.35350138e-04,   3.35350138e-04,   3.35350138e-04],
                                        [  9.99664664e-01,   9.99664664e-01,   9.99664664e-01,   9.99664664e-01]],

                                       [[  3.35350138e-04,   3.35350138e-04,   3.35350138e-04,   3.35350138e-04],
                                        [  9.99664664e-01,   9.99664664e-01,   9.99664664e-01,   9.99664664e-01]]])
          """


          result = softmax_partial(c, 2, [2, 3]).eval()
          print(result)

          result = softmax_partial(c, 1, [2, 3]).eval()
          print(result)

          result = softmax_partial(c, 2, [1, 2]).eval()
          print(result)

          result = softmax_partial(c, 1, [1, 2]).eval()
          print(result)

if __name__ == '__main__':
    tf.test.main()
