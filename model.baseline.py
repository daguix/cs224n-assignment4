import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

from os.path import join as pjoin
import numpy as np
import tensorflow as tf
from utils import Progbar

from evaluate import evaluate

DATA_DIR = "./data/squad"


def load_from_file(file):
    with open(pjoin(DATA_DIR, file), "r") as f:
        return np.array([list(map(int, line.strip().split()))
                         for line in f])


def create_dataset(file):
    dataset = tf.data.TextLineDataset(pjoin(DATA_DIR, file))
    string_split = dataset.map(lambda string: tf.string_split([string]).values)
    integer_dataset = string_split.map(
        lambda x: tf.string_to_number(x, out_type=tf.int32))
    return integer_dataset


def with_length(dataset):
    with_length = dataset.map(lambda x: (x, tf.size(x)))
    return with_length


def load_word_embeddings():
    return np.load(pjoin(DATA_DIR, "glove.trimmed.100.npz"))["glove"].astype(np.float32)


def load_vocabulary():
    with open(pjoin(DATA_DIR, "vocab.dat"), "r") as f:
        return np.array([line.strip() for line in f])


def convert_indices_to_text(vocabulary, context, start, end):
    if end < start:
        return ''
    elif end >= len(context):
        return ''
    else:
        return ' '.join(np.take(vocabulary, np.take(context, range(start, end+1))))


def preprocess_softmax(tensor, mask):
    inverse_mask = tf.subtract(tf.constant(1.0), tf.cast(mask, tf.float32))
    penalty_value = tf.multiply(inverse_mask, tf.constant(-1e9))
    return tf.where(mask, tensor, penalty_value)


def bilstm(question_embeddings, question_lengths, lstm_hidden_size, keep_prob=1.0):
    lstm_cell_fw = tf.nn.rnn_cell.GRUCell(
        lstm_hidden_size, name="gru_cell_fw")
    lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
        lstm_cell_fw, input_keep_prob=keep_prob)
    lstm_cell_bw = tf.nn.rnn_cell.GRUCell(
        lstm_hidden_size, name="gru_cell_bw")
    lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
        lstm_cell_bw, input_keep_prob=keep_prob)

    (question_output_fw, question_output_bw), (question_output_final_fw, question_output_final_bw) = tf.nn.bidirectional_dynamic_rnn(
        lstm_cell_fw, lstm_cell_bw, question_embeddings, sequence_length=question_lengths, dtype=tf.float32, time_major=False)

    question_output = tf.concat(
        [question_output_fw, question_output_bw], 2)

    question_output_final = tf.concat(
        [question_output_final_fw, question_output_final_bw], 1)
    return (question_output, question_output_final)


class QRNN_f_pooling(tf.nn.rnn_cell.RNNCell):
    def __init__(self, out_fmaps):
        self.__out_fmaps = out_fmaps

    @property
    def state_size(self):
        return self.__out_fmaps

    @property
    def output_size(self):
        return self.__out_fmaps

    def __call__(self, inputs, state, scope=None):
        """
        inputs: 2-D tensor of shape [batch_size, Zfeats + [gates]]
        """
        # pool_type = self.__pool_type
        print('QRNN pooling inputs shape: ', inputs.get_shape())
        print('QRNN pooling state shape: ', state.get_shape())
        with tf.variable_scope(scope or "QRNN-f-pooling"):
                # extract Z activations and F gate activations
            Z, F = tf.split(inputs, 2, 1)
            print('QRNN pooling Z shape: ', Z.get_shape())
            print('QRNN pooling F shape: ', F.get_shape())
            # return the dynamic average pooling
            output = tf.multiply(F, state) + tf.multiply(tf.subtract(1., F), Z)
            return output, output


class QRNN_fo_pooling(tf.nn.rnn_cell.RNNCell):
    def __init__(self, out_fmaps):
        self.__out_fmaps = out_fmaps

    @property
    def state_size(self):
        return self.__out_fmaps

    @property
    def output_size(self):
        return self.__out_fmaps

    def __call__(self, inputs, state, scope=None):
        """
        inputs: 2-D tensor of shape [batch_size, Zfeats + [gates]]
        """
        # pool_type = self.__pool_type
        print('QRNN pooling inputs shape: ', inputs.get_shape())
        print('QRNN pooling state shape: ', state.get_shape())
        with tf.variable_scope(scope or "QRNN-f-pooling"):
                # extract Z activations and F gate activations
            Z, F, O = tf.split(inputs, 3, 1)
            print('QRNN pooling Z shape: ', Z.get_shape())
            print('QRNN pooling F shape: ', F.get_shape())
            print('QRNN pooling O shape: ', O.get_shape())
            # return the dynamic average pooling
            new_state = tf.multiply(F, state) + \
                tf.multiply(tf.subtract(1., F), Z)
            output = tf.multiply(O, new_state)
            return output, new_state


def bi_qrnn_f(question_embeddings, question_lengths, hidden_size, keep_prob=1.0):
    filter_width = 2
    in_fmaps = question_embeddings.get_shape().as_list()[-1]
    out_fmaps = hidden_size
    padded_input = tf.pad(question_embeddings, [
        [0, 0], [filter_width - 1, 0], [0, 0]])
    with tf.variable_scope('convolutions'):
        Wz = tf.get_variable('Wz', [filter_width, in_fmaps, out_fmaps],
                             initializer=tf.random_uniform_initializer(minval=-.05, maxval=.05))
        z_a = tf.nn.conv1d(padded_input, Wz, stride=1, padding='VALID')
        Z = tf.nn.tanh(z_a)
        Wf = tf.get_variable('Wf',
                             [filter_width, in_fmaps, out_fmaps],
                             initializer=tf.random_uniform_initializer(minval=-.05, maxval=.05))
        f_a = tf.nn.conv1d(padded_input, Wf, stride=1, padding='VALID')
        F = tf.sigmoid(f_a)
        T = tf.concat([Z, F], 2)
    with tf.variable_scope('pooling'):
        pooling_fw = QRNN_f_pooling(out_fmaps)
        pooling_bw = QRNN_f_pooling(out_fmaps)
        (question_output_fw, question_output_bw), (question_output_final_fw, question_output_final_bw) = tf.nn.bidirectional_dynamic_rnn(
            pooling_fw, pooling_bw, T, sequence_length=question_lengths, dtype=tf.float32)
        question_output = tf.concat(
            [question_output_fw, question_output_bw], 2)

        question_output_final = tf.concat(
            [question_output_final_fw, question_output_final_bw], 1)
    return (question_output, question_output_final)


def bi_qrnn_fo(question_embeddings, question_lengths, hidden_size, keep_prob=1.0):
    filter_width = 2
    in_fmaps = question_embeddings.get_shape().as_list()[-1]
    out_fmaps = hidden_size
    padded_input = tf.pad(question_embeddings, [
        [0, 0], [filter_width - 1, 0], [0, 0]])
    with tf.variable_scope('convolutions'):
        Wz = tf.get_variable('Wz', [filter_width, in_fmaps, out_fmaps],
                             initializer=tf.random_uniform_initializer(minval=-.05, maxval=.05))
        z_a = tf.nn.conv1d(padded_input, Wz, stride=1, padding='VALID')
        Z = tf.nn.tanh(z_a)
        Wf = tf.get_variable('Wf',
                             [filter_width, in_fmaps, out_fmaps],
                             initializer=tf.random_uniform_initializer(minval=-.05, maxval=.05))
        f_a = tf.nn.conv1d(padded_input, Wf, stride=1, padding='VALID')
        F = tf.sigmoid(f_a)
        # zoneout to implement here
        Wo = tf.get_variable('Wo',
                             [filter_width, in_fmaps, out_fmaps],
                             initializer=tf.random_uniform_initializer(minval=-.05, maxval=.05))
        f_o = tf.nn.conv1d(padded_input, Wo, stride=1, padding='VALID')
        O = tf.sigmoid(f_o)
        T = tf.concat([Z, F, O], 2)
    with tf.variable_scope('pooling'):
        pooling_fw = QRNN_fo_pooling(out_fmaps)
        pooling_bw = QRNN_fo_pooling(out_fmaps)
        (question_output_fw, question_output_bw), (question_output_final_fw, question_output_final_bw) = tf.nn.bidirectional_dynamic_rnn(
            pooling_fw, pooling_bw, T, sequence_length=question_lengths, dtype=tf.float32)
        question_output = tf.concat(
            [question_output_fw, question_output_bw], 2)

        question_output_final = tf.concat(
            [question_output_final_fw, question_output_final_bw], 1)
    return (question_output, question_output_final)


class Baseline(object):
    def __init__(self, train_dataset, val_dataset, embedding, vocabulary, batch_size=128):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.embedding = embedding
        self.batch_size = batch_size
        self.lr = 0.005
        self.gstep = tf.Variable(0, dtype=tf.int32,
                                 trainable=False, name='global_step')
        self.lstm_hidden_size = 100
        self.vocabulary = vocabulary
        self.handle = tf.placeholder(tf.string, shape=[])
        self.keep_prob = tf.placeholder(tf.float32, shape=[])
        self.train_max_context_length = 744
        self.train_max_question_length = 60

    def encoder(self, embeddings, lengths, hidden_size, keep_prob=1.0):
        return bilstm(embeddings, lengths, hidden_size, keep_prob)

    def pred(self):
        with tf.variable_scope("embedding_layer"):
            (self.questions, question_lengths), (self.contexts,
                                                 context_lengths), self.answers = self.iterator.get_next()

            max_context_length = tf.reduce_max(context_lengths)
            max_question_length = tf.reduce_max(question_lengths)

            #max_context_length = self.train_max_context_length
            #max_question_length = self.train_max_question_length

            context_mask = tf.sequence_mask(
                context_lengths, maxlen=max_context_length)

            question_mask = tf.sequence_mask(
                question_lengths, maxlen=max_question_length)

            question_embeddings = tf.nn.embedding_lookup(
                self.embedding, self.questions)
            context_embeddings = tf.nn.embedding_lookup(
                self.embedding, self.contexts)

        with tf.variable_scope("question_embedding_layer"):
            question_output, _ = self.encoder(
                question_embeddings, question_lengths, self.lstm_hidden_size, keep_prob=self.keep_prob)

        with tf.variable_scope("context_embedding_layer"):
            context_output, _ = self.encoder(
                context_embeddings, context_lengths, self.lstm_hidden_size, self.keep_prob)

            d = context_output.get_shape().as_list()[-1]

            # context_output dimension is BS * max_context_length * d
            # where d = 2*lstm_hidden_size

        with tf.variable_scope("attention_layer"):
            # d is equal to 2*self.lstm_hidden_size

            question_tiled = tf.tile(tf.expand_dims(
                question_output, 1), [1, max_context_length, 1, 1])
            print('question_tiled', question_tiled.get_shape().as_list())
            context_tiled = tf.tile(tf.expand_dims(
                context_output, 2), [1, 1, max_question_length, 1])
            print('context_tiled', context_tiled.get_shape().as_list())
            product_tiled = tf.reshape(tf.reshape(question_tiled, [-1, d]) * tf.reshape(
                context_tiled, [-1, d]), [-1, max_context_length, max_question_length, d])
            print('product_tiled', product_tiled.get_shape().as_list())

            similarity_matrix = tf.reduce_sum(product_tiled, axis=3)
            print('similarity_matrix', similarity_matrix.get_shape().as_list())

            context_mask_aug = tf.tile(tf.expand_dims(context_mask, 2), [
                                       1, 1, max_question_length])
            question_mask_aug = tf.tile(tf.expand_dims(
                question_mask, 1), [1, max_context_length, 1])

            mask_aug = context_mask_aug & question_mask_aug

            similarity_matrix = preprocess_softmax(
                similarity_matrix, mask_aug)
            print('similarity_matrix', similarity_matrix.get_shape().as_list())

            context_to_query_attention_weights = tf.nn.softmax(
                similarity_matrix, axis=2)
            print('context_to_query_attention_weights',
                  context_to_query_attention_weights.get_shape().as_list())

            context_to_query = tf.matmul(
                context_to_query_attention_weights, question_output)
            print('context_to_query', context_to_query.get_shape().as_list())

            attention = tf.concat([context_output, context_to_query], axis=2)
            print('attention', attention.get_shape().as_list())

        with tf.variable_scope("modeling_layer"):
            m1, _ = self.encoder(attention, context_lengths,
                                 self.lstm_hidden_size, self.keep_prob)
            print('m1', m1.get_shape().as_list())

        with tf.variable_scope("output_layer_start"):
            W1 = tf.get_variable("W1", initializer=tf.contrib.layers.xavier_initializer(
            ), shape=(d, 1), dtype=tf.float32)
            print('W1',
                  W1.get_shape().as_list())
            pred_start = tf.matmul(tf.reshape(
                m1, shape=[-1, d]), W1)
            print('pred_start',
                  pred_start.get_shape().as_list())
            pred_start = tf.reshape(
                pred_start, shape=[-1, max_context_length])
            print('pred_start',
                  pred_start.get_shape().as_list())
            self.pred_start = preprocess_softmax(pred_start, context_mask)
            print('self.pred_start',
                  self.pred_start.get_shape().as_list())

        with tf.variable_scope("output_layer_end"):
            W2 = tf.get_variable("W2", initializer=tf.contrib.layers.xavier_initializer(
            ), shape=(d, 1), dtype=tf.float32)
            pred_end = tf.matmul(tf.reshape(
                m1, shape=[-1, d]), W2)
            pred_end = tf.reshape(
                pred_end, shape=[-1, max_context_length])
            self.pred_end = preprocess_softmax(pred_end, context_mask)

            self.preds = tf.transpose(
                [tf.argmax(self.pred_start, axis=1), tf.argmax(self.pred_end, axis=1)])

    def loss(self):
        with tf.variable_scope("loss"):
            loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.pred_start, labels=self.answers[:, 0])
            loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.pred_end, labels=self.answers[:, 1])

            self.total_loss = tf.reduce_mean(
                loss_start) + tf.reduce_mean(loss_end)

    def optimize(self):
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.total_loss,
                                                                          global_step=self.gstep)

    def build(self):
        self.get_data()
        self.pred()
        self.loss()
        self.optimize()

    def get_data(self):
        padded_shapes = ((tf.TensorShape([None]),  # question of unknown size
                          tf.TensorShape([])),  # size(question)
                         (tf.TensorShape([None]),  # context of unknown size
                          tf.TensorShape([])),  # size(context)
                         tf.TensorShape([2]))

        padding_values = ((0, 0), (0, 0), 0)
        train_batch = self.train_dataset.padded_batch(
            self.batch_size, padded_shapes=padded_shapes, padding_values=padding_values)

        # train_evaluation = self.train_dataset.

        train_eval_batch = self.train_dataset.shuffle(10000).padded_batch(
            self.batch_size, padded_shapes=padded_shapes, padding_values=padding_values)

        val_batch = self.val_dataset.shuffle(10000).padded_batch(
            500, padded_shapes=padded_shapes, padding_values=padding_values).prefetch(1)

        # Create a one shot iterator over the zipped dataset
        self.train_iterator = train_batch.make_initializable_iterator()
        self.val_iterator = val_batch.make_initializable_iterator()
        self.train_eval_iterator = train_eval_batch.make_initializable_iterator()

        # self.iterator = train_batch.make_initializable_iterator()
        self.iterator = tf.data.Iterator.from_string_handle(
            self.handle, self.train_iterator.output_types, self.train_iterator.output_shapes)

    def train(self, n_iters):
        eval_step = 10

        with tf.Session() as sess:

            self.train_iterator_handle = sess.run(
                self.train_iterator.string_handle())
            self.val_iterator_handle = sess.run(
                self.val_iterator.string_handle())
            self.train_eval_iterator_handle = sess.run(
                self.train_eval_iterator.string_handle())

            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(
                'graphs/attention1', sess.graph)
            initial_step = self.gstep.eval()
            sess.run(self.val_iterator.initializer)
            sess.run(self.train_eval_iterator.initializer)

            variables = tf.trainable_variables()
            num_vars = np.sum([np.prod(v.get_shape().as_list())
                               for v in variables])

            print("Number of variables in models: {}".format(num_vars))
            for epoch in range(n_iters):
                print("epoch #", epoch)
                num_batches = int(67978.0 / self.batch_size)
                progress = Progbar(target=num_batches)
                sess.run(self.train_iterator.initializer)
                index = 0
                total_loss = 0
                progress.update(index, [("training loss", total_loss)])
                while True:
                    index += 1
                    try:
                        total_loss, opt = sess.run(
                            [self.total_loss, self.opt], feed_dict={self.handle: self.train_iterator_handle, self.keep_prob: 0.75})  # , options=options, run_metadata=run_metadata)

                        progress.update(index, [("training loss", total_loss)])

                    except tf.errors.OutOfRangeError:
                        break
                print(
                    'evaluation on 500 training elements:')
                preds, contexts, answers = sess.run([self.preds, self.contexts, self.answers], feed_dict={
                    self.handle: self.train_eval_iterator_handle, self.keep_prob: 1.0})
                predictions = []
                ground_truths = []
                for i in range(len(preds)):
                    predictions.append(convert_indices_to_text(
                        self.vocabulary, contexts[i], preds[i, 0], preds[i, 1]))
                    ground_truths.append(convert_indices_to_text(
                        self.vocabulary, contexts[i], answers[i, 0], answers[i, 1]))
                print(evaluate(predictions, ground_truths))
                print(
                    'evaluation on 500 validation elements:')
                preds, contexts, answers = sess.run([self.preds, self.contexts, self.answers], feed_dict={
                    self.handle: self.val_iterator_handle, self.keep_prob: 1.0})
                predictions = []
                ground_truths = []
                for i in range(len(preds)):
                    predictions.append(convert_indices_to_text(
                        self.vocabulary, contexts[i], preds[i, 0], preds[i, 1]))
                    ground_truths.append(convert_indices_to_text(
                        self.vocabulary, contexts[i], answers[i, 0], answers[i, 1]))
                print(evaluate(predictions, ground_truths))
                predictions = []
                ground_truths = []
            writer.close()


if __name__ == '__main__':
    print("ok")

    embedding = load_word_embeddings()

    vocabulary = load_vocabulary()

    # with tf.Session() as sess:
    #    z = sess.run([y])
    #    print('embedding', y.get_shape(), z)

    # print("shapes", embedding.shape)
    train_questions = with_length(create_dataset("train.ids.question"))
    train_answers = create_dataset("train.span")
    train_context = with_length(create_dataset("train.ids.context"))

    train_dataset = tf.data.Dataset.zip(
        (train_questions, train_context, train_answers))

    val_questions = with_length(create_dataset("val.ids.question"))
    val_answers = create_dataset("val.span")
    val_context = with_length(create_dataset("val.ids.context"))

    val_dataset = tf.data.Dataset.zip(
        (val_questions, val_context, val_answers))

    # with tf.Session() as sess:
    # sess.run(iterator.initializer)
    # x = iterator.get_next()
    # a = sess.run([x])
    # print(x.output_shapes, a)

    machine = Baseline(train_dataset, val_dataset,
                       embedding, vocabulary, batch_size=128)
    machine.build()
    machine.train(10)
