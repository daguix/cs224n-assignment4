import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

from os.path import join as pjoin
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

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
        self.batch_max_context_length = tf.Variable(0, dtype=tf.int32)
        self.handle = tf.placeholder(tf.string, shape=[])

    def pred(self):
        with tf.variable_scope("lstm"):
            (self.questions, question_lengths), (self.contexts,
                                                 context_lengths), self.answers = self.iterator.get_next()

            self.max_context_length = tf.reduce_max(context_lengths)
            self.max_question_length = tf.reduce_max(question_lengths)

            context_mask = tf.sequence_mask(
                context_lengths, maxlen=self.max_context_length)

            question_embeddings = tf.nn.embedding_lookup(
                self.embedding, self.questions)
            context_embeddings = tf.nn.embedding_lookup(
                self.embedding, self.contexts)

            normalized_question_embeddings = tf.nn.l2_normalize(
                question_embeddings, axis=2)
            normalized_context_embeddings = tf.nn.l2_normalize(
                context_embeddings, axis=2)  #

            # (BS, MCL, EMBED_DIM) @ (BS, EMBED_DIM, MQL) -> (BS ,MCL, MQL)

            similarity = tf.matmul(normalized_context_embeddings, tf.transpose(
                normalized_question_embeddings, [0, 2, 1]))

            similarity_max = tf.reduce_max(similarity, axis=2)

            similarity_max_reshaped = tf.reshape(
                similarity_max, [-1, self.max_context_length, 1])

            context_embeddings = similarity_max_reshaped * context_embeddings

            lstm_cell_fw = tf.nn.rnn_cell.GRUCell(
                self.lstm_hidden_size, name="gru_cell_fw")
            lstm_cell_bw = tf.nn.rnn_cell.GRUCell(
                self.lstm_hidden_size, name="gru_cell_bw")

            question_output, (question_output_final_fw, question_output_final_bw) = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw, question_embeddings, sequence_length=question_lengths, dtype=tf.float32, time_major=False)

            (context_output_fw, context_output_bw), context_output_final = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw, context_embeddings, sequence_length=context_lengths,
                dtype=tf.float32, time_major=False, initial_state_fw=question_output_final_fw,
                initial_state_bw=question_output_final_bw)

            context_output = tf.concat(
                [context_output_fw, context_output_bw], 2)

            # aligned_question_output_final_start = tf.layers.dense(
            #    question_output_final_reshaped, 2*self.lstm_hidden_size, activation=tf.nn.tanh)

            context_inverse_mask = tf.subtract(
                tf.constant(1.0), tf.cast(context_mask, tf.float32))
            penalty_context_value = tf.multiply(
                context_inverse_mask, tf.constant(-1e9))

            d = context_output.get_shape().as_list()[2]
            context = tf.reshape(context_output, shape=[-1, d])
            W1 = tf.get_variable("W1", initializer=tf.contrib.layers.xavier_initializer(
            ), shape=(d, 1), dtype=tf.float32)
            pred_start = tf.matmul(context, W1)
            pred_start = tf.reshape(
                pred_start, shape=[-1, self.max_context_length])
            self.pred_start = tf.where(
                context_mask, pred_start, penalty_context_value)
            W2 = tf.get_variable("W2", initializer=tf.contrib.layers.xavier_initializer(
            ), shape=(d, 1), dtype=tf.float32)
            pred_end = tf.matmul(context, W2)
            pred_end = tf.reshape(
                pred_end, shape=[-1, self.max_context_length])
            self.pred_end = tf.where(
                context_mask, pred_end, penalty_context_value)

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
                         (tf.TensorShape([None]),  # context of self.max_context_length size
                          tf.TensorShape([])),  # size(context)
                         tf.TensorShape([2]))

        padding_values = ((0, 0), (0, 0), 0)
        train_batch = self.train_dataset.padded_batch(
            self.batch_size, padded_shapes=padded_shapes, padding_values=padding_values)

        # train_evaluation = self.train_dataset.

        val_batch = self.val_dataset.shuffle(10000).padded_batch(
            self.batch_size, padded_shapes=padded_shapes, padding_values=padding_values).prefetch(1)

        # Create a one shot iterator over the zipped dataset
        self.train_iterator = train_batch.make_initializable_iterator()
        self.val_iterator = val_batch.make_initializable_iterator()

        #self.iterator = train_batch.make_initializable_iterator()
        self.iterator = tf.data.Iterator.from_string_handle(
            self.handle, self.train_iterator.output_types, self.train_iterator.output_shapes)

    def train(self, n_iters):
        skip_step = 1

        with tf.Session() as sess:

            ###############################
            # TO DO:
            # 1. initialize your variables
            # 2. create writer to write your graph
            ###############################

            self.train_iterator_handle = sess.run(
                self.train_iterator.string_handle())
            self.val_iterator_handle = sess.run(
                self.val_iterator.string_handle())

            sess.run(tf.global_variables_initializer())
            # writer = tf.summary.FileWriter(
            #    'graphs/baseline', sess.graph)

            initial_step = self.gstep.eval()
            index = 0
            for epoch in range(n_iters):
                print("epoch #", epoch)
                sess.run(self.train_iterator.initializer)

                start_time = time.time()

                # print('test for cosine similarity', sess.run([self.question_embeddings, self.context_embeddings, self.normalized_question_embeddings,
                #                                              self.normalized_context_embeddings, self.similarity, self.similarity_max], feed_dict={self.handle: self.train_iterator_handle}))

                while True:
                    index += 1
                    if index > 5 and index <= 100:
                        skip_step = 10
                    elif index > 100:
                        skip_step = 20
                    try:
                        # options = tf.RunOptions(
                        #    trace_level=tf.RunOptions.FULL_TRACE)
                        #run_metadata = tf.RunMetadata()
                        total_loss, opt, preds, contexts, answers = sess.run(
                            [self.total_loss, self.opt, self.preds, self.contexts, self.answers], feed_dict={self.handle: self.train_iterator_handle})  # , options=options, run_metadata=run_metadata)
                        # preds, contexts, answers, total_loss, opt = sess.run(
                        #    [self.preds, self.contexts, self.answers, self.total_loss, self.opt])

                        # print("batch_max_context_length",
                        #      batch_max_context_length, self.max_context_length)

                        # fetched_timeline = timeline.Timeline(
                        #    run_metadata.step_stats)
                        #chrome_trace = fetched_timeline.generate_chrome_trace_format()
                        # with open('./profiling/timeline_'+str(index)+'.json', 'w') as f:
                        #    f.write(chrome_trace)

                        if index % skip_step == 0:

                            print('Batch {}'.format(
                                index))
                            print('   Loss:', total_loss)
                            print('   Took: {} seconds'.format(
                                time.time() - start_time))
                            start_time = time.time()

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

                            step = 0
                            if (index + 1) % 20 == 0:
                                step += 1
                                ###############################
                                # TO DO: save the variables into a checkpoint
                                ###############################
                    except tf.errors.OutOfRangeError:
                        break

            # writer.close()


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
                       embedding, vocabulary, batch_size=256)
    machine.build()
    machine.train(10)
