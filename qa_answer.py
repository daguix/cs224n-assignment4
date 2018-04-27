from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import json
import sys
import random
import time
from datetime import datetime
from os.path import join as pjoin

from tqdm import tqdm
import numpy as np
from six.moves import xrange
import tensorflow as tf

from qa_model import Encoder, QASystem, Decoder
from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url, \
    invert_map, tokenize, token_idx_map
import qa_data

import logging

logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 10, "Batch size to use during training.")
# tf.app.flags.DEFINE_integer("epochs", 0, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{vocab_dim}.npz)")
tf.app.flags.DEFINE_string("dev_path", "data/squad/dev-v1.1.json", "Path to the JSON dev set to evaluate against (default: ./data/squad/dev-v1.1.json)")
tf.app.flags.DEFINE_integer("vocab_dim", 100, "GLoVe embedding dimension")

def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def read_dataset(dataset, tier, vocab):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""

    context_data = []
    query_data = []
    question_uuid_data = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = tokenize(question)
                question_uuid = qas[qid]['id']

                context_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in context_tokens]
                qustion_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in question_tokens]

                context_data.append(' '.join(context_ids))
                query_data.append(' '.join(qustion_ids))
                question_uuid_data.append(question_uuid)

    return context_data, query_data, question_uuid_data


def prepare_dev(prefix, dev_filename, vocab):
    print("Downloading {}".format(dev_filename))
    # Don't check file size, since we could be using other datasets
    dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)

    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    context_data, question_data, question_uuid_data = read_dataset(dev_data, 'dev', vocab)

    return context_data, question_data, question_uuid_data


def generate_answers(session, model, dataset, rev_vocab):
    """
    Loop over the dev or test dataset and generate answer.

    Note: output format must be answers[uuid] = "real answer"
    You must provide a string of words instead of just a list, or start and end index

    In main() function we are dumping onto a JSON file

    evaluate.py will take the output JSON along with the original JSON file
    and output a F1 and EM

    You must implement this function in order to submit to Leaderboard.

    :param sess: active TF session
    :param model: a built QASystem model
    :param rev_vocab: this is a list of vocabulary that maps index to actual words
    :return:
    """
    c_ids, c_len, q_ids, q_len, uid = dataset
    print ("c_ids len=", len(c_ids))
    answers = {}
    for index in range(len(c_ids)):
        a_s, a_e = model.answer(session, (c_ids[index], c_len[index], q_ids[index], q_len[index]) )
        answer_span = c_ids[index][a_s: a_e+1]

        c_words = []
        for i in answer_span:
            c_words.append(rev_vocab[i])
        #print (uid[index], type(uid[index]))
        answers[uid[index]] = " ".join(c_words)
    return answers

def load_data(context, question, uid):

    tic = time.time()
    logging.info("Loading SQUAD data")

    # c_file = open(c_path, mode="rb")
    # q_file = open(q_path, mode="rb")
    # s_file = open(s_path, mode="rb")
    assert len(context)==len(question)
    data = []

    line = 0
    counter = 0
    for i in range(len(context)):
        line += 1

        c_ids = map(int, context[i].lstrip().rstrip().split(" "))
        q_ids = map(int, question[i].lstrip().rstrip().split(" "))

        data.append((c_ids, q_ids, uid[i]))

    max_c = FLAGS.max_c
    max_q = FLAGS.max_q

    data.sort(key=lambda tup: len(tup[0])*100+len(tup[1])) # Sort by context len then by question len

    stop = next(( idx for idx, xi in enumerate(data) if len(xi[0])>max_c), len(data))
    #assert len(data[stop-1][0])<=max_c

    c_ids = np.array([xi[0]+[0]*(max_c-len(xi[0])) for xi in data[:stop]], dtype=np.int32)
    q_ids = np.array([xi[1]+[0]*(max_q-len(xi[1])) for xi in data[:stop]], dtype=np.int32)
    c_len  = np.array([len(xi[0]) for xi in data[:stop]], dtype=np.int32)
    q_len  = np.array([len(xi[1]) for xi in data[:stop]], dtype=np.int32)

    uid = np.array([xi[2] for xi in data[:stop]])

    data_size = c_ids.shape[0]

    assert q_ids.shape[0] == data_size
    assert  q_len.shape == (data_size,)
    assert  c_len.shape == (data_size,)
    samples = len(data)

    toc = time.time()
    logging.info("Complete: %d samples loaded in %f secs)" % (samples, toc - tic))
    logging.info("Median context length: %d" % len(data[counter//2][0]))
    logging.info("Len data == %d, Context == %d, C_IDS Length = %d, stop = %d" % (len(data), len(context), len(c_ids), stop))   

    return [c_ids, c_len, q_ids, q_len, uid]

def main(_):

    vocab, rev_vocab = initialize_vocab(FLAGS.vocab_path)

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))

    global_train_dir = '/tmp/cs224n-squad-train'
    # Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    # file paths saved in the checkpoint. This allows the model to be reloaded even
    # if the location of the checkpoint files has moved, allowing usage with CodaLab.
    # This must be done on both train.py and qa_answer.py in order to work.
    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    os.symlink(os.path.abspath(FLAGS.train_dir), global_train_dir)
    train_dir = global_train_dir

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    # ========= Load Dataset =========
    # You can change this code to load dataset in your own way

    dev_dirname = os.path.dirname(os.path.abspath(FLAGS.dev_path))
    dev_filename = os.path.basename(FLAGS.dev_path)

    context_data, question_data, question_uuid_data = prepare_dev(dev_dirname, dev_filename, vocab)
    dataset = load_data(context_data, question_data, question_uuid_data)

    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    # ========= Model-specific =========
    # You must change the following code to adjust to your model
    embed_path = FLAGS.embed_path or "data/squad/glove.trimmed.{}.npz".format(FLAGS.embedding_size)
    embeddingz = np.load(embed_path)
    embeddings = embeddingz['glove']
    embeddingz.close()

    assert embeddings.shape[1] == FLAGS.embedding_size

    qa = QASystem(train_dir, embeddings)

    with tf.Session() as sess:
        initialize_model(sess, qa, train_dir)
        answers = generate_answers(sess, qa, dataset, rev_vocab)

        # write to json file to root dir
        with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:
            f.write(unicode(json.dumps(answers, ensure_ascii=False)))


if __name__ == "__main__":
  tf.app.run()
