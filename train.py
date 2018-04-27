from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime
import json
import time
import random

import tensorflow as tf
import numpy as np
import argparse
from tqdm import tqdm
from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_integer("max_train_samples", 0, "Max number of training samples.")
tf.app.flags.DEFINE_integer("max_val_samples", 100, "Max number of validation samples (0--load all).")
tf.app.flags.DEFINE_integer("embedding_size", 300, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{vocab_dim}.npz)")
tf.app.flags.DEFINE_boolean("restart", False, "Do not attempt to restore the model.")
tf.app.flags.DEFINE_string("restore_path", "", "Directory Path to load model weights if Restart = TRUE. (default: /tmp/cs224n-squad-train)")
tf.app.flags.DEFINE_integer("verify_only", 0, "Print N random samples and exit.")
tf.app.flags.DEFINE_boolean("check_embeddings", False, "Check embedding ids for our of bound conditions")

FLAGS = tf.app.flags.FLAGS


def initialize_model(session, model, train_dir):
    if FLAGS.restore_path and FLAGS.restart:
        ckpt = tf.train.get_checkpoint_state(FLAGS.restore_path)
    else:
        ckpt = tf.train.get_checkpoint_state(train_dir)

    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""''
    logging.info("Restart Flag = %s" % (FLAGS.restart))
    saver = tf.train.Saver()
    if FLAGS.restart and ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path)
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

def load_squad(data_path, prefix, max_vocab, max_samples=0):
    prefix_path = pjoin(FLAGS.data_dir, prefix)

    c_path  = prefix_path+".ids.context"
    if not tf.gfile.Exists(c_path):
        raise ValueError("Context file %s not found.", c_path)

    q_path = prefix_path+".ids.question"
    if not tf.gfile.Exists(q_path):
        raise ValueError("Question file %s not found.", q_path)

    s_path = prefix_path+".span"
    if not tf.gfile.Exists(s_path):
        raise ValueError("Span file %s not found.", s_path)

    tic = time.time()
    logging.info("Loading SQUAD data from %s" % prefix_path)

    c_file = open(c_path, mode="rb")
    q_file = open(q_path, mode="rb")
    s_file = open(s_path, mode="rb")

    valid_range = range(0, max_vocab)

    data = []

    max_c  = 0
    max_q = 0

    c_buckets = [0]*10
    q_buckets = [0]*10

    line = 0
    counter = 0
    for c, q, s in tqdm(zip(c_file, q_file, s_file)):
        line += 1

        c_ids = map(int, c.lstrip().rstrip().split(" "))
        q_ids = map(int, q.lstrip().rstrip().split(" "))
        span  = map(int, s.lstrip().rstrip().split(" "))

        if not (len(span) == 2 and span[0] <= span[1] and span[1] < len(c_ids)):
            #print( "Invalid span at line {}. {} <= {} < {}".format(line, span[0], span[1], len(c_ids)))
            continue

        if max_vocab and not (all( id in valid_range for id in c_ids ) and all( id in valid_range for id in q_ids )):
            print( "Vocab id is out of bound")
            continue

        data.append((c_ids, q_ids, [span[0], span[1]]))

        c_buckets[ len(c_ids)//100] += 1
        q_buckets[ len(q_ids)//10] += 1

        max_c = max(max_c, len(c_ids))
        max_q = max(max_q, len(q_ids))

        if max_samples and len(data) >= max_samples:
            break

    samples = len(data)

    assert sum(c_buckets) == samples
    assert sum(q_buckets) == samples

    data.sort(key=lambda tup: len(tup[0])*100+len(tup[1])) # Sort by context len then by question len

    toc = time.time()
    logging.info("Complete: %d samples loaded in %f secs)" % (samples, toc - tic))
    logging.info("Question length histogram (10 in each bucket): %s" % str(c_buckets));
    logging.info("Context length histogram (100 in each bucket): %s" % str(q_buckets));
    logging.info("Median context length: %d" % len(data[counter//2][0]));

    return data

def print_sample(sample, rev_vocab):
    print("Context:")
    print(" ".join([rev_vocab[s] for s in sample[0]]))
    print("Question:")
    print(" ".join([rev_vocab[s] for s in sample[1]]))
    print("Answer:")
    print(" ".join([rev_vocab[s] for s in sample[0][sample[2][0]:sample[2][1]+1]]))

def print_samples(data, n, rev_vocab):
    all_samples = range(len(data))
    for ix in random.sample( all_samples, n) if n > 0 else all_samples:
        print_sample(data[ix], rev_vocab)

def main(args):
    if args:
        restore = args

    embed_path = FLAGS.embed_path or "data/squad/glove.trimmed.{}.npz".format(FLAGS.embedding_size)
    embeddingz = np.load(embed_path)
    embeddings = embeddingz['glove']
    embeddingz.close()
    assert embeddings.shape[1] == FLAGS.embedding_size

    vocab_len = embeddings.shape[0]

    train = load_squad(FLAGS.data_dir, "train", max_vocab=vocab_len if FLAGS.check_embeddings else 0, max_samples=FLAGS.max_train_samples)
    val   = load_squad(FLAGS.data_dir, "val",   max_vocab=vocab_len if FLAGS.check_embeddings else 0, max_samples=FLAGS.max_val_samples)

    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    if FLAGS.verify_only:
	print_samples(train, FLAGS.verify_only, rev_vocab)

        return

    global_train_dir = '/tmp/cs224n-squad-train'
    # Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    # file paths saved in the checkpoint. This allows the model to be reloaded even
    # if the location of the checkpoint files has moved, allowing usage with CodaLab.
    # This must be done on both train.py and qa_answer.py in order to work.
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    os.symlink(os.path.abspath(FLAGS.train_dir), global_train_dir)
    train_dir = global_train_dir

    qa = QASystem(train_dir, embeddings)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        initialize_model(sess, qa, train_dir)

        qa.train(sess, train)

        qa.evaluate_answer(sess, qa.preprocess_sequence_data(val), log=True)

if __name__ == "__main__":
    tf.app.run()
