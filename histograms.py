from tensorflow.python.platform import gfile
from os.path import join as pjoin
from matplotlib import pyplot


def counter(data_path):
    print("Counting data in %s" % data_path)
    with gfile.GFile(pjoin("./data/squad/", data_path), mode="rb") as data_file:
        word_lengths = []
        for line in data_file:
            word_lengths.append(len(line.strip().split()))
    max_list = max(word_lengths)
    print("max word length", max_list)
    pyplot.hist(word_lengths, range=(0, max_list+1), bins=max_list+1)
    pyplot.show()


if __name__ == '__main__':
    counter("train.context")
