"""
Simple char-rnn based on

    https://github.com/sherjilozair/char-rnn-tensorflow

Original article:

    http://karpathy.github.io/2015/05/21/rnn-effectiveness/

"""
import codecs
import os
import pickle
from collections import Counter
import numpy as np

class Reader(object):
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir   = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding   = encoding

        input_file  = os.path.join(data_dir, 'input.txt')
        vocab_file  = os.path.join(data_dir, 'vocab.pkl')
        tensor_file = os.path.join(data_dir, 'data.npy')
        if not os.path.exists(vocab_file) or not os.path.exists(tensor_file):
            print("Reading text file.")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("Loading preprocessed files.")
            self.load_preprocessed(vocab_file, tensor_file)

        self.create_batches()

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, 'r', encoding=self.encoding) as f:
            data = f.read()

        counter = Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: x[1])[::-1]
        self.chars, _ = zip(*count_pairs)
        self.vocab = {c: i for i, c in enumerate(self.chars)}
        self.tensor = np.array(list(map(self.vocab.get, data)))

        with open(vocab_file, 'wb') as f:
            pickle.dump(self.chars, f)
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = pickle.load(f)
        self.vocab  = {c: i for i, c in enumerate(self.chars)}
        self.tensor = np.load(tensor_file)

    def create_batches(self):
        chars_per_batch  = self.batch_size * self.seq_length
        self.num_batches = self.tensor.size // chars_per_batch
        assert self.num_batches > 0

        n = self.num_batches * chars_per_batch
        x_data = self.tensor[:n]
        if n < self.tensor.size:
            y_data = self.tensor[1:n+1]
        else:
            y_data = np.concatenate((x_data[1:], [x_data[0]]))
        self.x_batches = np.split(x_data.reshape((self.batch_size, -1)),
                                  self.num_batches, 1)
        self.y_batches = np.split(y_data.reshape((self.batch_size, -1)),
                                  self.num_batches, 1)
        self.pointer = 0

    def next_batch(self):
        batch = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batches
        return batch

#///////////////////////////////////////////////////////////////////////////////

if __name__ == '__main__':
    data_dir   = 'datasets/tinyshakespeare'
    batch_size = 2
    seq_length = 4
    data = Reader(data_dir, batch_size, seq_length)
    x, y = data.next_batch()

    print("")
    print("Example batch from char-rnn")
    print("---------------------------")
    for i in range(x.shape[0]):
        print("{} -> {}".format(x[i], y[i]))
