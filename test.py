from __future__ import print_function
import tensorflow as tf

import argparse
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

from six import text_type

def main():
    parser = argparse.ArgumentParser(
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                        help='data directory containing input.txt')
    args = parser.parse_args()
    test(args)

def test(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    saved_args.batch_size = 1
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)

    data_loader = TextLoader(saved_args.data_dir, saved_args.batch_size, saved_args.seq_length, train_split=1.0)


    model = Model(saved_args)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        data_loader.reset_batch_pointer()
        state = sess.run(model.initial_state)
        for b in range(data_loader.num_batches):
            x, y = data_loader.next_batch()
            feed = {model.input_data: x, model.targets: y}
            for i, (c, h) in enumerate(model.initial_state):
                feed[c] = state[i].c
                feed[h] = state[i].h
            train_loss, probs, state = sess.run([model.cost, model.probs, model.final_state], feed)
            print(probs.shape)

if __name__ == '__main__':
    main()
