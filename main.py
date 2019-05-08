"""
main.py
"""
import tensorflow as tf

# from tasks.vqa.env.generate_data import generate_addition
from tasks.vqa.eval import evaluate_vqa
from tasks.vqa.train import train_vqa
from tasks.vqa.test_direct import test_vqa



FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("task", "vqa", "Which NPI Task to run - [vqa].")

tf.app.flags.DEFINE_boolean("do_train", False, "Boolean whether to continue training model.")
tf.app.flags.DEFINE_boolean("do_test", True , "Boolean whether to continue training model.")
tf.app.flags.DEFINE_boolean("do_eval", False, "Boolean whether to perform model evaluation.")

tf.app.flags.DEFINE_integer("num_epochs", 5, "Number of training epochs to perform.")


def main(_):
    if FLAGS.task == "vqa":

        # Train Model (if necessary)
        if FLAGS.do_train:
            train_vqa(FLAGS.num_epochs)

        if FLAGS.do_test:
            test_vqa()

        # Evaluate Model
        if FLAGS.do_eval:
            evaluate_vqa()


if __name__ == "__main__":
    tf.app.run()
