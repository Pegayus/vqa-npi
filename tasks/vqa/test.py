"""
train.py

Core training script for the VQA task-specific NPI. Instantiates a model, then trains using
the precomputed data.
"""
import numpy as np
from model.npi import NPI
from tasks.vqa.vqa import VQAcore
from tasks.vqa.env.config import CONFIG, get_args, Scene
import pickle
import tensorflow as tf
import statistics as stat
# import json

EX_PROG_PID = [2,5,3]
PAR_PROG_PID = [1,2,4,7]
# MOVE_PTR_PID, UNIQUE_PID, DELETE_ROW_PID = EX_PROG_PID
# ROW_PTR, POSITION_PTR, COLOR_PTR, MATERIAL_PTR, SHAPE_PTR, SIZE_PTR = range(6)
# DOWN, RESET = 0, 1

DATA_PATH = "tasks/vqa/data/train.pickle"
LOG_PATH = "tasks/vqa/log/"
CKPT_PATH = "tasks/vqa/log/model.ckpt"


def test_vqa(test, npi, core, sess, verbose=0):
    """
    Instantiates a VQA Core, NPI, then loads and fits model to data.

    :param epochs: Number of epochs to train for.
    """

    # Load Data
    # with open(DATA_PATH, 'rb') as f:
    #     data = pickle.load(f)
    #     data = data[1:2]
    data = test
    # Initialize VQA Core
    # print ('Initializing VQA Core!')
    # core = VQAcore()
    #
    # # Initialize NPI Model
    # print ('Initializing NPI Model!')
    # npi = NPI(core, CONFIG, LOG_PATH, verbose=verbose)

    # with tf.Session() as sess:
        # Restore from Checkpoint
    saver = tf.train.Saver()
    saver.restore(sess, CKPT_PATH)


    term_acct = []
    prog_acct = []
    arg_acct = []
    # Start Testing
    for i in range(len(data)):
        # Reset NPI States
        # npi.reset_state()

        # Setup Environment
        _,imgid, qid, qtype, steps = data[i]
        scene = Scene(imgid)
        x, y = steps[:-1], steps[1:]
        if len(x) == 0 or len(y) == 0:
            continue

        # Run through steps, and fit!
        step_def_loss, step_arg_loss, term_acc, prog_acc, = 0.0, 0.0, 0.0, 0.0
        arg0_acc, arg1_acc, arg2_acc, num_args = 0.0, 0.0, 0.0, 0
        for j in range(len(x)):
            (prog_name, prog_in_id), arg, term = x[j]
            (_, prog_out_id), arg_out, term_out = y[j]

            # Update Environment if MOVE or WRITE
            if prog_in_id in EX_PROG_PID:
                scene.execute(prog_in_id, arg)

            # Get Environment, Argument Vectors
            env_in = [scene.get_env()]
            arg_in, arg_out = [get_args(arg, arg_in=True)], get_args(arg_out, arg_in=False)
            prog_in, prog_out = [[prog_in_id]], [prog_out_id]
            term_out = [1] if term_out else [0]

            # Fit!
            if prog_out_id in PAR_PROG_PID:

                loss, t_acc, p_acc, a_acc= sess.run(
                    [npi.arg_loss, npi.t_metric, npi.p_metric, npi.a_metrics],
                    feed_dict={npi.env_in: env_in, npi.arg_in: arg_in, npi.prg_in: prog_in,
                            npi.y_prog: prog_out, npi.y_term: term_out,
                            npi.y_args[0]: [arg_out[0]], npi.y_args[1]: [arg_out[1]],
                            npi.y_args[2]: [arg_out[2]]})
                step_arg_loss += loss
                term_acc += t_acc
                prog_acc += p_acc
                arg0_acc += a_acc[0]
                arg1_acc += a_acc[1]
                arg2_acc += a_acc[2]
                num_args += 1

            else:

                loss, t_acc, p_acc= sess.run(
                    [npi.default_loss, npi.t_metric, npi.p_metric],
                    feed_dict={npi.env_in: env_in, npi.arg_in: arg_in, npi.prg_in: prog_in,
                            npi.y_prog: prog_out, npi.y_term: term_out})
                step_def_loss += loss
                term_acc += t_acc
                prog_acc += p_acc



        try:
            print ("Step {} Default Step Loss {}, " \
            "Argument Step Loss {}, Term: {}, Prog: {}, A0: {}, " \
            "A1: {}, A2: {}".format(i, step_def_loss / len(x), step_arg_loss / len(x), term_acc / len(x),
                    prog_acc / len(x), arg0_acc / num_args, arg1_acc / num_args,
                    arg2_acc / num_args))
            tmp = stat.mean([arg0_acc / num_args, arg1_acc / num_args, arg2_acc / num_args])
            term_acct.append(term_acc / len(x))
            prog_acct.append(prog_acc/len(x))
            arg_acct.append(tmp)
        except:
            print('main print failed')

    return(stat.mean(term_acct), stat.mean(prog_acct), stat.mean(arg_acct))