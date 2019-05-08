"""
train.py

Core training script for the VQA task-specific NPI. Instantiates a model, then trains using
the precomputed data.
"""
import numpy as np
from model.npi import NPI
from tasks.vqa.test import test_vqa
from tasks.vqa.vqa import VQAcore
from tasks.vqa.env.config import CONFIG, get_args, Scene
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import statistics as stat
import json

EX_PROG_PID = [2,5,3]
PAR_PROG_PID = [1,2,4,7]
COMPARE_PID, MOVE_PTR_PID, QUERY_PID, FILTER_PID = PAR_PROG_PID
# ROW_PTR, POSITION_PTR, COLOR_PTR, MATERIAL_PTR, SHAPE_PTR, SIZE_PTR = range(6)
# DOWN, RESET = 0, 1

DATA_PATH = "tasks/vqa/data/train_query.pickle"
DATA_PATH2 = "tasks/vqa/data/train_count.pickle"
DATA_PATH3 = "tasks/vqa/data/train_exist.pickle"
LOG_PATH = "tasks/vqa/log/"
SAVE_PATH = "general/acc/"

def train_vqa(epochs, verbose=0):
    """
    Instantiates a VQA Core, NPI, then loads and fits model to data.

    :param epochs: Number of epochs to train for.
    """
    # Load Data
    with open(DATA_PATH, 'rb') as f:
        dataT = pickle.load(f)
        data = dataT[:80]
        test = dataT[80:100]
    with open(DATA_PATH2, 'rb') as f:
        dataT2 = pickle.load(f)
        test_out = dataT2[:20]
    with open(DATA_PATH3, 'rb') as f:
        dataT3 = pickle.load(f)
        test_out2 = dataT3[:20]


    # Initialize VQA Core
    print ('Initializing VQA Core!')
    core = VQAcore()

    # Initialize NPI Model
    print ('Initializing NPI Model!')
    npi = NPI(core, CONFIG, LOG_PATH, verbose=verbose)

    # Initialize TF Saver
    saver = tf.train.Saver()

    # Initialize TF Session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Start Training
        removed = {}
        errors = {}
        # for learning curve
        count = 0
        # tot_loss_def = []
        # tot_loss_arg = []
        # test_loss_def = []
        # test_loss_arg = []
        # test1_loss_def = []
        # test1_loss_arg = []
        # test2_loss_def = []
        # test2_loss_arg = []
        test_term_acct = []
        test_prog_acct = []
        test_arg_acct = []
        train_term_acct = []
        train_prog_acct = []
        train_arg_acct = []
        test1_term_acct = []
        test1_prog_acct = []
        test1_arg_acct = []
        test2_term_acct = []
        test2_prog_acct = []
        test2_arg_acct = []

        step = []
        for ep in range(1, epochs + 1):
            removed[ep] = 0
            for i in range(len(data)):
                # Reset NPI States
                npi.reset_state()

                # Setup Environment
                _, imgid, qid, qtype, steps = data[i]
                scene = Scene(imgid)
                x, y = steps[:-1], steps[1:]
                if len(x) == 0 or len(y) == 0:
                    removed[ep] += 1
                    continue
                count += 1

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
                    # env_in = [np.asarray(list(env_in.values())).transpose().flatten()]
                    arg_in, arg_out = [get_args(arg, arg_in=True)], get_args(arg_out, arg_in=False)
                    prog_in, prog_out = [[prog_in_id]], [prog_out_id]
                    term_out = [1] if term_out else [0]

                    # Fit!
                    if prog_out_id in PAR_PROG_PID :
                        loss, t_acc, p_acc, a_acc, _ = sess.run(
                            [npi.arg_loss, npi.t_metric, npi.p_metric, npi.a_metrics, npi.arg_train_op],
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
                        loss, t_acc, p_acc, _ = sess.run(
                            [npi.default_loss, npi.t_metric, npi.p_metric, npi.default_train_op],
                            feed_dict={npi.env_in: env_in, npi.arg_in: arg_in, npi.prg_in: prog_in,
                                    npi.y_prog: prog_out, npi.y_term: term_out})
                        step_def_loss += loss
                        term_acc += t_acc
                        prog_acc += p_acc



                try:
                    print ("Epoch {0:02d} Step {1:03d} Default Step Loss {2:05f}, " \
                    "Argument Step Loss {3:05f}, Term: {4:03f}, Prog: {5:03f}, A0: {6:03f}, " \
                    "A1: {7:03f}, A2: {8:03}".format(ep, i, step_def_loss / len(x), step_arg_loss / len(x), term_acc / len(x),
                            prog_acc / len(x), arg0_acc / num_args, arg1_acc / num_args,
                            arg2_acc / num_args))
                    if count % 10 == 0:
                        # Save Model
                        tmp = stat.mean([arg0_acc / num_args, arg1_acc / num_args, arg2_acc / num_args])
                        saver.save(sess, 'tasks/vqa/log/model.ckpt')
                        train_arg_acct.append(tmp/len(x))
                        train_prog_acct.append(prog_acc / len(x))
                        train_term_acct.append(term_acc / len(x))
                        step.append(count)
                        a , b, c = test_vqa(test, npi, core, sess)
                        test_arg_acct.append(c)
                        test_prog_acct.append(b)
                        test_term_acct.append(a)
                        a, b, c = test_vqa(test_out, npi, core, sess)
                        test1_arg_acct.append(c)
                        test1_prog_acct.append(b)
                        test1_term_acct.append(a)
                        a, b, c = test_vqa(test_out2, npi, core, sess)
                        test2_arg_acct.append(c)
                        test2_prog_acct.append(b)
                        test2_term_acct.append(a)
                except:
                    print('main print failed')



            # Save Model
            saver.save(sess, 'tasks/vqa/log/model.ckpt')
        # print learning curve
        print('train term,prog,arg: ', test_term_acct[-1], test_prog_acct[-1], test_arg_acct[-1])
        print('test_inside term,prog,arg: ', test_term_acct[-1], test_prog_acct[-1], test_arg_acct[-1])
        print('test_out term,prog,arg: ', test1_term_acct[-1], test1_prog_acct[-1], test1_arg_acct[-1])
        print('test_out2 term,prog,arg: ', test2_term_acct[-1], test2_prog_acct[-1], test2_arg_acct[-1])

        plt.figure(figsize=(20, 5))
        plt.plot(step, train_term_acct, 'b', label='train_query_term')
        plt.plot(step, test_term_acct, 'm', label='test_query_term')
        plt.plot(step, test1_term_acct, 'c', label='test_count_term')
        plt.plot(step, test2_term_acct, 'k', label='test_exist_term')
        plt.legend()
        plt.xticks(step)
        plt.xlabel('step')
        plt.ylabel('acc')
        plt.title('learning curve for termination')
        plt.savefig(SAVE_PATH + 'acc_query_term')
        plt.close()
        plt.figure(figsize=(20, 5))
        plt.plot(step, train_prog_acct, 'b', label='train_query_prog')
        plt.plot(step, test_prog_acct, 'm', label='test_query_prog')
        plt.plot(step, test1_prog_acct, 'c', label='test_count_prog')
        plt.plot(step, test2_prog_acct, 'k', label='test_exist_prog')
        plt.legend()
        plt.xticks(step)
        plt.xlabel('step')
        plt.ylabel('acc')
        plt.title('learning curve for program')
        plt.savefig(SAVE_PATH + 'acc_query_prog')
        plt.close()
        plt.figure(figsize=(20, 5))
        plt.plot(step, train_arg_acct, 'b', label='train_query_arg')
        plt.plot(step, test_arg_acct, 'm', label='test_query_arg')
        plt.plot(step, test1_arg_acct, 'c', label='test_count_arg')
        plt.plot(step, test2_arg_acct, 'k', label='test_exist_arg')
        plt.legend()
        plt.xticks(step)
        plt.xlabel('step')
        plt.ylabel('acc')
        plt.title('learning curve for arguments')
        plt.savefig(SAVE_PATH + 'acc_query_arg')
        plt.close()
        # plt.hold
        # print learning curve
        # plt.plot(step, test_loss_def, 'r', label='test_inside_loss_def')
        # plt.plot(step, test_loss_arg, 'm', label='test_inside_loss_arg')
        # plt.legend()
        # plt.xticks(step)
        # plt.xlabel('step')
        # plt.ylabel('loss')
        # plt.title('learning curve')
        # plt.savefig(SAVE_PATH + 'learning_curve')
        # plt.close()
