"""
eval.py

Loads in an Addition NPI, and starts a REPL for interactive addition.
"""
from model.npi import NPI
from tasks.vqa.vqa import VQAcore
from tasks.vqa.env.config import CONFIG, get_args, PROGRAM_SET, Scene
import numpy as np
import pickle
import tensorflow as tf

LOG_PATH = "tasks/vqa/log/"
CKPT_PATH = "tasks/vqa/log/model.ckpt"
TEST_PATH = "tasks/vqa/data/train.pickle"
EX_PROG_PID = [2,5,3]
COUNT_PID, COMPARE_PID, MOVE_PTR_PID, DELETE_ROW_PID, QUERY_PID, UNIQUE_PID, EXIST_PID, FILTER_PID =\
    range(8)
COND = {0:'EQ', 1:'GT', 2:'LT', 3:'NEQ'}
PTRS = {0: "ROW_PTR", 1: "POSITION_PTR", 2: "COLOR_PTR", 3: "MATERIAL_PTR", 4: "SHAPE_PTR", 5: "SIZE_PTR"}
D_R = {0: "DOWN", 1: "RESET"}
QR = {1: 'POSITION', 2: 'COLOR', 3: 'MATERIAL', 4: 'SHAPE', 5: 'SIZE'}
FLTR1 = {0:'COLOR', 1:'MATERIAL', 2:'SHAPE', 3:'SIZE'}
FLTR2 = {0:'BLUE',1:'BROWN',2:'CYAN',3:'GRAY',4:'GREEN',5:'PURPLE',6:'RED',7:'YELLOW',
        8:'RUBBER',9:'METAL',10:'CUBE',11:'CYLINDER',12:'SPHERE',13:'LARGE',14:'SMALL'}



def evaluate_vqa():
    """
    Load NPI Model from Checkpoint, and initialize REPL, for interactive carry-addition.
    """
    # Load Data
    with open(TEST_PATH, 'rb') as f:
        data = pickle.load(f)
        data = data[50:100]

    # Initialize Addition Core
    core = VQAcore()

    # Initialize NPI Model
    npi = NPI(core, CONFIG, LOG_PATH)

    with tf.Session() as sess:
        # Restore from Checkpoint
        saver = tf.train.Saver()
        saver.restore(sess, CKPT_PATH)

        # Run REPL
        repl(sess, npi, data)


def repl(session, npi, data):
    input('You are going to see a VQA-NPI test. Each sample is drawn at random. Press any key to continue.')

    while True:
        imgid, qid, qtype, steps = data[np.random.randint(len(data))]

        # Reset NPI States
        print ()
        npi.reset_state()

        # Setup Environment
        scene = Scene(imgid)
        (prog_name, prog_id), arg, term = steps[0]

        cont = 'y'
        while cont == 'y' or cont == 'Y':
            # Print Step Output
            if prog_id == COMPARE_PID:
                a0 = COND.get(arg[0], "OOPS!")
                a_str = "[%s]" % (str(a0))
            elif prog_id == MOVE_PTR_PID:
                a0, a1 = PTRS.get(arg[0], "OOPS!"), D_R.get(arg[1], 'OOPS!')
                a_str = "[%s, %s]" % (str(a0), str(a1))
            elif prog_id == QUERY_PID:
                a0 = QR.get(arg[0], 'OOPS!')
                a_str = "[%s]" % (str(a0))
            elif prog_id == FILTER_PID:
                a0, a1 = FLTR1.get(arg[0], "OOPS!"), FLTR2.get(arg[1], 'OOPS!')
                a_str = "[%s, %s]" % (str(a0), str(a1))
            else:
                a_str = "[]"

            print ('Step: %s, Arguments: %s, Terminate: %s' % (prog_name, a_str, str(term)))
            print ('Pointers:')
            print('ROW: %s, POSITION: %s, COLOR: %s, MATERIAL: %s, SHAPE: %s, SIZE: %s' % (scene.row_ptr[1],
                                                              scene.position_ptr[1],
                                                              scene.color_ptr[1],
                                                              scene.material_ptr[1],
                                                              scene.shape_ptr[1],
                                                              scene.size_ptr[1]))

            # Update Environment if MOVE or WRITE
            if prog_id in EX_PROG_PID:
                scene.execute(prog_id, arg)

            # Print Environment
            scene.print_scene()

            # Get Environment, Argument Vectors
            env_in, arg_in, prog_in = [scene.get_env()], [get_args(arg, arg_in=True)], [[prog_id]]
            t, n_p, n_args = session.run([npi.terminate, npi.program_distribution, npi.arguments],
                                         feed_dict={npi.env_in: env_in, npi.arg_in: arg_in,
                                                    npi.prg_in: prog_in})

            if np.argmax(t) == 1:
                print('Step: %s, Arguments: %s, Terminate: %s' % (prog_name, a_str, str(term)))
                print('Pointers:')
                print('ROW: %s, POSITION: %s, COLOR: %s, MATERIAL: %s, SHAPE: %s, SIZE: %s' % (scene.row_ptr[1],
                                                                       scene.position_ptr[1],
                                                                       scene.color_ptr[1],
                                                                       scene.material_ptr[1],
                                                                       scene.shape_ptr[1],
                                                                       scene.size_ptr[1]))
                # Update Environment if MOVE or WRITE
                if prog_id in EX_PROG_PID:
                    scene.execute(prog_id, arg)

                # Print Environment
                scene.print_scene()


                # print ("Correct!" if output == (x + y) else "Incorrect!")
                break

            else:
                prog_id = np.argmax(n_p)
                prog_name = PROGRAM_SET[prog_id][0]

                if prog_id == COMPARE_PID:
                    arg = [np.argmax(n_args[0])]
                elif prog_id == MOVE_PTR_PID:
                    arg = [np.argmax(n_args[0]), np.argmax(n_args[1])]
                elif prog_id == QUERY_PID:
                    arg = [np.argmax(n_args[0])]
                elif prog_id == FILTER_PID:
                    arg = [np.argmax(n_args[0]), np.argmax(n_args[1])]
                else:
                    arg = "[]"
                term = False

            cont = 'y'
            cont = input('Continue? ')