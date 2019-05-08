"""
config.py

Configuration Variables for the VQA NPI Task => Stores:
    - table dimensions (row and column)
    - program embedding/arguments information
Info from:
    - CLEVR datset generation repo: https://github.com/facebookresearch/clevr-dataset-gen.git
    - VQA repo: https://github.com/kexinyi/ns-vqa
    - NPI repo: https://github.com/siddk/npi
"""
import numpy as np
import sys
import time
import pandas as pd
import json

PATH = "tasks/vqa/data/scenes.json"
with open(PATH, 'r') as f:
    SCENES = json.load(f)

# for making environment
COLORS = {'BLUE': 0, 'BROWN': 1, 'CYAN': 2, 'GRAY': 3, 'GREEN': 4, 'PURPLE': 5, 'RED': 6, 'YELLOW': 7}
MATERIALS = {'RUBBER': 0, 'METAL': 1}
SHAPES = {'CUBE': 0, 'CYLINDER': 1, 'SPHERE': 2}
SIZES = {'LARGE': 0, 'SMALL': 1}


CONFIG = {
    # "CLVR_ROW": 10,        # Number of objects in a scene - from CLEVR repo: min=3, max=10
    # "CLVR_COL": 6 ,        # Table columns: id, position(x,y,z), color, material, shape, size

    "ENVIRONMENT_COL": 20,        # one hot encoding of valid(2),x,y,z(3),color(8),material(2),shape(3),size(2)
    "ENVIRONMENT_ROW": 10,        # Number of objects in a scene - from CLEVR repo: min=3, max=10
    # "ENVIRONMENT_DEPTH": 10,    # not applicable in vqa
    "PTR_ROW": 6,                 # 6 pointers
    "PTR_COL": 10,                # one hot encoding of env rows (0 to 9)

    "ARGUMENT_NUM": 3,            # Maximum Number of Program Arguments
    "ARGUMENT_DEPTH": 17,         # Size of Argument Vector => One-Hot, Options 0-9, Default (10)
    "DEFAULT_ARG_VALUE": 16,      # Default Argument Value

    "PROGRAM_NUM": 8,            # Maximum Number of Subroutines
    "PROGRAM_KEY_SIZE": 8,        # Size of the Program Keys
    "PROGRAM_EMBEDDING_SIZE": 15  # Size of the Program Embeddings
}

PROGRAM_SET = [
    ("COUNT",),               # COUNT operation
    ("COMPARE", 4),           # Given either of [EQ,NEQ,GT,LS] (4 options) gives 'yes' or 'no' (2 options)
    ("MOVE_PTR", 6, 2),       # Move pointers (9 options) either DOWN or RESET [-1] (2 options)
    ("DELETE_ROW",),             # Given pointer 9, delete the pointed row from the scene
    ("QUERY", 7),             # Given either of [COLOR, MATERIAL, SHAPE, SIZE, X, Y, Z], give object's value
    ("UNIQUE",),              # UNIQUE operation on SCENE
    ("EXIST",),               # EXIST operation, [calls subroutines]
    ("FILTER", 4, 16)        # Given either of [COLOR, MATERIAL, SHAPE, SIZE] for either of [BLUE, BROWN, CYAN, GRAY, GREEN, PURPLE, RED, YELLOW, RUBBER, METAL, CUBE, CYLINDER, SPHERE, LARGE, SMALL] filters the rows into NEW_SCENE. [calls subroutines]
    # ("RELATE", 4),            # Given either of [BEHIND, FRONT, LEFT, RIGHT] find all that satisfy, [calls subroutine]
    # ("SAME", 4)               # Given either of [COLOR, MATERIAL, SHAPE, SIZE] find all that is the same, [calls subroutines]
]

PROGRAM_ID = {x[0]: i for i, x in enumerate(PROGRAM_SET)}
TABLE_COLUMNS = ['position', 'color', 'material', 'shape', 'size','valid']

class Scene():           # Table Environment
    def __init__(self, imgid):
        # Initialize the scene as a list of dictionary [{'color:...},{'color':...},..]
        self.scene = SCENES[str(imgid)]
        for i in range(len(self.scene)):
            self.scene[i]['valid'] = 1

        # Pointers initially all start at top of each column -1, objects(rows) are from 0 to 10
        self.rows, self.cols = CONFIG['ENVIRONMENT_ROW'], CONFIG['ENVIRONMENT_COL']
        self.row_ptr, self.position_ptr, self.color_ptr, self.material_ptr, self.shape_ptr, self.size_ptr  =\
            self.ptrs = [(x, -1) for x in range(6)]

    #**************************************************************
    # not used
    #***************************************************************
    # def get_scene(self):
    #     return self.scene
    #
    # def set_scene(self, s):
    #     self.scene = s

    # def new_scene(self):
    #     return []
    #
    # def count(self):
    #     valid_scene = [i for i in self.scene if i['valid'] == 1]
    #     return len(valid_scene)
    #
    # def compare(self, a, b, mod):
    #     if type(a) == type(b):
    #         if mod == 'EQ':
    #             return a == b
    #         if mod == 'NEQ':
    #             return a != b
    #         if mod == 'GT':
    #             return a > b
    #         if mod == 'LT':
    #             return a< b
    #     else:
    #         return 'error'
    #
    # def add_row(self, row, new_scene):
    #     new_scene.append(row)
    #     return new_scene
    #
    # def query(self, a, row):
    #     if a in row.keys():
    #         return row[a]
    #     else:
    #         return 'error'
    #
    # def exist(self):
    #     valid_scene = [i for i in self.scene if i['valid'] == 1]
    #     if len(valid_scene) > 0:
    #         return 'yes'
    #     else:
    #         return 'no'

    def print_scene(self):
        # make a dict of list
        valid_scene = [i for i in self.scene if i['valid'] == 1]
        if len(valid_scene) > 0:
            tmp = {k:[] for k in valid_scene[0].keys()}
            for row in valid_scene:
                for item in row:
                    tmp[item].append(row[item])
            scene_df = pd.DataFrame(tmp)
        else:
            scene_df = pd.DataFrame()
        print(scene_df)
        print('')
        time.sleep(.1)
        sys.stdout.flush()

    def get_env(self):
        # make pointer array
        ptr = np.zeros((6, 10))
        for p in self.ptrs:
            ptr[p[0]] = np.eye(10)[p[1]]
        # make main table
        env = np.zeros((self.rows, self.cols))
        for i in range(self.rows):
            if i < len(self.scene):
                env[i][0:2] = np.eye(2)[self.scene[i]['valid']]  # valid
                env[i][2:5] = self.scene[i]['position']  # x, y, z
                env[i][5:13] = np.eye(8)[COLORS[self.scene[i]['color'].upper()]]  # color
                env[i][13:15] = np.eye(2)[MATERIALS[self.scene[i]['material'].upper()]]  # material
                env[i][15:18] = np.eye(3)[SHAPES[self.scene[i]['shape'].upper()]]  # shape
                env[i][18:20] = np.eye(2)[SIZES[self.scene[i]['size'].upper()]]  # size
            else:
                env[i][0:2] = np.eye(2)[0]
        return np.concatenate((env.flatten(), ptr.flatten()))

    def execute(self, prog_id, args):
        if prog_id == PROGRAM_ID['MOVE_PTR']:               # MOVE_PTR
            valid_scene = [i for i in range(len(self.scene)) if self.scene[i]['valid'] == 1]
            if len(valid_scene) > 0:
                valid_end = max(valid_scene)
            else:
                valid_end = -1
            ptr, dr = args
            if ptr == 0:
                if dr == 0 and self.row_ptr[1] < valid_end:  #Down
                    self.row_ptr = (self.row_ptr[0], self.row_ptr[1] + 1)
                else:       # Reset
                    self.row_ptr = (self.row_ptr[0], -1)
            elif ptr == 1:
                if dr == 0 and self.position_ptr[1] < valid_end:  #Down
                    self.position_ptr = (self.position_ptr[0], self.position_ptr[1] + 1)
                else:       # Reset
                    self.position_ptr = (self.position_ptr[0], -1)
            elif ptr == 2:
                if dr == 0 and self.color_ptr[1] < valid_end:  #Down
                    self.color_ptr = (self.color_ptr[0], self.color_ptr[1] + 1)
                else:       # Reset
                    self.color_ptr = (self.color_ptr[0], -1)
            elif ptr == 3:
                if dr == 0 and self.material_ptr[1] < valid_end:  #Down
                    self.material_ptr = (self.material_ptr[0], self.material_ptr[1] + 1)
                else:       # Reset
                    self.material_ptr = (self.material_ptr[0], -1)
            elif ptr == 4:
                if dr == 0 and self.shape_ptr[1] < valid_end:  #Down
                    self.shape_ptr = (self.shape_ptr[0], self.shape_ptr[1] + 1)
                else:       # Reset
                    self.shape_ptr = (self.shape_ptr[0], -1)
            elif ptr == 5:
                if dr == 0 and self.size_ptr[1] < valid_end:  #Down
                    self.size_ptr = (self.size_ptr[0], self.size_ptr[1] + 1)
                else:       # Reset
                    self.size_ptr = (self.size_ptr[0], -1)
            else:
                raise NotImplementedError
            self.ptrs = [self.row_ptr, self.position_ptr, self.color_ptr, self.material_ptr, self.shape_ptr, self.size_ptr]

        elif prog_id == PROGRAM_ID['UNIQUE']:             # UNIQUE
            valid_scene = [i for i in self.scene if i['valid'] == 1]
            if len(valid_scene) > 0:
                count = 0
                for i in range(len(self.scene)):
                    if self.scene[i]['valid'] == 1:
                        if count == 0:
                            count += 1
                        else:
                            self.scene[i]['valid'] = 0

        elif prog_id == PROGRAM_ID['DELETE_ROW']:             # DELETE_ROW
            if self.row_ptr[1] < len(self.scene):
                self.scene[self.row_ptr[1]]['valid'] = 0
            else:
                print('row_ptr index out of range.')
                raise NotImplementedError




class Arguments():             # Program Arguments
    def __init__(self, args, num_args=CONFIG["ARGUMENT_NUM"], arg_depth=CONFIG["ARGUMENT_DEPTH"]):
        self.args = args
        self.arg_vec = np.zeros((num_args, arg_depth), dtype=np.float32)


def get_args(args, arg_in=True):
    if arg_in:
        arg_vec = np.zeros((CONFIG["ARGUMENT_NUM"], CONFIG["ARGUMENT_DEPTH"]), dtype=np.int32)
    else:
        arg_vec = [np.zeros((CONFIG["ARGUMENT_DEPTH"]), dtype=np.int32) for _ in
                   range(CONFIG["ARGUMENT_NUM"])]
    if isinstance(args, str):
        args = eval(args)
    if len(args) > 0:
        for i in range(CONFIG["ARGUMENT_NUM"]):
            if i >= len(args):
                arg_vec[i][CONFIG["DEFAULT_ARG_VALUE"]] = 1
            else:
                 arg_vec[i][args[i]] = 1

    else:
        for i in range(CONFIG["ARGUMENT_NUM"]):
            arg_vec[i][CONFIG["DEFAULT_ARG_VALUE"]] = 1
    return arg_vec.flatten() if arg_in else arg_vec

