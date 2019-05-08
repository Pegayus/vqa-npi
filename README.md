# VQA-NPI
This repository has been built on top of [this](https://github.com/siddk/npi) implementation of NPI for simple addition task. Everything under the folder task/vqa is written by me and the rest are modified version of the mentioned repository. In creation of this repo, these two resources have been greatly helpful in obtaining and processing data:<br/>
1- https://github.com/kexinyi/ns-vqa  <br/>
2- https://github.com/facebookresearch/clevr-dataset-gen <br/>
## RUN
Please make sure that you are haveing the same version of all packages as specified in requirements.txt (specially the tensorflow version that needs to be 1.12.0 (cpu or gpu). To run vqa-npi you need to <br/>
* Clone this repository
* Go to main.py under the main directory and adjust the flags according to the purpose (trainin, testing, validation)
* Run main.py <br/>
It is recommended to only run one module at a time (one flag = TRUE). If you wish to have more than one flag = TRUE at the time, you need to disable the tf.sessions from each modlue and pass it as an input. For example, if you would like to run the test directly after training, you need to go to tasks/vqa/test_direct.py and tasks/vqa/train.py and comment the tensorflow session, add "sess" as input and create the session in main.py to pass it to the both modules. <br/>
## Reproducing Experiments
For obtaining the results in the report, changes need to be made in either of these directories: main.py, train.py, test_direct.py, test.py, npi.py, vqa.py.
### General Results
Go to tasks/vqa/train.py and change the dataset you want to use for training (train_query, or train_count, or train_exist). If you would like to generate figures, be sure to change the labels on the plotting section accordingly. 
### Hyperparameter Tuning
Changes to be made in npi.py (for weight, learning rate, regularization and number of layers) and vqa.py (number of dense layers in NPI environemnt encoder).
### Ablation Study
Go to tasks/vqa/test_direct.py and change the threshold (ABAL_THR) for a specific probability of adding noise. For threshold = 1, you will run the test normally.
