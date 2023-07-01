# VQA-NPI

Welcome to the VQA-NPI repository. This project has been developed as an extension of an existing Neural Program Interpreter (NPI) implementation for simple addition tasks. The additional work and modifications have been primarily undertaken in the `task/vqa` directory, while the remainder of the files constitute adaptations of the original repository. Please note that this project utilizes Python 3.5 and Tensorflow 1.12.0, differing from the base repository which was written in Python 2 and Tensorflow 0.12. 

In the process of constructing this repository, the following resources were invaluable in obtaining and processing data:

1. [ns-vqa](https://github.com/kexinyi/ns-vqa)
2. [clevr-dataset-gen](https://github.com/facebookresearch/clevr-dataset-gen)

## Getting Started

To effectively utilize this repository, please follow the steps outlined below:

1. Ensure your environment meets the necessary requirements as specified in the `requirements.txt` file, with particular attention to Tensorflow version 1.12.0.

2. Clone this repository to your local machine.

3. Navigate to `main.py` in the main directory and adjust the flags in accordance with your intended usage (training, testing, validation).

4. Execute `main.py`.

For optimal results, it is recommended to run one module at a time (i.e., one flag set to TRUE). In cases where more than one flag is set to TRUE, it becomes necessary to disable the tf.sessions from each module and pass it as input. For instance, if you plan on running the test immediately after training, you should modify `tasks/vqa/test_direct.py` and `tasks/vqa/train.py` to comment out the Tensorflow session, add "sess" as input, and create the session in `main.py` to pass it to both modules.

### Important Notes

* The existing model under `tasks/vqa/log` is trained on 80 samples from the `train_query` dataset. To run tests or evaluations on different data, ensure you first train and save the model in the log directory.

* The "evaluation" script guides you through the internal workings of NPI, allowing for interactive engagement with the environment at each step (advancement requires pressing "y" or "Y"). This will provide insight into the environment updates and network predictions.

## Replicating Experimental Results

To reproduce the results detailed in the report, modifications will need to be made in the following directories: `main.py`, `train.py`, `test_direct.py`, `test.py`, `npi.py`, `vqa.py`.

### General Results

For general results, navigate to `tasks/vqa/train.py` and specify the dataset for training (`train_query`, `train_count`, or `train_exist`). If you intend to generate figures, ensure you adjust the labels in the plotting section accordingly.

### Hyperparameter Tuning

Modifications for hyperparameter tuning can be made in `npi.py` (for weight, learning rate, regularization and number of layers) and `vqa.py` (for the number of dense layers in the NPI environment encoder).

### Ablation Study

For the ablation study, navigate to `tasks/vqa/test_direct.py` and adjust the `ABAL_THR` threshold for the specific probability of adding noise. A threshold of 1 will allow normal test execution.

