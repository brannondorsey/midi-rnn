# MIDI RNN

Generate monophonic melodies using a basic LSTM RNN. Great for Machine Learning MIDI generation baselines. Made using Keras.

## Getting Started

`midi-rnn` should work in MacOS and Linux environments. Open a terminal and run:
```bash
# clone this repo
git clone https://github.com/brannondorsey/midi-rnn.git

# Install the dependencies. You may need to prepend sudo to 
# this command if you get an error
pip install requirements.txt
``` 

If you have CUDA installed and would like to train using your GPU, additionally run:
```bash
pip install tensorflow-gpu
``` 

## Training a Model

First create a folder of MIDI files that you would like to train your model with. I've included ~130 files from the [Lakh MIDI Dataset](http://colinraffel.com/projects/lmd/) inside `data/midi` that you can use to get started.

Once you've got a collection of midi files you can train your model with `train.py`.

```bash
python train.py --data_dir data/midi
```

For a list of supported command line flags, run:

```
python train.py --help
```

Or [see below](#trainpy) for a detailed description of each option. By default, model checkpoints are saved in auto-incrementing folders inside of `experiments`, however, their location can be set explicitly with the `--experiment_dir flag`.

### Monitoring Training with Tensorboard

`model-rnn` logs training metrics using Tensorboard. These logs are stored in a folder called `tensorboard-logs` inside of your `--experiment_dir`.

```
# Compare the training metrics of all of your experiments at once
tensorboard --logdir experiments/
```

Once Tensorboard is running, navigate to `http://localhost:6006` to view the training metrics for your model in real time.

## Generating MIDI

Once you've trained your model, you can generate MIDI files using `sample.py`.

```bash
python sample.py
```

By default, this creates 10 MIDI files using a checkpoint from the most recent folder in `experiments/` and saves the generated files to `generated/` inside of that experiment (e.g. `experiments/01/generated/`). You can specify which model you would like to use when generating using the `--experiment_dir` flag. You can also specify where you would like to save the generated files by including a value for the `--save_dir` flag. For a complete list of command line flags, see below.

## Command Line Arguments

### `train.py`

- `--data_dir`: A folder containing `.mid` files to use for training. All files in this folder will be used for training.
- `--experiment_dir`: The name of the folder to use when saving the model checkpoints and Tensorboard logs. If omitted, a new folder will be created with an auto-incremented number inside of `experiments/`.
- `--rnn_size` (default: 64): The number of neurons in hidden layers.
- `--num_layers` (default: 1): The number of hidden layers.
- `--learning_rate` (default: 0.002): The learning rate to use with the optimizer. It is recomended to adjust this value in multiples of 10.
- `--window_size` (default: 20): The number of previous notes (and rests) to use as input to the network at each step (measured in 16th notes). It is helpful to think of this as the fixed width of a piano roll rather than individual events.
- `--batch_size` (default: 32): The number of samples to pass through the network before updating weights (backpropagating).
- `--num_epochs` (default: 10): The number of epochs before completing training. One epoch is equal to one full pass through all midi files in `--data_dir`. Because of the way files are lazy loaded, this number can only be an estimate.
- `--dropout` (default: 0.2): The normalized percentage (0-1) of weights to randomly turn "off" in each layer during a training step. This is a regularization technique called which helps prevent model overfitting. Recommended values are between 0.2 and 0.5, or 20% and 50%. 
- `--optimizer` (default: "adam"): The optimization algorithm to use when minimizing your loss function. See https://keras.io/optimizers for a list of supported optimizers and and links to their descriptions.
- `--grad_clip` (default: 5.0): Clip backpropagated gradients to this value.
- `--message`: An optional note that can be used to describe your experiment. This text will be saved to `message.txt` inside of `--experiment_dir`. Including a value for this flag is very helpful if you find yourself running many experiments.
- `--n_jobs` (default 1): The number of CPU cores to use when loading and parsing MIDI files from `--data_dir`. Increasing this value can dramatically speed up training. I commonly set this value to use all cores, which for my quad-core machine is 8 (Intel CPUs often have 2 virtual cores per CPU).
- `--max_files_in_ram` (default: 50): Files in `--data_dir` are loaded into RAM in small batches, processed, and then released to avoid having to load all training files into memory at once (which may be impossible when training on hundreds of files on a machine with limited memory). This value specifies the maximum number of MIDI files to keep in RAM at any one time. Using a larger number significantly speeds up training, however it also runs the risk of using too much RAM and causing your machine to [thrash](https://en.wikipedia.org/wiki/Thrashing_(computer_science)) or crash. You can find a nice balance by inspecting your system monitor (Activity Monitor on MacOS and Monitor on Ubuntu) while training and adjusting accourdingly.

```
usage: train.py [-h] [--data_dir DATA_DIR] [--experiment_dir EXPERIMENT_DIR]
                [--rnn_size RNN_SIZE] [--num_layers NUM_LAYERS]
                [--learning_rate LEARNING_RATE] [--window_size WINDOW_SIZE]
                [--batch_size BATCH_SIZE] [--num_epochs NUM_EPOCHS]
                [--dropout DROPOUT]
                [--optimizer {sgd,rmsprop,adagrad,adadelta,adam,adamax,nadam}]
                [--grad_clip GRAD_CLIP] [--message MESSAGE] [--n_jobs N_JOBS]
                [--max_files_in_ram MAX_FILES_IN_RAM]

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   data directory containing .mid files to use
                        fortraining (default: data/midi)
  --experiment_dir EXPERIMENT_DIR
                        directory to store checkpointed models and tensorboard
                        logs.if omitted, will create a new numbered folder in
                        experiments/. (default: experiments/default)
  --rnn_size RNN_SIZE   size of RNN hidden state (default: 64)
  --num_layers NUM_LAYERS
                        number of layers in the RNN (default: 1)
  --learning_rate LEARNING_RATE
                        learning rate. If not specified, the recommended
                        learning rate for the chosen optimizer is used.
                        (default: None)
  --window_size WINDOW_SIZE
                        Window size for RNN input per step. (default: 20)
  --batch_size BATCH_SIZE
                        minibatch size (default: 32)
  --num_epochs NUM_EPOCHS
                        number of epochs before stopping training. (default:
                        10)
  --dropout DROPOUT     percentage of weights that are turned off every
                        training set step. This is a popular regularization
                        that can help with overfitting. Recommended values are
                        0.2-0.5 (default: 0.2)
  --optimizer {sgd,rmsprop,adagrad,adadelta,adam,adamax,nadam}
                        The optimization algorithm to use. See
                        https://keras.io/optimizers for a full list of
                        optimizers. (default: adam)
  --grad_clip GRAD_CLIP
                        clip gradients at this value. (default: 5.0)
  --message MESSAGE, -m MESSAGE
                        a note to self about the experiment saved to
                        message.txt in --experiment_dir. (default: None)
  --n_jobs N_JOBS, -j N_JOBS
                        Number of CPUs to use when loading and parsing midi
                        files. (default: 1)
  --max_files_in_ram MAX_FILES_IN_RAM
                        The maximum number of midi files to load into RAM at
                        once. A higher value trains faster but uses more RAM.
                        A lower value uses less RAM but takes significantly
                        longer to train. (default: 50)
```

### `sample.py`

- `--experiment_dir` (default: most recent folder in `experiments/`): Directory from which to load model checkpoints. If left unspecified, it loads the model from the most recently added folder in `experiments/`.
- `--save_dir` (default: `generated/` inside of `--experiment_dir`): Directory to save generated files to.
- `--midi_instrument` (default: "Acoustic Grand Piano"): The name (or program number, `0-127`) of the General MIDI instrument to use for the generated files. A complete list of General MIDI instruments can be found [here](https://www.midi.org/specifications/item/).
- `--num_files` (default: 10): The number of MIDI files to generate.
- `--file_length` (default: 1000): The length of each generated MIDI file, specified in 16th notes.
- `--prime_file`: The path to a `.mid` file to use to prime/seed the generated files. A random window of this file will be used to seed each generated file.
- `--data_dir`: Used to select random files to prime/seed from if `--prime_file` is not specified.

```
usage: sample.py [-h] [--experiment_dir EXPERIMENT_DIR] [--save_dir SAVE_DIR]
                 [--midi_instrument MIDI_INSTRUMENT] [--num_files NUM_FILES]
                 [--file_length FILE_LENGTH] [--prime_file PRIME_FILE]
                 [--data_dir DATA_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --experiment_dir EXPERIMENT_DIR
                        directory to load saved model from. If omitted, it
                        will use the most recent directory from experiments/.
                        (default: experiments/default)
  --save_dir SAVE_DIR   directory to save generated files to. Directory will
                        be created if it doesn't already exist. If not
                        specified, files will be saved to generated/ inside
                        --experiment_dir. (default: None)
  --midi_instrument MIDI_INSTRUMENT
                        MIDI instrument name (or number) to use for the
                        generated files. See
                        https://www.midi.org/specifications/item/gm-level-1
                        -sound-set for a full list of instrument names.
                        (default: Acoustic Grand Piano)
  --num_files NUM_FILES
                        number of midi files to sample. (default: 10)
  --file_length FILE_LENGTH
                        Length of each file, measured in 16th notes. (default:
                        1000)
  --prime_file PRIME_FILE
                        prime generated files from midi file. If not specified
                        random windows from the validation dataset will be
                        used for for seeding. (default: None)
  --data_dir DATA_DIR   data directory containing .mid files to use
                        forseeding/priming. Required if --prime_file is not
                        specified (default: data/midi)

```