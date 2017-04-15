#!/usr/bin/env python

import os, time, sys, argparse
import utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

OUTPUT_SIZE = 129 # 0-127 notes + 1 for rests

def parse_args():

    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='data/midi',
                        help='data directory containing .mid files to use for' \
                             'training')
    parser.add_argument('--experiment_dir', type=str,
                        default='experiments/default',
                        help='directory to store checkpointed models and tensorboard logs.' \
                             'if omitted, will create a new numbered folder in experiments/.')
    parser.add_argument('--n_jobs', '-j', type=int, 
                        help='Number of CPUs to use when loading and parsing midi files.')
    parser.add_argument('--rnn_size', type=int, default=64,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--window_size', type=int, default=50,
                        help='Window size for RNN input per step.')
    # parser.add_argument('--model', type=str, default='lstm',
    #                     help='rnn, gru, lstm')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--output_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the hidden layer (1.0 - this value = Dropout)')
    parser.add_argument('--input_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the input layer (1.0 - this value = Dropout)')
    parser.add_argument('--verbose', '-v', type=bool, default=1,
                        help='log verbose.')
    return parser.parse_args()

# create or load a saved model
# returns the model and the epoch number (>1 if loaded from checkpoint)
def get_model(args, experiment_dir=None):
    
    epoch = 0
    
    if not experiment_dir:
        model = Sequential()
        model.add(LSTM(args.rnn_size,
                       return_sequences=False,
                       input_shape=(args.window_size, OUTPUT_SIZE)))
        model.add(Dropout(1.0 - args.input_keep_prob))
        # model.add(LSTM(32, return_sequences=False))
        # model.add(Dropout(0.2))
        model.add(Dense(OUTPUT_SIZE))
        model.add(Activation('softmax'))
    else:
        model, epoch = model_utils.load_model_from_checkpoint(experiment_dir)

    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam',
                  metrics=['accuracy'])
    return model, epoch

def get_callbacks(experiment_dir, checkpoint_monitor='val_acc', model_index=0):
    
    callbacks = []
    
    # save model checkpoints
    filepath = os.path.join(experiment_dir, 
                            'checkpoints', 
                            'model-' + 
                             str(model_index) + 
                             '_checkpoint-epoch_{epoch:03d}-val_acc_{val_acc:.3f}.hdf5')

    callbacks.append(ModelCheckpoint(filepath, 
                                     monitor=checkpoint_monitor, 
                                     verbose=1, 
                                     save_best_only=False, 
                                     mode='max'))

    callbacks.append(ReduceLROnPlateau(monitor='val_loss', 
                                       factor=0.5, 
                                       patience=3, 
                                       verbose=1, 
                                       mode='auto', 
                                       epsilon=0.0001, 
                                       cooldown=0, 
                                       min_lr=0))

    callbacks.append(TensorBoard(log_dir=os.path.join(experiment_dir, 'tensorboard-logs'), 
                                histogram_freq=0, 
                                write_graph=True, 
                                write_images=False))

def main():

    args = parse_args()

    # get paths to midi files in --data_dir
    midi_files = [os.path.join(args.data_dir, path) \
                  for path in os.listdir(args.data_dir) \
                  if '.mid' in path or '.midi' in path]

    utils.log(
        'Found {} midi files in {}'.format(len(midi_files), args.data_dir),
        args.verbose
    )

    if len(midi_files) < 1:
        utils.log(
            'Error: no midi files found in {}. Exiting.'.format(args.data_dir),
            args.verbose
        )
        exit(1)

    # create the experiment directory and return its name
    experiment_dir = utils.create_experiment_dir(args.experiment_dir, args.verbose)

    val_split = 0.2 # use 20 percent for validation
    val_split_index = int(float(len(midi_files)) * val_split)
    
    # use generators to lazy load train/validation data, ensuring that the
    # user doesn't have to load all midi files into RAM at once
    train_generator = utils.get_data_generator(midi_files[0:val_split_index], 
                                               window_size=args.window_size,
                                               batch_size=args.batch_size,
                                               num_threads=args.n_jobs)

    val_generator = utils.get_data_generator(midi_files[val_split_index:], 
                                             window_size=args.window_size,
                                             batch_size=args.batch_size,
                                             num_threads=args.n_jobs)

    model, epoch = get_model(args)
    if args.verbose:
        print(model.summary())

    utils.save_model(model, experiment_dir)
    utils.log('Saved model to {}'.format(os.path.join(experiment_dir, 'model.json')),
              args.verbose)

    callbacks = get_callbacks(experiment_dir)
    
    print('fitting model...')
    model.fit_generator(train_generator,
                        steps_per_epoch=10000, 
                        epochs=10,
                        validation_data=val_generator, 
                        validation_steps=2000,
                        verbose=1, 
                        callbacks=callbacks,
                        initial_epoch=epoch)

if __name__ == '__main__':
    main()