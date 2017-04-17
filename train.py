#!/usr/bin/env python
import os, argparse, time
import utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

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
    parser.add_argument('--rnn_size', type=int, default=64,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='learning rate. If not specified, the recommended learning '\
                        'rate for the chosen optimizer is used.')
    parser.add_argument('--window_size', type=int, default=20,
                        help='Window size for RNN input per step.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs before stopping training.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='percentage of weights that are turned off every training '\
                        'set step. This is a popular regularization that can help with '\
                        'overfitting. Recommended values are 0.2-0.5')
    parser.add_argument('--optimizer', 
                        choices=['sgd', 'rmsprop', 'adagrad', 'adadelta', 
                                 'adam', 'adamax', 'nadam'], default='adam',
                        help='The optimization algorithm to use. '\
                        'See https://keras.io/optimizers for a full list of optimizers.')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='clip gradients at this value.')
    parser.add_argument('--message', '-m', type=str,
                        help='a note to self about the experiment saved to message.txt '\
                        'in --experiment_dir.')
    parser.add_argument('--n_jobs', '-j', type=int, default=1, 
                        help='Number of CPUs to use when loading and parsing midi files.')
    parser.add_argument('--max_files_in_ram', default=25, type=int,
                        help='The maximum number of midi files to load into RAM at once.'\
                        ' A higher value trains faster but uses more RAM. A lower value '\
                        'uses less RAM but takes significantly longer to train.')
    return parser.parse_args()

# create or load a saved model
# returns the model and the epoch number (>1 if loaded from checkpoint)
def get_model(args, experiment_dir=None):
    
    epoch = 0
    
    if not experiment_dir:
        model = Sequential()
        for layer_index in range(args.num_layers):
            kwargs = dict() 
            kwargs['units'] = args.rnn_size
            # if this is the first layer
            if layer_index == 0:
                kwargs['input_shape'] = (args.window_size, OUTPUT_SIZE)
                if args.num_layers == 1:
                    kwargs['return_sequences'] = False
                else:
                    kwargs['return_sequences'] = True
                model.add(LSTM(**kwargs))
            else:
                # if this is a middle layer
                if not layer_index == args.num_layers - 1:
                    kwargs['return_sequences'] = True
                    model.add(LSTM(**kwargs))
                else: # this is the last layer
                    kwargs['return_sequences'] = False
                    model.add(LSTM(**kwargs))
            model.add(Dropout(args.dropout))
        model.add(Dense(OUTPUT_SIZE))
        model.add(Activation('softmax'))
    else:
        model, epoch = utils.load_model_from_checkpoint(experiment_dir)

    # these cli args aren't specified if get_model() is being
    # being called from sample.py
    if 'grad_clip' in args and 'optimizer' in args:
        kwargs = { 'clipvalue': args.grad_clip }

        if args.learning_rate:
            kwargs['lr'] = args.learning_rate

        # select the optimizers
        if args.optimizer == 'sgd':
            optimizer = SGD(**kwargs)
        elif args.optimizer == 'rmsprop':
            optimizer = RMSprop(**kwargs)
        elif args.optimizer == 'adagrad':
            optimizer = Adagrad(**kwargs)
        elif args.optimizer == 'adadelta':
            optimizer = Adadelta(**kwargs)
        elif args.optimizer == 'adam':
            optimizer = Adam(**kwargs)
        elif args.optimizer == 'adamax':
            optimizer = Adamax(**kwargs)
        elif args.optimizer == 'nadam':
            optimizer = Nadam(**kwargs)
        else:
            utils.log(
                'Error: {} is not a supported optimizer. Exiting.'.format(args.optimizer),
                True)
            exit(1)
    else: # so instead lets use a default (no training occurs anyway)
        optimizer = Adam()

    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model, epoch

def get_callbacks(experiment_dir, checkpoint_monitor='val_acc'):
    
    callbacks = []
    
    # save model checkpoints
    filepath = os.path.join(experiment_dir, 
                            'checkpoints', 
                            'checkpoint-epoch_{epoch:03d}-val_acc_{val_acc:.3f}.hdf5')

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

    return callbacks

def main():

    args = parse_args()
    args.verbose = True

    try:
        # get paths to midi files in --data_dir
        midi_files = [os.path.join(args.data_dir, path) \
                      for path in os.listdir(args.data_dir) \
                      if '.mid' in path or '.midi' in path]
    except OSError as e:
        log('Error: Invalid --data_dir, {} directory does not exist. Exiting.', args.verbose)
        exit(1)

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

    # write --message to experiment_dir
    if args.message:
        with open(os.path.join(experiment_dir, 'message.txt'), 'w') as f:
            f.write(args.message)
            utils.log('Wrote {} bytes to {}'.format(len(args.message), 
                os.path.join(experiment_dir, 'message.txt')), args.verbose)

    val_split = 0.2 # use 20 percent for validation
    val_split_index = int(float(len(midi_files)) * val_split)

    # use generators to lazy load train/validation data, ensuring that the
    # user doesn't have to load all midi files into RAM at once
    train_generator = utils.get_data_generator(midi_files[0:val_split_index], 
                                               window_size=args.window_size,
                                               batch_size=args.batch_size,
                                               num_threads=args.n_jobs,
                                               max_files_in_ram=args.max_files_in_ram)

    val_generator = utils.get_data_generator(midi_files[val_split_index:], 
                                             window_size=args.window_size,
                                             batch_size=args.batch_size,
                                             num_threads=args.n_jobs,
                                             max_files_in_ram=args.max_files_in_ram)

    model, epoch = get_model(args)
    if args.verbose:
        print(model.summary())

    utils.save_model(model, experiment_dir)
    utils.log('Saved model to {}'.format(os.path.join(experiment_dir, 'model.json')),
              args.verbose)

    callbacks = get_callbacks(experiment_dir)
    
    print('fitting model...')
    # this is a somewhat magic number which is the average number of length-20 windows
    # calculated from ~5K MIDI files from the Lakh MIDI Dataset.
    magic_number = 827
    start_time = time.time()
    model.fit_generator(train_generator,
                        steps_per_epoch=len(midi_files) * magic_number / args.batch_size, 
                        epochs=args.num_epochs,
                        validation_data=val_generator, 
                        validation_steps=len(midi_files) * 0.2 * magic_number / args.batch_size,
                        verbose=1, 
                        callbacks=callbacks,
                        initial_epoch=epoch)
    utils.log('Finished in {:.2f} seconds'.format(time.time() - start_time), args.verbose)

if __name__ == '__main__':
    main()