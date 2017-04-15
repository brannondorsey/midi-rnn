import os
import pretty_midi
import numpy as np
from multiprocessing import Pool as ThreadPool

def log(message, verbose):
	if verbose:
		print('[*] {}'.format(message))

def parse_midi(path):
    midi = None
    with open(path, 'r') as f:
        try:
            midi = pretty_midi.PrettyMIDI(f)
            midi.remove_invalid_notes()
        except:
            pass
    return midi

def get_percent_monophonic(pm_instrument_roll):
    mask = pm_instrument_roll.T > 0
    notes = np.sum(mask, axis=1)
    n = np.count_nonzero(notes)
    single = np.count_nonzero(notes == 1)
    if single > 0:
        return float(single) / float(n)
    elif single == 0 and n > 0:
        return 0.0
    else: # no notes of any kind
        return 0.0
    
def filter_monophonic(pm_instruments, percent_monophonic=0.99):
    return [i for i in pm_instruments if \
            get_percent_monophonic(i.get_piano_roll()) >= percent_monophonic]


# if the experiment dir doesn't exist create it and its subfolders
def create_experiment_dir(experiment_dir, verbose=False):
    
    # if the experiment directory was specified and already exists
    if experiment_dir != 'experiments/default' and \
       os.path.exists(experiment_dir):
    	# raise an error
    	raise Exception('Error: Invalid --experiemnt_dir, {} already exists' \
    		            .format(experiment_dir))

    # if the experiment directory was not specified, create a new numeric folder
    if experiment_dir == 'experiments/default':
    	
    	experiments = os.listdir('experiments')
    	experiments = [dir_ for dir_ in experiments \
    	               if os.path.isdir(os.path.join('experiments', dir_))]
    	
    	most_recent_exp = 0
    	for dir_ in experiments:
    		try:
    			most_recent_exp = max(int(dir_), most_recent_exp)
    		except ValueError as e:
    			# ignrore non-numeric folders in experiments/
    			pass

    	experiment_dir = os.path.join('experiments', 
    		                          str(most_recent_exp + 1).rjust(2, '0'))

    os.mkdir(experiment_dir)
    log('Created experiment directory {}'.format(experiment_dir), verbose)
    os.mkdir(os.path.join(experiment_dir, 'checkpoints'))
    log('Created checkpoint directory {}'.format(os.path.join(experiment_dir, 'checkpoints')),
    	verbose)
    os.mkdir(os.path.join(experiment_dir, 'tensorboard-logs'))
    log('Created log directory {}'.format(os.path.join(experiment_dir, 'tensorboard-logs')), 
    	verbose)

    return experiment_dir

# load data with a lazzy loader
def get_data_generator(midi_paths, 
                       window_size=20, 
                       batch_size=32,
                       num_threads=8):

    load_size = 170 # this refers to files

     # load midi data
    pool = ThreadPool(num_threads)

    load_index = 0

    while True:
        
        load_files = midi_paths[load_index:load_index + load_size]
        # print('length of load files: {}'.format(len(load_files)))
        load_index = (load_index + load_size) % len(midi_paths)

        # print('loading large batch: {}'.format(load_size))
        # print('Parsing midi files...')
        # start_time = time.time()
        parsed = pool.map(parse_midi, load_files)
        # print('Finished in {:.2f} seconds'.format(time.time() - start_time))
        # print('parsed, now extracting data')
        data = _windows_from_monophonic_instruments(parsed, window_size)
        batch_index = 0
        while batch_index + batch_size < len(data[0]):
            # print('getting data...')
            # print('yielding small batch: {}'.format(batch_size))
            
            res = (data[0][batch_index: batch_index + batch_size], 
                   data[1][batch_index: batch_index + batch_size])
            yield res
            batch_index = batch_index + batch_size
        
        # probably unneeded but why not
        del parsed # free the mem
        del data # free the mem

def save_model(model, model_dir):
    with open(os.path.join(model_dir, 'model.json'), 'w') as f:
        f.write(model.to_json())

# returns X, y data windows from all monophonic instrument
# tracks in a pretty midi file
def _windows_from_monophonic_instruments(midi, window_size):
    X, y = [], []
    for m in midi:
        if m is not None:
            melody_instruments = filter_monophonic(m.instruments, 1.0)
            for instrument in melody_instruments:
                if len(instrument.notes) > window_size:
                    windows = _encode_sliding_windows(instrument, window_size)
                    for w in windows:
                        X.append(w[0])
                        y.append(w[1])
    return (np.asarray(X), np.asarray(y))

# one-hot encode a sliding window of notes from a pretty midi instrument.
# This approach uses the piano roll method, where each step in the sliding
# window represents a constant unit of time (fs=4, or 1 sec / 4 = 250ms).
# This allows us to encode rests.
# expects pm_instrument to be monophonic.
def _encode_sliding_windows(pm_instrument, window_size):
    
    roll = np.copy(pm_instrument.get_piano_roll(fs=4).T)

    # trim beginning silence
    summed = np.sum(roll, axis=1)
    mask = (summed > 0).astype(float)
    roll = roll[np.argmax(mask):]
    
    # transform note velocities into 1s
    roll = (roll > 0).astype(float)
    
    # calculate the percentage of the events that are rests
    # s = np.sum(roll, axis=1)
    # num_silence = len(np.where(s == 0)[0])
    # print('{}/{} {:.2f} events are rests'.format(num_silence, len(roll), float(num_silence)/float(len(roll))))

    # append a feature: 1 to rests and 0 to notes
    rests = np.sum(roll, axis=1)
    rests = (rests != 1).astype(float)
    roll = np.insert(roll, 0, rests, axis=1)
    
    windows = []
    for i in range(0, roll.shape[0] - window_size - 1):
        windows.append((roll[i:i + window_size], roll[i + window_size + 1]))
    return windows