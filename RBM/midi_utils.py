import mido, glob, random, math
import numpy as np

#For reproducibility
random.seed(42)

def make_midi(encoded_midi, instrument, timescale, filename):
    '''
    Take encoded MIDI data and output a proper MIDI file.
    '''
    #Initialize output file.
    outfile = mido.MidiFile()
    track = mido.MidiTrack()
    outfile.tracks.append(track)
    track.append(mido.Message("program_change", program = instrument))

    #Set variables.
    encoded_midi = np.array(encoded_midi)
    volume = 80
    delta = 0
    state_last = np.zeros([1,128])
    d_on_notes = np.array([])
    d_off_notes = np.array([])
    d_off_notes_last = np.array([])


    #Loop through the encoded MIDI.
    for state in encoded_midi:
        
        #Find when to write "on" notes and "off" notes to midi file.
        on_notes = np.where((state - state_last) == 1)[0]
        off_notes = np.where((state - state_last) == -1)[0]

        #Collect all "on" and "off" notes for current time delta
        d_on_notes = np.concatenate([d_on_notes, on_notes])
        d_off_notes = np.concatenate([d_off_notes, off_notes])

        #Append a silent note with a new delta to allow for intact chords.
        if np.array_equal(state,state_last):
            delta += 1
        elif delta > 0:
            #Initiate "chord" in MIDI track.
            track.append(mido.Message("note_on", note = 0,
                                      velocity = 0, time = delta * timescale))
            delta = 0

            #Register "on" and "off" notes in MIDI file.
            for i, notes in enumerate([d_on_notes, d_off_notes_last]):
                
                #Write MIDI message to track.
                for note in list(notes):
                    message = "note_on" if not(i) else "note_off"
                    track.append(mido.Message(message,
                                              note = int(note),
                                              velocity = volume,
                                              time = 0))

            #Reset d_on_notes and d_off_notes arrays.
            d_on_notes = np.array([])
            d_off_notes_last = d_off_notes.copy()
            d_off_notes = np.array([])

        #Update previous state and loop around
        state_last = state.copy()

    #Save MIDI file.
    outfile.save(filename)

def encode_midi(midifile):
    '''
    Encode MIDI file for training.
    MIDI notes are binary encoded at each time point and gathered.
    '''
    #Read MIDI file.
    midfile = mido.MidiFile(midifile)
    
    #Initialize variables.
    encoded_midi = []
    last_moment = 32 #Quarter note in ticks.

    #Iterate over each track.
    for i, track in enumerate(midfile.tracks):

        #Initialize an empty time state.
        #This list containts 128 elements - one for each possible MIDI note.
        state_last = [0] * 128

        #Encode for each message in track.
        for message in track[:-1]:
            state = state_last
           #Encode note_on events as 1 for specified note.
            if message.type == "note_on":
                state[np.clip(message.note,0,127)] = 1
                #Encode note 0 volume drops as 0.
                if message.velocity == 0:
                    state[np.clip(message.note,0,127)] = 0
            #Encode note_off events as 0.
            elif message.type == "note_off":
                state[np.clip(message.note,0,127)] = 0
            
            #If at a new MIDI delta, append last time state to encoded list.
            if message.time != 0:
                last_states = [state_last] * (last_moment // 32)
                encoded_midi.extend(last_states.copy())
                last_moment = int(pow(2, math.ceil(math.log(message.time, 2))))
            
            #Update last state.
            state_last = state.copy()

        #Pad encoded midi for later "on" and "off" message encoding.
        encoded_midi.append([0] * 128)

    return encoded_midi

def load_midis(midi_dir):
    '''
    Load and binary encode all midis in a given directory.
    '''  
    #Shuffle file list for random binning into test or train sets.
    file_list = glob.glob(midi_dir + "*.mid*")
    random.shuffle(file_list)
    train_songs = []

    #Place encoded MIDI files in training set.
    for f in file_list:
        song = np.array(encode_midi(f))
        train_songs.append(song)
    return train_songs
