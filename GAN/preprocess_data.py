import mido
import numpy as np
import cv2
import os

time_dif = 12
time_of_image = 96
total_songs = 0


def dir_recursion(path):
    global total_songs
    global img_id
    global save_dir
    entries = os.listdir(path + '/')
    for item in entries:
        if os.path.isdir(path+'/'+item):
            dir_recursion(path+'/'+item)
        elif os.path.isfile(path+'/'+item):
            if item.endswith(".midi"):
                print(item)
                fname = path + "\\" + item
                image = open_messages(fname)
                total_time = len(image[0])
                save_dataset(image, total_time)
                total_songs = total_songs+1
        else:
            print("Warning! - dir_recursion bug.")


def save_dataset(image, total_time):
    global img_id
    global save_dir
    offset = 0
    while offset + (time_of_image-1) < total_time:
        temp_img = np.zeros((96, time_of_image))
        empty = True
        for y in range(offset, offset + time_of_image):
            for x in range(0, 96):
                temp_img[x][y - offset] = image[x][y]
                if image[x][y] != 0.0:
                    empty = False
        offset += time_of_image
        if empty:
            continue
        cv2.imwrite(save_dir+str(img_id)+'.png', temp_img)
        img_id += 1


def concatenate_tracks(image, track_image):
    for x in range(len(track_image)):
        for y in range(len(track_image[0])):
            image[x][y] = max(image[x][y], track_image[x][y])
    return image


def create_image(total_time, messages):
    image = np.zeros((96, total_time + time_dif * 8))  # + time_dif * 4))
    curr_time = 0
    k = 0
    notes = []
    notes_dict = {}
    terminate = False
    for msg in messages:
        k += msg['time']
        if terminate:
            break
        curr_time = msg['time']

        for curr_notes in notes_dict.keys():
            if terminate:
                break
            for time in range(k - curr_time, k):
                if k - curr_time > total_time:
                    terminate = True
                    break
                if notes_dict[str(curr_notes)] == 'on':     # indicates note on
                    image[96 - int(curr_notes)][time] = 255.0
                if notes_dict[str(curr_notes)] == 'hold':
                    image[96 - int(curr_notes)][time] = 127.0
                if notes_dict[str(curr_notes)] == 'on':        # == 255.0
                    notes_dict[str(curr_notes)] = 'hold'        # == 127.0
        if msg['type'] == 'note_on' and msg['velocity'] != 0:
            notes_dict[str(msg['note'])] = 'on'
        if msg['type'] == 'note_off':
            if str(msg['note']) in notes_dict:
                del notes_dict[str(msg['note'])]
    return image


def open_messages(fname):
    mid = mido.MidiFile(fname)
    track_list = mido.MidiTrack()
    track_list.append(mido.MetaMessage(type='set_tempo', tempo=600000, time=0))
    image = np.zeros((96, 0))

    for i, track in enumerate(mid.tracks):
        total_time = 0
        dt = 0
        switch = True
        messages = []
        for msg in track:
            if msg.type == 'note_on' or msg.type == 'note_off':
                if switch:
                    msg.time = 0
                    dt = 0
                    total_time = 0
                    switch = False
                if msg.velocity == 0 and msg.type == 'note_on':
                    msg.time = msg.time + dt
                    messages.append({'type': 'note_off', 'note': msg.note, 'velocity': msg.velocity,
                                     'time': round(msg.time * time_dif / mid.ticks_per_beat)})
                    track_list.append(mido.Message(type='note_off', note=msg.note, velocity=msg.velocity,
                                                   time=round(msg.time * time_dif / mid.ticks_per_beat)))
                    dt = 0
                else:
                    msg.time = msg.time + dt
                    messages.append({'type': msg.type, 'note': msg.note, 'velocity': msg.velocity,
                                     'time': round(msg.time * time_dif / mid.ticks_per_beat)})
                    track_list.append(mido.Message(msg.type, note=msg.note, velocity=msg.velocity,
                                                   time=round(msg.time * time_dif / mid.ticks_per_beat)))
                    dt = 0
            else:
                dt += msg.time
            total_time += msg.time
        total_time = round(total_time * time_dif / mid.ticks_per_beat)
        track_image = create_image(total_time, messages)    # create an image of notes for the track
        if len(image[0]) > len(track_image[0]):
            image = concatenate_tracks(image, track_image)
        else:
            image = concatenate_tracks(track_image, image)

    for y in range(len(image[0])):
        for x in range(len(image)):
            if image[x][y] == 127.0:
                if image[x][y + 1] == 255.0:
                    image[x][y] = 0.0
                else:
                    image[x][y] = 255.0
    return image


def preprocess():
    whole_seq = []
    path = 'GAN\\data\\preproc\\'
    entries = os.listdir(path+'\\')
    save_dir = path
    img_id = 0
    dir_recursion(path)
    print("MIDI to image conversion for the "+" MIDI files complete.")
    print("Total songs: ", total_songs)
    print("Total images: ", img_id)

