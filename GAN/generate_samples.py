import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
import mido
import cv2
import os

i = 0
threshold = 0.55
model_str = "generator_model_144.h5"
model_str = " "


def clean_image(image):
    # threshold = .55
    xmax, xmin = image.max(), image.min()
    image = (image - xmin) / (xmax - xmin)
    max = image.max()
    # print(max)
    for x in range(len(image)):
        for y in range(len(image[0])):
            if image[x][y] < threshold:
                image[x][y] = 0.0
            else:
                image[x][y] = 255.0

    for x in range(len(image)):
        for y in range(len(image[0]) - 1):
            if y != 0 and y != len(image[0]):
                if image[x][y] == 255.0 and (
                    image[x][y - 1] == 0.0 and image[x][y + 1] == 0
                ):
                    image[x][y] = 0.0
    return image


def image2midi(image, fname):
    mid = mido.MidiFile()
    track_list = mido.MidiTrack()
    track_list.append(mido.MetaMessage(type="set_tempo", tempo=600000, time=0))
    time = 0
    for y in range(len(image[0])):
        for x in range(len(image)):
            if y != 0:
                if (
                    image[x][y] == 255 and image[x][y - 1] == 0
                ):  # indicator that a new note is played
                    track_list.append(
                        mido.Message(
                            type="note_on", note=96 - x, velocity=100, time=time
                        )
                    )
                    time = 0
                if (
                    image[x][y] == 0 and image[x][y - 1] == 255
                ):  # indicator a note is set off
                    track_list.append(
                        mido.Message(
                            type="note_off", note=96 - x, velocity=100, time=time
                        )
                    )
                    time = 0
        time += 1
    mid.tracks.append(track_list)
    mid.ticks_per_beat = 12
    mid.save("results\\midi\\" + fname + ".mid")
    return


# z = np.random.uniform(-1, 1, (64, 100))

model = keras.models.load_model("model_dir" + "\\" + model_str)
latent_dim = 100
n_samples = 1000
x_input = np.random.randn(latent_dim * n_samples)
x_input = x_input.reshape(n_samples, latent_dim)
imgs = model.predict(x_input)
imgs = imgs * 0.5 + 0.5


if not os.path.exists("results"):
    os.mkdir("results")

if not os.path.exists("results\\images\\"):
    os.mkdir("results\\images\\")

if not os.path.exists("results\\midi\\"):
    os.mkdir("results\\midi\\")

if not os.path.exists("results\\images\\"):
    os.mkdir("results\\images\\")

if not os.path.exists("results\\midi\\"):
    os.mkdir("results\\midi\\")

# plt.imshow(imgs[0, :, :, 0], cmap="gray")
# plt.imsave("results\\someimage.png", imgs[0, :, :, 0])
# cv2.imwrite("results\\someimage_2.png", imgs[0, :, :, 0])
for i in range(len(imgs)):
    image = clean_image(imgs[i])
    cv2.imwrite("results\\images\\" + str(i) + ".png", image)
    image2midi(image, str(i))

# print("no")

################################################################################################
