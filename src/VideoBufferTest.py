#!/usr/bin/env python
# from ipywidgets import interact
from matplotlib import pyplot as plt
from tkinter import *
from PIL import ImageTk, Image
import ffmpeg
# import ipywidgets as widgets
import numpy as np

probe = ffmpeg.probe('in.mp4')
video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
width = int(video_info['width'])
height = int(video_info['height'])
num_frames = int(video_info['nb_frames'])

def loadVideo():
    out, err = (
        ffmpeg
        .input('in.mp4')
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True)
    )
    video = (
        np
        .frombuffer(out, np.uint8)
        .reshape([-1, height, width, 3])
    )
    return video

def forward():
    return

def back():
    return


if __name__ == '__main__':
    root = Tk()
    root.title("ImageViewer")
    #root.iconbitmap('logo.ico')

    video = loadVideo()

    # image = Image.open("testImg.jpg")
    # arr = np.asarray(image)
    # v_img = ImageTk.PhotoImage(image)

    im_vid = Image.fromarray(video[5], mode = "RGB")
    v_vid = ImageTk.PhotoImage(im_vid)
    v_label = Label(image = v_vid)
    v_label.pack()

    v_label.grid(row = 0, column = 0, columnspan= 3)

    b_back =  Button(root, text = "<<", command = )



    root.mainloop()