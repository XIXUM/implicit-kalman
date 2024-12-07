from tkinter import Label, Button, Tk, Entry, END
from PIL import ImageTk, Image
import ffmpeg
import numpy as np
from pathlib import Path



class Video:

    frame_number = None
    video_info = None
    width = None
    height = None
    num_frames = None
    buffer = None
    path = None

    def __init__(self, videoPath : Path):
        self.frame_number = 0
        self.path = videoPath.absolute()
        probe = ffmpeg.probe(str(self.path))
        self.video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        self.width = int(self.video_info['width'])
        self.height = int(self.video_info['height'])
        self.num_frames = int(self.video_info['nb_frames'])

    def loadVideo(self):
        out, err = (
            ffmpeg
                .input(str(self.path))
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True)
        )
        if out != None:
            video = (
                np
                    .frombuffer(out, np.uint8)
                    .reshape([-1, self.height, self.width, 3])
            )
            return video
        else:
            return None

    def showVideo(self):
        # if self.buffer == None:
        #    self.buffer = self.loadVideo()
        videoImage = VideoWindow(self)
        videoImage.createWindow()
        return videoImage

    @property
    def videoBuffer(self):
        if not isinstance(self.buffer, np.ndarray):  #if ==None does not work here
            self.buffer = self.loadVideo()
        return self.buffer

    @property
    def videoFrame(self, frame):
        self.frame_number = frame
        return self.videoBuffer[frame]

    @property
    def currentFrame(self):
        return self.videoBuffer[self.frame_number]

    @property
    def nextFrame(self):
        if self.frame_number < self.num_frames - 1:
            self.frame_number += 1
        return self.currentFrame

    @property
    def previousFrame(self):
        if self.frame_number > 0:
            self.frame_number -= 1
        return self.currentFrame



class VideoWindow:

    video = None
    root = None
    v_label = None

    def __init__(self, video: Video):
        self.video = video
        # self.createWindow()

    def forward(self):
        frame = ImageTk.PhotoImage(Image.fromarray(self.video.nextFrame, mode = "RGB"))
        self.update(frame)
        self.root.update()

    def back(self):
        frame = ImageTk.PhotoImage(Image.fromarray(self.video.previousFrame, mode = "RGB"))
        self.update(frame)
        self.root.update()

    def update(self, frameObj):
        self.v_label.configure(image=frameObj)
        self.v_label.image = frameObj
        self.v_label.update()
        self.e_num.delete(0, END)
        self.e_num.insert(0, str(self.video.frame_number))

    def on_closing(self):
        self.root.quit()


    def createWindow(self):
        self.root = Tk()
        self.root.title("ImageViewer")
        # root.iconbitmap('logo.ico')

        # video = self.loadVideo()
        # video = self.video.videoBuffer

        im_vid = Image.fromarray(self.video.currentFrame, mode="RGB")
        v_vid = ImageTk.PhotoImage(im_vid)
        self.v_label = Label(image=v_vid)
        self.v_label.pack()

        self.v_label.grid(row=0, column=0, columnspan=3)
        self.b_back = Button(self.root, text="<<", command=lambda: self.back())
        self.b_quit = Button(self.root, text="close", command=self.root.quit)
        self.b_forward = Button(self.root, text=">>", command=lambda: self.forward())
        self.l_num = Label(text=f"Frame:")
        self.e_num = Entry(self.root, width=4)
        self.e_num.insert(0,str(self.video.frame_number))


        self.b_back.grid(row=1, column=0)
        self.b_quit.grid(row=1, column=1)
        self.b_forward.grid(row=1, column=2)
        self.l_num.grid(row=2, column=0, columnspan=2)
        self.e_num.grid(row=2, column=2)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        return self.root

    def run(self):
        self.root.mainloop()
