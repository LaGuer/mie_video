from mie_video.localization.localize import localize
import cv2
from matplotlib import animation
from matplotlib.patches import Rectangle
import pylab as pl
import numpy as np


class Animate(object):
    """Creates an animated video of particle tracking
    """

    def __init__(self, video_fn, dest='test_anim.avi', save=True,
                 method='circletransform', linked_df=None, bg=1.,
                 **kwargs):
        # I/O
        self.video_fn = video_fn
        self.dest = dest
        self.save = save
        # Localization params
        self.method = method
        self.df = linked_df
        self.frame_no = 0
        # Plotting
        self.fig, self.ax = pl.subplots(figsize=(8, 6))
        self.ax.set_xlabel('X [pixel]')
        self.ax.set_ylabel('Y [pixel]')
        self.im = None
        self.rects = None
        # Image processing
        self.bg = bg
        self.cap = cv2.VideoCapture(self.video_fn)

    def process(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float)
        if self.bg == 1.:
            frame /= np.mean(frame)
        return frame / self.bg

    def run(self):
        ret, frame = self.cap.read()
        if ret:
            frame = self.process(frame)
            self.im = self.ax.imshow(frame, interpolation='none',
                                     cmap=pl.get_cmap('gray'))
            self.anim = animation.FuncAnimation(self.fig,
                                                self.anim,
                                                init_func=self.init,
                                                blit=True,
                                                interval=50)
            self.anim.save(self.dest)
        else:
            print("Failed")

    def init(self):
        ret = False
        while not ret:
            ret, frame = self.cap.read()
        frame = self.process(frame)
        self.im.set_data(frame)
        return self.im,

    def anim(self, i):
        ret, frame = self.cap.read()
        self.frame_no += 1
        if ret:
            frame = self.process(frame)
            if self.df is None:
                if self.method == 'circletransform':
                    features, frame_ct = localize(frame, self.frame_no,
                                                  nfringes=25)
                else:
                    raise(ValueError("method must be either \'circletransform\' or \'tf\'"))
            else:
                features = self.df.loc[self.df.frame == self.frame_no]
                features = np.array(self.df[['x', 'y', 'w', 'h']])
            if self.rects is not None:
                for rect in self.rects:
                    rect.remove()
            self.rects = []
            for feature in features:
                x, y, w, h = feature
                rect = Rectangle(xy=(x - w/2, y - h/2),
                                 width=w, height=h,
                                 fill=False, linewidth=1, edgecolor='r')
                self.rects.append(rect)
                self.ax.add_patch(rect)
            self.im.set_array(frame)
        return self.im,


if __name__ == '__main__':
    import sys
    args = sys.argv
    if len(args) > 1:
        anim = Animate(args[1])
    else:
        anim = Animate("animation/example.avi")
    anim.run()
