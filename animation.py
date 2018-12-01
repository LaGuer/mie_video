'''Module for localizing particle trajectories with tensorflow tracking.'''

import numpy as np
import pandas as pd
import trackpy as tp
import cv2
from tracker import tracker
from mie_video.editing import inflate, crop
import pylab as pl
from matplotlib import animation
from matplotlib.patches import Rectangle
import lab_io.h5video as h5
import features.circletransform as ct
from time import time


def oat(norm, frame_no, minmass=30.0, nfringes=25,
        diameter=100, topn=None):
    '''
    Use the orientational alignment transform
    on every pixel of an image and return features.'''
    t = time()
    circ = ct.circletransform(norm, theory='orientTrans')
    circ = circ / np.amax(circ)
    circ = h5.TagArray(circ, frame_no)
    feats = tp.locate(circ,
                      diameter,
                      minmass=minmass,
                      engine='numba',
                      topn=topn)
    feats['w'] = 400
    feats['h'] = 400
    features = np.array(feats[['x', 'y', 'w', 'h']])
    for idx, feature in enumerate(features):
        s = feature_extent(norm, (feature[0], feature[1]))
        features[idx][2] = s
        features[idx][3] = s
    print("Time to find {} features at frame {}: {}".format(features.shape[0],
                                                            frame_no,
                                                            time() - t))
    print("Mass of particles: {}".format(list(feats['mass'])))
    return features, circ


def feature_extent(norm, center, nfringes=20, maxrange=350):
    ravg, rstd = aziavg(norm, center)
    b = ravg - 1.
    ndx = np.where(np.diff(np.sign(b)))[0] + 1.
    if len(ndx) <= nfringes:
        return maxrange
    else:
        return float(ndx[nfringes])+30


def aziavg(data, center):
    x_p, y_p = center
    y, x = np.indices((data.shape))
    d = data.ravel()
    r = np.hypot(x - x_p, y - y_p).astype(np.int).ravel()
    nr = np.bincount(r)
    ravg = np.bincount(r, d) / nr
    avg = ravg[r]
    rstd = np.sqrt(np.bincount(r, (d - avg)**2) / nr)
    return ravg, rstd


class Animate(object):
    """Creates an animated video of particle tracking
    """

    def __init__(self, video, method='oat', transform=True,
                 dest='animation/test_mpl_anim_oat.avi', bg=None, **kwargs):
        self.frame_no = 0
        self.transform = transform
        self.video = video
        self.dest = dest
        self.fig, self.ax = pl.subplots(figsize=(8, 6))
        self.ax.set_xlabel('X [pixel]')
        self.ax.set_ylabel('Y [pixel]')
        self.cap = cv2.VideoCapture(self.video)
        self.im = None
        self.method = method
        self.rects = None
        if self.method == 'tf':
            self.trk = tracker.tracker()
        self.bg = bg

    def run(self):
        ret, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.bg is not None:
            frame = (frame.astype(float) - 13) / (self.bg - 13)
        if self.transform:
            features, frame = oat(frame, self.frame_no)
        if ret:
            self.im = self.ax.imshow(frame, interpolation='none',
                                     cmap=pl.get_cmap('gray'))
            self.anim = animation.FuncAnimation(self.fig,
                                                self.anim, init_func=self.init,
                                                blit=True, interval=50)
            self.anim.save(self.dest)
        else:
            print("Failed")

    def init(self):
        ret = False
        while not ret:
            ret, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.bg is not None:
            frame = (frame.astype(float) - 13) / (self.bg - 13)
        self.im.set_data(frame)
        return self.im,

    def anim(self, i):
        ret, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.bg is not None:
            frame = (frame.astype(float) - 13) / (self.bg - 13)
        self.frame_no += 1
        if ret:
            if self.method == 'tf':
                features = self.trk.predict(inflate(frame))
            elif self.method == 'oat':
                features, frame_ct = oat(frame, self.frame_no)
            else:
                raise(ValueError("method must be either \'oat\' or \'tf\'"))
            if self.rects is not None:
                for rect in self.rects:
                    rect.remove()
            self.rects = []
            for feature in features:
                x, y, w, h = feature
                rect = Rectangle(xy=(x - w/2, y - h/2),
                                 width=w, height=h,
                                 fill=False, linewidth=3, edgecolor='r')
                self.rects.append(rect)
                self.ax.add_patch(rect)
            if self.transform:
                self.im.set_array(frame_ct)
            else:
                self.im.set_array(frame)
        return self.im,


if __name__ == '__main__':
    import sys
    args = sys.argv
    anim = Animate(args[1], dest=args[2])
    anim.run()
