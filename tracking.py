'''Module for localizing particle trajectories with tensorflow tracking.'''

from mie_video.utilities.h5video import TagArray
from mie_video.utilities.circletransform import circletransform
import trackpy as tp
import numpy as np
from time import time
import cv2


def oat(norm, frame_no=None,
        locate_params={'diameter': 31,
                       'minmass': 30.},
        nfringes=25,
        maxrange=300.,
        crop_threshold=None):
    '''
    Use the orientational alignment transform
    on every pixel of an image and return features.'''
    t = time()
    circ = circletransform(norm, theory='orientTrans')
    circ = circ / np.amax(circ)
    circ = TagArray(circ, frame_no=frame_no)
    feats = tp.locate(circ,
                      engine='numba',
                      **locate_params)
    feats['w'] = 400.
    feats['h'] = 400.
    features = np.array(feats[['x', 'y', 'w', 'h']])
    for idx, feature in enumerate(features):
        s = feature_extent(norm, (feature[0], feature[1]),
                           nfringes=nfringes,
                           maxrange=maxrange)
        if crop_threshold is not None and s > crop_threshold:
            s = crop_threshold
        features[idx][2] = s
        features[idx][3] = s
    msg = "Time to find {} features".format(features.shape[0])
    if type(frame_no) is int:
        msg += " at frame {}".format(frame_no)
    msg += ": {:02}".format(time() - t)
    print(msg)
    print("Last feature size: {}".format(s))
    return features, circ


def feature_extent(norm, center, nfringes=20, maxrange=550.):
    ravg, rstd = aziavg(norm, center)
    b = ravg - 1.
    ndx = np.where(np.diff(np.sign(b)))[0] + 1.
    if len(ndx) <= nfringes:
        return maxrange
    else:
        return float(ndx[nfringes])


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


if __name__ == '__main__':
    data = cv2.imread('sample.png', cv2.IMREAD_GRAYSCALE).astype(float)
    data /= np.mean(data)
    oat(data)
