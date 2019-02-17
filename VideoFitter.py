'''Class to fit trajectories in a video to Lorenz-Mie theory.'''

import mie_video.editing as editing
from CNNLorenzMie.Localizer import Localizer
from CNNLorenzMie.Estimator import Estimator
from CNNLorenzMie.nodoubles import nodoubles
from mie_video.animation import Animate
from pylorenzmie.theory.Feature import Feature
from pylorenzmie.detection.localize import localize, feature_extent
from pylorenzmie.lmtool.LMTool import LMTool
from pylorenzmie.theory.Instrument import Instrument, coordinates
from PyQt5 import QtWidgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trackpy as tp
import json
import cv2
import os
import sys
from time import time
import signal
import logging
logger = logging.getLogger()
logger.setLevel(logging.WARNING)
logger.setLevel(logging.INFO)


def handler(signum, frame):
    raise(TimeoutError("Function timed out."))


signal.signal(signal.SIGALRM, handler)
signal.alarm(0)

dir = os.path.expanduser('~/python/CNNLorenzMie')
keras_model_path = os.path.join(dir, 'keras_models/predict_stamp_auto.h5')
with open(os.path.join(dir, 'keras_models/predict_stamp_auto.json')) as f:
    config_json = json.load(f)


class VideoFitter(object):

    def __init__(self, fn,
                 guesses={'r_p': None, 'n_p': None, 'a_p': None},
                 frame_size=(1024, 1280),
                 background=1.,
                 localized_df=None,
                 detection_method='cnn',
                 estimator=False,
                 localizer=None):
        """
        Args:
            fn: filename
            guesses: initial guesses of particle parameters for
                     first frame of video.
        Keywords:
            background: background image or filename of a background video
            detection_method: 'circletransform': circletransform
                              'cnn': CNNLorenzMie localizer
            localized_df: input if localized_df has already been calculated and saved
        """
        self.instrument = Instrument(wavelength=.447,
                                     magnification=.048,
                                     n_m=1.340,
                                     dark_count=13.)
        self._fit_dfs = None
        self._localized_df = None
        self._frame_size = frame_size
        self._fn = os.path.expanduser(fn)
        self._init_background(background)
        self._init_fitting(guesses, estimator)
        self._init_localization(localized_df, detection_method, localizer)

    def _init_background(self, background):
        """
        Initialize background from video file or np.ndarray
        """
        self.forced_crop = None
        if type(background) is str and background[-4:] == '.avi':
            self.instrument.background = editing.background(background,
                                                            shape=self._frame_size)
        elif type(background) is np.ndarray or background == 1.:
            self.instrument.background = background
        else:
            raise(ValueError("background must be .avi filename, np.ndarray, or 1."))

    def _init_localization(self, localized_df, detection_method, localizer):
        """
        Initialize parameters for detection.

        If using the orientational alignment transform,
        adjust parameters nfringes, maxrange, tp_params, and
        crop_threshold as needed.
        """
        self._detection_method = detection_method
        self._trajectories = None
        self.nfringes = 25
        self.maxrange = 400.
        self.crop_threshold = 350.
        self._set_localized_df(localized_df)
        if detection_method == 'circletransform':
            self.tp_params = {'diameter': 31, 'topn': 1}
        elif detection_method == 'cnn':
            self._localizer = Localizer() if (localizer is None) else localizer

    def _init_fitting(self, guesses, estimator):
        """
        Initialize parameters for fitting.
        """
        if estimator is True:
            self._estimator = Estimator(model_path=keras_model_path,
                                        config_file=config_json,
                                        instrument=self.instrument)
            self._estimator.params_range['n_p'] = [1.40, 1.46]
        elif type(estimator) == Estimator:
            self._estimator = estimator
        else:
            self._estimator = None
        self.fitter = Feature(**guesses)
        self.fitter.model.instrument = self.instrument

    @property
    def localized_df(self):
        """
        DataFrame of particle trajectories
        
        Columns:
            ['x', 'y', 'w', 'h', 'frame', 'particle']
        """
        return self._localized_df

    def _set_localized_df(self, localized_df):
        if type(localized_df) is pd.DataFrame:
            self._localized_df = localized_df[['x', 'y', 'w', 'h', 'frame', 'particle']]
            self._fit_dfs = [None for _ in range(len(self.trajectories))]
            logging.info(str(len(self.trajectories)) + " trajectories found.")
        else:
            self._localized_df = None

    @property
    def trajectories(self):
        """
        List where an element at index i is a DataFrame
        for particle i's localized trajectory.
        
        Columns:
            ['x', 'y', 'w', 'h', 'frame']
        """
        return self._separate(self.localized_df)

    @property
    def fit_dfs(self):
        """
        List where an element at index i is a DataFrame
        for particle i's fitted trajectory.
        
        Columns:
            ['x_p', 'y_p', 'z_p', 'a_p', 'n_p', 'k_p', 'frame', 'redchi']
        """
        return self._fit_dfs

    @property
    def particle(self):
        '''Model for next guesses'''
        return self.fitter.model.particle

    def localize(self, maxframe=None, minframe=None):
        '''
        Returns DataFrame of particle parameters in each frame
        of a video linked with their trajectory index
        '''
        if type(self.localized_df) is pd.DataFrame: return
        cap = cv2.VideoCapture(self._fn)
        if type(minframe) == int:
            cap.set(1, minframe)
            frame_no = minframe
        else:
            frame_no = 0
        cols = ['x', 'y', 'w', 'h', 'frame']
        self.unlinked_df = pd.DataFrame(columns=cols)
        while(cap.isOpened()):
            if frame_no == maxframe: break
            ret, frame = cap.read()
            if ret is False: break
            t = time()
            if self._detection_method == 'circletransform':
                norm = self._process(frame)
                features, circ = localize(norm,
                                          frame_no=frame_no,
                                          locate_params=self.tp_params,
                                          maxrange=self.maxrange,
                                          nfringes=self.nfringes)
                s = features[-1][2]
            elif self._detection_method == 'cnn':
                norm = self._process(frame)
                features = []
                feats = self._localizer.predict([editing.inflate(norm)])
                feats = nodoubles(feats, tol=15)
                bboxs = []
                for feature in feats[0]:
                    xc, yc, w, h = feature['bbox']
                    s = feature_extent(norm/np.mean(norm), (xc, yc),
                                       nfringes=self.nfringes,
                                       maxrange=self.maxrange)
                    if s > self.crop_threshold:
                        s = self.crop_threshold
                    features.append([xc, yc, s, s])
                    bboxs.append(s)
            else:
                raise(ValueError("method must be either \'circletransform\' or \'cnn\'"))
            msg = "Time to find {} features".format(len(features))
            msg += " at frame {}".format(frame_no)
            msg += ": {:.2f}".format(time() - t)
            logging.info(msg + "\nFeature sizes: {}".format(bboxs))
            for feature in features:
                feature = np.append(feature, frame_no)
                self._write(feature, self.unlinked_df)
            frame_no += 1
        cap.release()
        self._set_localized_df(tp.link(self.unlinked_df,
                                       search_range=20,
                                       memory=3,
                                       pos_columns=['y', 'x']))

    def fit(self, trajectory_no, maxframe=None, minframe=None):
        '''
        None DataFrame of fitted parameters in each frame
        for a given trajectory.
        
        Args:
            trajectory_no: index of particle trajectory in self.trajectories.
        '''
        idx = trajectory_no
        cap = cv2.VideoCapture(self._fn)
        if type(minframe) == int:
            cap.set(1, minframe)
            frame_no = minframe
        else:
            frame_no = 0
        cols = np.append(list(self.fitter._keys), ['frame', 'redchi'])
        self.fit_dfs[idx] = pd.DataFrame(columns=cols)
        p_df = self.trajectories[idx]
        while(cap.isOpened()):
            if frame_no == maxframe: break
            ret, frame = cap.read()
            if ret is False:
                logging.warning("Frame not read.")
                break
            norm = self._process(frame)
            # Crop feature of interest.
            feats = p_df.loc[p_df['frame'] == frame_no]
            if len(feats) == 0:
                logging.warning('No particle found in frame {}'.
                                format(frame_no))
                frame_no += 1
                continue
            x, y, w, h, frame, particle = feats.iloc[0]
            # Set initial guesses for z_p, a_p, n_p
            if self._estimator is not None:
                stamp = self._crop(norm, x, y, 201, 201)
                data = self._estimator.predict(img_list=[stamp])
                self.particle.z_p = data['z_p'][0]
                self.particle.a_p = data['a_p'][0]
                self.particle.n_p = data['n_p'][0]
                msg = "Estimator guesses: z_p={:.2f}, a_p={:.2f}, n_p={:.2f}"
                logging.info(msg.format(self.particle.z_p,
                                        self.particle.a_p,
                                        self.particle.n_p))
            feature = self._crop(norm, x, y, w, h)
            feature /= np.mean(feature)
            # Set initial guesses for center
            xc, yc = feature[0].size / 2, feature[1].size / 2
            self.particle.x_p, self.particle.y_p = (xc, yc)
            # Fit frame
            signal.alarm(60)
            start = time()
            try:
                self.fitter.model.coordinates = coordinates(feature.shape)
                self.fitter.data = (feature.reshape(feature.size))
                result = self.fitter.optimize()
            except TimeoutError:
                logging.warning("Fit timed out.")
                continue
            signal.alarm(0)
            p = result.params
            fn = self._fn.split('/')[-1]
            msg = "LMFit results: z_p={:.2f}, a_p={:.2f}, n_p={:.2f}".format(p['z_p'].value,
                                                                             p['a_p'].value,
                                                                             p['n_p'].value)
            msg += "\n{} time to fit frame {}: {:.2f}".format(fn[:-4],
                                                              frame_no,
                                                              time() - start)
            msg += "\nredchi={:.2f}".format(result.redchi)
            logging.info(msg)
            # Add fit to dataset
            row = [frame_no, result.redchi]
            for key in reversed(self.fitter._keys):
                value = result.params[key].value
                value = value - xc + x if key == 'x' else value
                value = value - yc + y if key == 'y' else value
                row.insert(0, result.params[key].value)
            self._write(row, self.fit_dfs[idx])
            frame_no += 1
        cap.release()

    def tool(self, frame_no=0, trajectory_no=0):
        '''
        Uses LMTool to find good initial guesses for fit.
        '''
        p_df = self.trajectories[trajectory_no]
        if frame_no > max(p_df.index) or frame_no < min(p_df.index):
            raise(IndexError("Trajectory not found in frame {} for particle {}"
                             .format(frame_no, trajectory_no)))
        cap = cv2.VideoCapture(self._fn)
        cap.set(1, frame_no)
        ret, frame = cap.read()
        if not ret:
            logging.warning("Frame not read.")
            return
        background = self.instrument.background
        app = QtWidgets.QApplication(sys.argv)
        lmtool = LMTool(data=self._process(frame),
                        normalization=background)
        lmtool.show()
        sys.exit(app.exec_())

    def compare(self, guesses, trajectory_no=0, frame_no=0):
        '''
        Plot guessed image vs. image of a trajectory at a given frame
        
        Args:
            guesses: list of parameters ordered
                     [x, y, z, a_p, n_p, n_m, mpp, lamb]
        Keywords:
            trajectory: index of trajectory in self.trajectory
            frame_no: index of frame to test
        Returns:
            Raw frame from camera
        '''
        p_df = self.trajectories[trajectory_no]
        if frame_no > max(p_df['frame']) and frame_no < min(p_df['frame']):
            raise(IndexError("Trajectory not found in frame {} for particle {}"
                             .format(frame_no, trajectory_no)))
        cap = cv2.VideoCapture(self._fn)
        cap.set(1, frame_no)
        ret, frame = cap.read()
        if not ret:
            logging.warning("Frame not read.")
            return
        norm = self._process(frame)
        # Crop feature
        feats = p_df.loc[p_df['frame'] == frame_no]
        x, y, w, h, frame, particle = feats.iloc[0]
        feature = self._crop(norm, x, y, w, h)
        # Generate guess
        hol = self.fitter.model.hologram().reshape(feature.shape)
        residual = (feature - hol)
        # Plot and return normalized image
        plt.imshow(np.hstack([feature, hol, residual]), cmap='gray')
        plt.show()
        cap.release()
        return norm

    def animate(self):
        '''
        Used after localizing to show tracking animation.
        '''
        if self.localized_df is None:
            raise UserWarning("Error: run localize() before animate().")
        elif type(self.localized_df) is pd.DataFrame:
            self.animation = Animate(self._fn, linked_df=self.localized_df)
            self.animation.run()
        else:
            raise ValueError("Error: localized_df is unknown datatype.")

    def _write(self, row, df):
        last_line = df.index.max()
        if last_line is np.nan:
            df.loc[0] = row
        else:
            df.loc[last_line + 1] = row

    def _process(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float)
        median = np.median(frame)
        dc = self.instrument.dark_count
        bg = self.instrument.background
        norm = [(frame - dc) / (bg - dc)]*median if bg != 1. else frame
        if type(self.forced_crop) is list or type(self.forced_crop) is tuple:
            if len(self.forced_crop) == 4:
                xc, yc, w, h = self.forced_crop
                norm = self._crop(norm, xc, yc, w, h, square=False)
            else:
                logging.warning("Forced crop must be (xc, yc, w, h)")
        return norm

    def _crop(self, image, xc, yc, w, h, square=True):
        cropped_image = image[int(yc-h/2): int(yc+h/2),
                              int(xc-w/2): int(xc+w/2)].astype(float)
        xdim, ydim = cropped_image.shape
        if square is True:
            if xdim == ydim:
                return cropped_image
            if xdim > ydim:
                cropped_image = cropped_image[:-1, :]
            else:
                cropped_image = cropped_image[:, :-1]
        return cropped_image

    def _separate(self, trajectories):
        result = []
        for idx in range(int(trajectories.particle.max()) + 1):
            result.append(trajectories[trajectories.particle == idx])
        return result


if __name__ == '__main__':
    fn = 'sample.avi'
    fitter = VideoFitter(fn, estimator=True)
    maxframe = 4
    fitter.localize(maxframe=maxframe)
    fitter.fit(0, maxframe=maxframe)
    fit_df = fitter.fit_dfs[0]
    fig, ax = plt.subplots(2)
    ax[0].scatter(x=fit_df.index, y=fit_df.a_p)
    ax[1].scatter(x=fit_df.index, y=fit_df.n_p)
    plt.show()
