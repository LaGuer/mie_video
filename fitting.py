'''Class to fit trajectories in a video to Lorenz-Mie theory.'''

import mie_video.utilities.editing as edit
from mie_video.tracking import oat, feature_extent
from CNNLorenzMie.Localizer import Localizer
from .animation import Animate
from pylorenzmie.fitting.mie_fit import Mie_Fitter
from pylorenzmie.lmtool.LMTool import LMTool
from pylorenzmie.theory.LMHologram import LMHologram
from pylorenzmie.theory.Instrument import coordinates
from PyQt5 import QtWidgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trackpy as tp
import cv2
from collections import OrderedDict
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


class VideoFitter(object):

    def __init__(self, fn,
                 guesses=OrderedDict(zip(['x', 'y', 'z', 'a_p', 'n_p',
                                          'n_m', 'mpp', 'lamb'],
                                         (None for n in range(8)))),
                 background=None,
                 linked_df=None,
                 detection_method='cnn',
                 localizer=None,
                 save=False):
        """
        Args:
            fn: filename
            guesses: initial guesses of particle parameters for
                     first frame of video. Use Video_Fitter.test
                     to find a good estimate
        Keywords:
            background: background image or filename of a background video
            detection_method: 'oat': Oriental alignment transform
                              'cnn': CNNLorenzMie localizer
            linked_df: input if linked_df has already been calculated and saved
        """
        self.save = save
        self.init_processing(fn, background)
        self.init_fitting(guesses)
        self.init_localization(linked_df, detection_method, localizer)

    def init_processing(self, fn, background):
        """
        Initialize parameters for files, cropping, and background.
        Change these parameters before localizing for best/preferred results.
        """
        self.fn = os.path.expanduser(fn)
        self.dark_count = 13
        self.frame_size = (1024, 1280)
        self.forced_crop = None
        if type(background) is str and background[-4:] == '.avi':
            frame_size = tuple(reversed(self.frame_size))
            signal.alarm(60)
            self.background = edit.background(background,
                                              shape=frame_size)
            signal.alarm(0)
        elif type(background) is np.ndarray or background is None:
            self.background = background
        else:
            raise(ValueError("background must be .avi filename, np.ndarray, or None."))

    def init_localization(self, linked_df, detection_method, localizer):
        """
        Initialize parameters for detection.

        If using the orientational alignment transform,
        adjust parameters nfringes, maxrange, tp_params, and
        crop_threshold as needed.
        """
        self.detection_method = detection_method
        self.linked_df = linked_df
        self.trajectories = None
        self.nfringes = 25
        self.maxrange = 375.
        self.crop_threshold = 350.
        if self.detection_method == 'oat':
            self.tp_params = {'diameter': 31, 'topn': 1}
        elif self.detection_method == 'cnn':
            if localizer is None:
                self.localizer = Localizer()
            else:
                self.localizer = localizer
        if type(self.linked_df) is pd.DataFrame:
            self.linked_df = self.linked_df[['x', 'y', 'w', 'h',
                                             'frame', 'particle']]
            self.trajectories = self._separate(self.linked_df)
            self.fit_dfs = [None for _ in range(len(self.trajectories))]
            logging.info(str(len(self.trajectories)) + " trajectories found.")

    def init_fitting(self, guesses):
        """
        Initialize parameters for fitting.
        """
        self._fixed_params = ['n_m', 'mpp', 'lamb']
        self._keys = ['x', 'y', 'z', 'a_p', 'n_p',
                      'n_m', 'mpp', 'lamb']
        self._params = guesses
        self.fitter = Mie_Fitter(self.params, fixed=self.fixed_params)

    @property
    def params(self):
        '''
        Returns OrderedDict of parameters x, y, z, a_p, n_p, n_m, mpp, lamb
        '''
        return self._params

    @params.setter
    def params(self, guesses):
        '''
        Sets parameters for fitter

        Args:
            guesses: list of parameters ordered
                     [x, y, z, a_p, n_p, n_m, mpp, lamb]
        '''
        if len(guesses) != 8:
            raise ValueError("guesses must be length 8")
        if type(guesses) is list:
            new_params = OrderedDict(zip(['x', 'y', 'z', 'a_p', 'n_p',
                                          'n_m', 'mpp', 'lamb'], guesses))
        elif type(guesses) is OrderedDict:
            new_params = guesses
        else:
            raise ValueError("Type of guesses must be list or OrderedDict")
        for key in self.params.keys():
            self.fitter.set_param(key, new_params[key])
        self._params = self.fitter.p.valuesdict()

    @property
    def fixed_params(self):
        return self._fixed_params

    @fixed_params.setter
    def fixed_params(self, params):
        """
        Set fixed parameters for fitting
        """
        self._fixed_params = params
        self.fitter = Mie_Fitter(self.params, fixed=params)

    def localize(self, maxframe=None, minframe=None):
        '''
        Returns DataFrame of particle parameters in each frame
        of a video linked with their trajectory index
        '''
        if type(self.linked_df) is pd.DataFrame:
            return
        c = "/"
        split = self.fn.split(c)
        dest = c.join(split[:-1]) + "/linked_df_" + split[-1][:-4] + ".csv"
        cap = cv2.VideoCapture(self.fn)
        if type(minframe) == int:
            cap.set(1, minframe)
            frame_no = minframe
        else:
            frame_no = 0
        cols = ['x', 'y', 'w', 'h', 'frame']
        self.unlinked_df = pd.DataFrame(columns=cols)
        while(cap.isOpened()):
            if frame_no == maxframe:
                break
            ret, frame = cap.read()
            if ret is False:
                break
            t = time()
            if self.detection_method == 'oat':
                norm = self._process(frame)
                features, circ = oat(norm, frame_no,
                                     locate_params=self.tp_params,
                                     maxrange=self.maxrange,
                                     nfringes=self.nfringes)
                s = features[-1][2]
            elif self.detection_method == 'cnn':
                norm = self._process(frame)
                features = []
                feats = self.localizer.predict([edit.inflate(norm)*100])
                for feature in feats[0]:
                    xc, yc, w, h = feature['bbox']
                    s = feature_extent(norm, (xc, yc),
                                       nfringes=self.nfringes,
                                       maxrange=self.maxrange)
                    if s > self.crop_threshold:
                        s = self.crop_threshold
                    features.append([xc, yc, s, s])
            else:
                raise(ValueError("method must be either \'oat\' or \'cnn\'"))
            msg = "Time to find {} features".format(len(features))
            msg += " at frame {}".format(frame_no)
            msg += ": {:.2f}".format(time() - t)
            logging.info(msg + "\nLast feature size: {}".format(s))
            for feature in features:
                feature = np.append(feature, frame_no)
                self._write(feature, self.unlinked_df, dest)
            frame_no += 1
        cap.release()
        self.linked_df = tp.link(self.unlinked_df, search_range=20, memory=3,
                                 pos_columns=['y', 'x'])
        if self.save:
            self.linked_df.to_csv(dest)
        self.trajectories = self._separate(self.linked_df)
        self.fit_dfs = [None for _ in range(len(self.trajectories))]
        logging.info(str(len(self.trajectories)) + " trajectories found.")

    def fit(self, trajectory_no, maxframe=None, minframe=None,
            fixed_guess={'x': 0., 'y': 0.}):
        '''
        None DataFrame of fitted parameters in each frame
        for a given trajectory.
        
        Args:
            trajectory_no: index of particle trajectory in self.trajectories.
        '''
        idx = trajectory_no
        c = "/"
        split = self.fn.split(c)
        dest = c.join(split[:-1]) + "/fit_df" + str(idx) + "_" + split[-1][:-4] + ".csv"
        cap = cv2.VideoCapture(self.fn)
        if type(minframe) == int:
            cap.set(1, minframe)
            frame_no = minframe
        else:
            frame_no = 0
        cols = np.append(list(self.params.keys()), ['frame', 'redchi'])
        self.fit_dfs[idx] = pd.DataFrame(columns=cols)
        p_df = self.trajectories[idx]
        while(cap.isOpened()):
            if frame_no == maxframe:
                break
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
            feature = self._crop(norm, x, y, w, h)
            # Fit frame
            signal.alarm(40)
            start = time()
            try:
                fit = self.fitter.fit(feature)
            except TimeoutError:
                logging.warning("Fit timed out")
                continue
            signal.alarm(0)
            msg = "{} time to fit frame {}: {:.2f}".format(split[-1][:-4],
                                                           frame_no,
                                                           time() - start)
            msg += "\nz={:.2f}, a_p={:.2f}, n_p={:.2f}".format(fit.params['z'].value,
                                                               fit.params['a_p'].value,
                                                               fit.params['n_p'].value)
            msg += "\nredchi={:.2f}".format(fit.redchi)
            logging.info(msg)
            # Add fit to dataset
            row = []
            for col in cols:
                if col == 'x':
                    row.append(fit.params[col].value + x)
                elif col == 'y':
                    row.append(fit.params[col].value + y)
                elif col == 'frame':
                    row.append(frame_no)
                elif col == 'redchi':
                    row.append(fit.redchi)
                else:
                    row.append(fit.params[col].value)
            self._write(row, self.fit_dfs[idx], dest)
            frame_no += 1
            # Set guesses for next fit
            # TODO: Replace with tensorflow estimates
            guesses = []
            for param in fit.params.values():
                if param.name in fixed_guess.keys():
                    guesses.append(fixed_guess[param.name])
                else:
                    guesses.append(param.value)
            self.params = guesses
        if self.save:
            self.fit_dfs[idx].to_csv(dest)
        cap.release()

    def tool(self, frame_no=0, trajectory_no=0):
        '''
        Uses LMTool to find good initial guesses for fit.
        '''
        p_df = self.trajectories[trajectory_no]
        if frame_no > max(p_df.index) or frame_no < min(p_df.index):
            raise(IndexError("Trajectory not found in frame {} for particle {}"
                             .format(frame_no, trajectory_no)))
        cap = cv2.VideoCapture(self.fn)
        cap.set(1, frame_no)
        ret, frame = cap.read()
        if not ret:
            logging.warning("Frame not read.")
            return
        app = QtWidgets.QApplication(sys.argv)
        lmtool = LMTool(data=self._process(frame),
                        normalization=self.background)
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
        print(min(p_df['frame']))
        if frame_no > max(p_df['frame']) and frame_no < min(p_df['frame']):
            raise(IndexError("Trajectory not found in frame {} for particle {}"
                             .format(frame_no, trajectory_no)))
        cap = cv2.VideoCapture(self.fn)
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
        x, y, z, a_p, n_p, n_m, mpp, lamb = guesses
        h = LMHologram(coordinates=coordinates(feature.shape))
        h.particle.r_p = [x+feature.shape[0]//2, y+feature.shape[1]//2, z]
        h.particle.a_p = a_p
        h.particle.n_p = n_p
        h.instrument.n_m = n_m
        h.instrument.magnification = mpp
        h.instrument.wavelength = lamb
        hol = h.hologram().reshape(feature.shape)
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
        if self.linked_df is None:
            raise UserWarning("Error: run localize() before animate().")
        elif type(self.linked_df) is pd.DataFrame:
            self.animation = Animate(self.fn, linked_df=self.linked_df)
            self.animation.run()
        else:
            raise ValueError("Error: linked_df is unknown datatype.")

    def _write(self, row, df, dest):
        last_line = df.index.max()
        if last_line is np.nan:
            df.loc[0] = row
        #    if self.save:
        #        df.to_csv(dest, mode='a')
        else:
            df.loc[last_line + 1] = row
        #    if self.save:
        #        df.to_csv(dest, mode='a', header=None)

    def _process(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float)
        if self.background is not None:
            norm = (frame - self.dark_count) / (self.background - self.dark_count)
        else:
            norm = frame / np.mean(frame)
        if type(self.forced_crop) is list or type(self.forced_crop) is tuple:
            if len(self.forced_crop) == 4:
                xc, yc, w, h = self.forced_crop
                norm = self._crop(norm, xc, yc, w, h, square=False)
            else:
                logging.warning("Forced crop must be (xc, yc, w, h)")
        return norm

    def _crop(self, image, xc, yc, w, h, square=True):
        '''
        Returns a cropped image.
        '''
        cropped_image = image[int(yc - h//2): int(yc + h//2),
                              int(xc - w//2): int(xc + w//2)].astype(float)
        xdim, ydim = cropped_image.shape
        if square is True:
            if xdim == ydim:
                return cropped_image
            if xdim > ydim:
                cropped_image = cropped_image[1:-1, :]
            else:
                cropped_image = cropped_image[:, 1:-1]
        return cropped_image

    def _separate(self, trajectories):
        '''
        Returns list of separated DataFrames for each particle
        
        Args:
             trajectories: Pandas DataFrame linked by trackpy.link(df)
        '''
        result = []
        for idx in range(int(trajectories.particle.max()) + 1):
            result.append(trajectories[trajectories.particle == idx])
        return result
