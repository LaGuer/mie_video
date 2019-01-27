'''Class to fit trajectories in a video to Lorenz-Mie theory.'''

import mie_video.utilities.editing as edit
from mie_video.tracking import oat
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
import logging
logger = logging.getLogger()
logger.setLevel(logging.WARNING)
logger.setLevel(logging.INFO)


class VideoFitter(object):

    def __init__(self, fn,
                 guesses=OrderedDict(zip(['x', 'y', 'z', 'a_p', 'n_p',
                                          'n_m', 'mpp', 'lamb'],
                                         (None for n in range(8)))),
                 background=None,
                 linked_df=None,
                 detection_method='oat',
                 save=True):
        """
        Args:
            fn: filename
            guesses: initial guesses of particle parameters for
                     first frame of video. Use Video_Fitter.test
                     to find a good estimate
        Keywords:
            background: background image or filename of a background video
            detection_method: 'oat': Oriental alignment transform
                              'tf': Tensorflow (you must use 640x480 frames)
            linked_df: input if linked_df has already been calculated and saved
        """
        self.save = save
        self.init_processing(fn, background)
        self.init_fitting(guesses)
        self.init_localization(linked_df, detection_method)

    def init_processing(self, fn, background):
        """
        Initialize parameters for files, cropping, and background.
        Change these parameters before localizing for best/preferred results.
        """
        self.fn = os.path.expanduser(fn)
        self.dark_count = 13
        self.frame_size = (1024, 1280)
        self.forced_crop = None
        if type(background) is str and background[:-4] == '.avi':
            self.background = edit.background(background,
                                              shape=self.frame_size)
        elif type(background) is np.ndarray or background is None:
            self.background = background
        else:
            raise(ValueError("background must be .avi filename, np.ndarray, or None."))

    def init_localization(self, linked_df, detection_method):
        """
        Initialize parameters for detection.

        If using the orientational alignment transform,
        adjust parameters nfringes, maxrange, tp_params, and
        crop_threshold as needed.
        """
        self.detection_method = detection_method
        self.linked_df = linked_df
        self.trajectories = None
        if self.detection_method == 'oat':
            self.nfringes = 25
            self.maxrange = 300.
            self.tp_params = {'diameter': 31, 'topn': 1}
            self.crop_threshold = 300.
        if type(self.linked_df) is pd.DataFrame:
            if 'particle' not in self.linked_df.columns:
                tp.link(self.linked_df, search_range=20, memory=3,
                        pos_columns=['y', 'x'])
            self.trajectories = self._separate(self.linked_df)
            self.fit_dfs = [None for _ in range(len(self.trajectories))]
            logging.info(str(len(self.trajectories)) + " trajectories found.")

    def init_fitting(self, guesses):
        """
        Initialize parameters for fitting.
        """
        self._fixed_params = ['n_m', 'mpp', 'lamb']
        self.params = guesses
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

    def localize(self, max_frame=None, minframe=None):
        '''
        Returns DataFrame of particle parameters in each frame
        of a video linked with their trajectory index
        '''
        if type(self.linked_df) is pd.DataFrame:
            return
        split = self.fn.split("/")
        dest_fn = split[:-1] + "linked_df_" + split[-1][:-4] + ".csv"
        cap = cv2.VideoCapture(self.fn)
        frame_no = 0
        if minframe is not None:
            cap.set(1, minframe)
            frame_no = minframe
        cols = ['x', 'y', 'w', 'h', 'frame']
        self.unlinked_df = pd.DataFrame(columns=cols,
                                        data=[(None for col in cols)])
        while(cap.isOpened()):
            try:
                if frame_no == max_frame:
                    break
                ret, frame = cap.read()
                if ret is False:
                    break
                # Normalize
                norm = self._process(frame)
                # Crop if feature of interest is there in all frames
                if self.forced_crop is not None:
                    norm = self._force_crop(norm)
                # Find features in current frame
                if self.detection_method == 'oat':
                    features, circ = oat(norm, frame_no,
                                         locate_params=self.tp_params,
                                         maxrange=self.maxrange,
                                         nfringes=self.nfringes)
                else:
                    raise(ValueError("method must be either \'oat\' or \'tf\'"))
                # Add features to total dataset.
                for feature in features:
                    feature = np.append(feature, frame_no)
                    i = len(self.unlinked_df) + 1
                    self.unlinked_df = self.unlinked_df.append(pd.DataFrame(columns=cols,
                                                                            data=[feature],
                                                                            index=[i]))
                if self.save:
                    self.unlinked_df.to_csv(dest_fn)
                # Advance frame_no
                frame_no += 1
            except KeyboardInterrupt:
                logging.warning("Ending progress...")
                break
        cap.release()
        # Put data set in DataFrame and link
        self.linked_df = tp.link(self.unlinked_df, search_range=20, memory=3,
                                 pos_columns=['y', 'x'])
        if self.save:
            self.linked_df.to_csv(dest_fn)
        self.trajectories = self._separate(self.linked_df)
        self.fit_dfs = [None for _ in range(len(self.trajectories))]
        logging.info(str(len(self.trajectories)) + " trajectories found.")

    def fit(self, trajectory_no, max_frame=None, minframe=None,
            fixed_guess={}):
        '''
        None DataFrame of fitted parameters in each frame
        for a given trajectory.
        
        Args:
            trajectory_no: index of particle trajectory in self.trajectories.
        '''
        idx = trajectory_no
        split = self.fn.split("/")
        dest = split[:-1] + "fit_df" + str(idx) + "_" + split[-1][:-4] + ".csv"
        p_df = self.trajectories[idx]
        cap = cv2.VideoCapture(self.fn)
        frame_no = 0
        if minframe is not None:
            cap.set(1, minframe)
            frame_no = minframe
        cols = np.append(list(self.params.keys()), ['frame', 'redchi'])
        self.fit_dfs[idx] = pd.DataFrame(columns=cols,
                                         data=[(None for col in cols)])
        while(cap.isOpened()):
            try:
                if frame_no == max_frame:
                    break
                ret, frame = cap.read()
                if ret is False:
                    break
                # Normalize image
                norm = self._process(frame)
                # Crop if feature of interest is there in all frames
                if self.forced_crop is not None:
                    norm = self._force_crop(norm)
                # Crop feature of interest.
                feats = p_df.loc[p_df['frame'] == frame_no]
                if len(feats) == 0:
                    logging.warning('No particle found in frame ' + str(frame_no))
                    frame_no += 1
                    continue
                x, y, w, h, frame, particle = feats.iloc[0]
                feature = self._crop(norm, x, y, w, h)
                # Fit frame
                start = time()
                fit = self.fitter.fit(feature)
                fit_time = time() - start
                logging.info(self.fn[-7:-4]
                             + " time to fit frame " + str(frame_no) +
                             ": " + str(fit_time))
                logging.info("Fit RedChiSq: " + str(fit.redchi))
                logging.info("Fit z: " + str(fit.params['z'].value))
                logging.info("Fit a_p, n_p: {}, {}".
                             format(fit.params['a_p'].value,
                                    fit.params['n_p'].value))
                # Add fit to dataset
                row = [fit.params['x'].value + x, fit.params['y'].value + y,
                       fit.params]
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
                i = len(self.fit_dfs[idx]) + 1
                self.fit_dfs[idx] = self.fit_dfs[idx].append(pd.DataFrame(columns=cols,
                                                                          data=[row],
                                                                          index=[i]))
                if self.save:
                    self.fit_dfs[idx].to_csv(dest_fn)
                frame_no += 1
                # Set guesses for next fit
                guesses = []
                for param in fit.params.values():
                    if param.name in fixed_guess.keys():
                        guesses.append(fixed_guess[param.name])
                    else:
                        guesses.append(param.value)
                self.params = guesses
            except KeyboardInterrupt:
                logging.warning("Ending progress...")
                break
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
        if frame_no > max(p_df.index) or frame_no < min(p_df.index):
            raise(IndexError("Trajectory not found in frame {} for particle {}"
                             .format(frame_no, trajectory_no)))
        cap = cv2.VideoCapture(self.fn)
        cap.set(1, frame_no)
        ret, frame = cap.read()
        if not ret:
            logging.warning("Frame not read.")
            return
        norm = self._process(frame)
        if type(self.forced_crop) is in (list, tuple) and len(self.forced_crop) == 4:
            norm = self._force_crop()
        # Crop feature
        feats = p_df.loc[p_df['frame'] == frame_no]
        x, y, w, h, frame, particle = feats.iloc[0]
        feature = self._crop(norm, x, y, w, h)
        # Generate guess
        x, y, z, a_p, n_p, n_m, mpp, lamb = guesses
        h = LMHologram(coordinates=coordinates(feature.shape))
        h.particle.r_p = [x + feature.shape[0] // 2, y + feature.shape[1] // 2, z]
        h.particle.a_p = a_p
        h.particle.n_p = n_p
        h.instrument.n_m = n_m
        h.instrument.magnification = mpp
        h.instrument.wavelength = lamb
        hol = h.hologram().reshape(feature.shape)
        # Plot and return normalized image
        plt.imshow(np.hstack([feature, hol, feature - hol]), cmap='gray')
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

#    def write(self, row, df):
#        idx = np.amax(df.index)
#        df.loc[idx]
#        if self.save:            
        
    def _process(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.float)
        if self.background is not None:
            norm = (frame - self.dark_count) / (self.background - self.dark_count)
        else:
            norm = frame / np.mean(frame)
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

    def _force_crop(self, frame):
        """
        Used to analyze only a region of the frame.
        No need to call this function--just set the self.forced_crop field.
        """
        xc, yc, w, h = self.forced_crop
        frame = self._crop(frame, xc, yc, w, h, square=False)
        return frame

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
