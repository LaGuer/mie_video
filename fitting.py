'''Class to fit trajectories in a video to Lorenz-Mie theory.'''

from tracker import tracker
import mie_video.editing as edit
import numpy as np
import pandas as pd
from lorenzmie.theory import spheredhm
from lorenzmie.fitting.mie_fit import Mie_Fitter
import cv2
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
from time import time
import trackpy as tp
import lab_io.h5video as h5
import features.circletransform as ct


class Video_Fitter(object):

    def __init__(self, fn, guesses,
                 background=None,
                 background_fn=None,
                 linked_df=None,
                 detection_method='oat'):
        """
        Args:
            fn: filename
            guesses: initial guesses of particle parameters for
                     first frame of video. Use Video_Fitter.test
                     to find a good estimate
        Keywords:
            background_fn: filename of a background video
            forced_crop: use this to analyze a subregion of the video
            fixed_params: fixed parameters for Lorenz-Mie fits
            detection_method: 'oat': Oriental alignment transform
                              'tf': Tensorflow (you must use 640x480 frames)
            minmass: minimum brightness for 'oat' detection
            linked_df: input if linked_df has already been calculated and saved
        """
        self.init_processing(fn, background_fn, background)
        self.init_fitting(guesses)
        self.init_localization(linked_df, detection_method)

    def init_processing(self, fn, background_fn, background):
        self.fn = os.path.expanduser(fn)
        self.dark_count = 13
        self.frame_size = (1024, 1280)
        self.forced_crop = None
        if background_fn is not None:
            if background is not None:
                s = "No passing both background video filename and background image"
                raise ValueError(s)
            self.background = edit.background(background_fn,
                                              shape=self.frame_size)
        else:
            self.background = background

    def init_localization(self, linked_df, detection_method):
        self.detection_method = detection_method
        self.linked_df = linked_df
        self.trajectories = None
        if self.detection_method == 'oat':
            self.nfringes = 25
            self.maxrange = 550.
            self.tp_params = {'diameter': 51, 'topn': 1}
            self.crop_thresh = 300
        if self.linked_df is not None:
            self.trajectories = self.separate(self.linked_df)
            self.fit_dfs = [None for _ in range(len(self.trajectories))]
            print(str(len(self.trajectories)) + " trajectories found.")

    def init_fitting(self, guesses):
        self._fixed_params = ['n_m', 'mpp', 'lamb']
        self._params = OrderedDict(zip(['x', 'y', 'z', 'a_p',
                                        'n_p', 'n_m', 'mpp', 'lamb'], guesses))
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
        new_params = OrderedDict(zip(['x', 'y', 'z', 'a_p', 'n_p',
                                      'n_m', 'mpp', 'lamb'], guesses))
        for key in self.params.keys():
            if key == 'x' or key == 'y':
                self.fitter.set_param(key, 0.0)
            else:
                self.fitter.set_param(key, new_params[key])
            self._params = self.fitter.p.valuesdict()

    @property
    def fixed_params(self):
        return self._fixed_params

    @fixed_params.setter
    def fixed_params(self, params):
        self._fixed_params = params
        self.fitter = Mie_Fitter(self.params, fixed=self._fixed_params)

    def localize(self, max_frames=None):
        '''
        Returns DataFrame of particle parameters in each frame
        of a video linked with their trajectory index

        Args:
            video: video filename
        Keywords:
            background: background image for normalization
            dark_count: dark count of camera
            minmass: min brightness for oat detection 
        '''
        if self.linked_df is not None:
            return
        # Create VideoCapture to read video
        cap = cv2.VideoCapture(self.fn)
        # Initialize components to build overall dataset.
        frame_no = 0
        data = []
        while(cap.isOpened()):
            if frame_no == max_frames:
                break
            # Get frame
            ret, frame = cap.read()
            if ret is False:
                break
            # Normalize
            norm = self.normalize(frame)
            # Crop if feature of interest is there in all frames
            norm = self.force_crop(norm)
            # Find features in current frame
            if self.detection_method == 'tf':
                tf = tracker.tracker()
                features = tf.predict(edit.inflate(norm))
            elif self.detection_method == 'oat':
                features, circ = self.oat(norm, frame_no)
            else:
                raise(ValueError("method must be either \'oat\' or \'tf\'"))
            # Add features to total dataset.
            for feature in features:
                feature = np.append(feature, frame_no)
                data.append(feature)
            # Advance frame_no
            frame_no += 1
        cap.release()
        # Put data set in DataFrame and link
        result_df = pd.DataFrame(columns=['x', 'y', 'w', 'h', 'frame'],
                                 data=data)
        self.linked_df = tp.link(result_df, search_range=20, memory=3,
                                 pos_columns=['y', 'x'])
        self.trajectories = self.separate(self.linked_df)
        self.fit_dfs = [None for _ in range(len(self.trajectories))]
        print(str(len(self.trajectories)) + " trajectories found.")

    def fit(self,
            trajectory,
            fixed_guess={'x': None, 'y': None,
                         'z': None, 'a_p': None, 'n_p': None,
                         'n_m': None, 'lamb': None,
                         'mpp': None},
            max_frames=None):
        '''
        None DataFrame of fitted parameters in each frame
        for a given trajectory.
        
        Args:
            trajectory: index of particle trajectory in self.trajectories.
        '''
        p_df = self.trajectories[trajectory]
        cap = cv2.VideoCapture(self.fn)
        frame_no = 0
        data = {}
        for key in self.params:
            data[key] = []
            data['frame'] = []
            data['redchi'] = []
        while(cap.isOpened()):
            if frame_no == max_frames:
                break
            ret, frame = cap.read()
            if ret is False:
                break
            # Normalize image
            norm = self.normalize(frame)
            # Crop if feature of interest is there in all frames
            norm = self.force_crop(norm)
            # Crop feature of interest.
            feats = p_df.loc[p_df['frame'] == frame_no]
            if len(feats) == 0:
                print('No particle found in frame ' + str(frame_no))
                frame_no += 1
                continue
            x, y, w, h, frame, particle = feats.iloc[0]
            feature = edit.crop(norm, x, y, w, h)
            # Fit frame
            start = time()
            fit = self.fitter.fit(feature)
            fit_time = time() - start
            print(self.fn[-7:-4] + " time to fit frame " + str(frame_no) +
                  ": " + str(fit_time))
            print("Fit RedChiSq: " + str(fit.redchi))
            print("Fit z: " + str(fit.params['z'].value))
            print("Fit a_p, n_p: {}, {}".format(fit.params['a_p'].value,
                                                fit.params['n_p'].value))
            # Add fit to dataset
            for key in data.keys():
                if key == 'x':
                    data[key].append(fit.params[key].value + x)
                elif key == 'y':
                    data[key].append(fit.params[key].value + y)
                elif key == 'frame':
                    data[key].append(frame_no)
                elif key == 'redchi':
                    data[key].append(fit.redchi)
                else:
                    data[key].append(fit.params[key].value)
            frame_no += 1
            # Set guesses for next fit
            guesses = []
            for param in fit.params.values():
                if fixed_guess[param.name] is not None:
                    guesses.append(fixed_guess[param.name])
                else:
                    guesses.append(param.value)
            self.params = guesses
        cap.release()
        self.fit_dfs[trajectory] = pd.DataFrame(data=data)

    def normalize(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.float)
        if self.background is not None:
            norm = (frame - self.dark_count) / (self.background - self.dark_count)
        else:
            norm = frame
        return norm

    def force_crop(self, frame):
        if self.forced_crop is not None:
            xc, yc, w, h = self.forced_crop
            frame = edit.crop(frame, xc, yc, w, h, square=False)
        return frame

    def separate(self, trajectories):
        '''
        Returns list of separated DataFrames for each particle
        
        Args:
             trajectories: Pandas DataFrame linked by trackpy.link(df)
        '''
        result = []
        for idx in range(int(trajectories.particle.max()) + 1):
            result.append(trajectories[trajectories.particle == idx])
        return result

    def oat(self, norm, frame_no):
        '''Use the orientational alignment transform
        on every pixel of an image and return features.'''
        t = time()
        circ = ct.circletransform(norm, theory='orientTrans')
        circ = circ / np.amax(circ)
        circ = h5.TagArray(circ, frame_no)
        feats = tp.locate(circ,
                          engine='numba',
                          **self.tp_params)
        feats['w'] = 400
        feats['h'] = 400
        features = np.array(feats[['x', 'y', 'w', 'h']])
        s = None
        for idx, feature in enumerate(features):
            s = self.feature_extent(norm, (feature[0], feature[1]))
            if self.crop_thresh is not None:
                if s > self.crop_thresh:
                    s = self.crop_thresh
            features[idx][2] = s
            features[idx][3] = s
        print("Time to find {} features at frame {}: {}".format(features.shape[0],
                                                                frame_no,
                                                                time() - t))
        print("Mass of particles: {}".format(list(feats['mass'])))
        print("Side length: {}".format(s))
        return features, circ

    def feature_extent(self, norm, center):
        ravg, rstd = self.aziavg(norm, center)
        b = ravg - 1.
        ndx = np.where(np.diff(np.sign(b)))[0] + 1.
        if len(ndx) <= self.nfringes:
            return self.maxrange
        else:
            return float(ndx[self.nfringes])

    def aziavg(self, data, center):
        x_p, y_p = center
        y, x = np.indices((data.shape))
        d = data.ravel()
        r = np.hypot(x - x_p, y - y_p).astype(np.int).ravel()
        nr = np.bincount(r)
        ravg = np.bincount(r, d) / nr
        avg = ravg[r]
        rstd = np.sqrt(np.bincount(r, (d - avg)**2) / nr)
        return ravg, rstd

    def test(self, guesses, trajectory=0, frame_no=0):
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
        p_df = self.trajectories[trajectory]
        if frame_no > max(p_df.index) or frame_no < min(p_df.index):
            raise(IndexError("Trajectory not found in frame {} for particle {}"
                             .format(frame_no, trajectory)))
        cap = cv2.VideoCapture(self.fn)
        cap.set(1, frame_no)
        ret, frame = cap.read()
        if not ret:
            print("Frame not read.")
            return
        # Normalize and force crop
        norm = self.normalize(frame)
        norm = self.force_crop(norm)
        # Crop feature
        x, y, w, h, frame_no, particle = p_df.iloc[0, :]
        feature = edit.crop(norm, x, y, w, h)
        # Generate guess
        x, y, z, a_p, n_p, n_m, mpp, lamb = guesses
        theory = spheredhm.spheredhm([0, 0, z],
                                     a_p, n_p, n_m,
                                     dim=feature.shape,
                                     lamb=lamb, mpp=mpp)
        # Plot and return normalized image
        plt.imshow(np.hstack([feature, theory]), cmap='gray')
        plt.show()
        cap.release()
        return norm
