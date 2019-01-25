# mie_video
A Python fitter to analyze holographic video microscopy flows with the Lorenz-Mie theory of light scattering.

**What does the fitter do?**<br/>
*One* fitter retrieves the size, refractive index, and three dimensional position of *one* particle across all frames in the video where the particle appears.<br/>
**How does it work?**<br/>
After initialization, VideoFitter.localize tracks all particles in all frames of the input video file, determines using trackpy.link the distinct particle trajectories, and compiles them into a DataFrame. Using VideoFitter.fit, the user chooses a trajectory to analyze, fits each frame to Lorenz-Mie theory, and compiles the fits into another DataFrame. With VideoFitter.animate it can create a matplotlib animation of the particle trajectories, and with VideoFitter.test it creates an instance of pylorenzmie's LMTool for initial guesses<br/>

Works alongside https://github.com/davidgrier/pylorenzmie<br/>

**COMING SOON:**<br/>
1. GPU Accelerated Localization and Fitting<br/>
2. No need for initial guesses! Machine learning models will localize frames and find accurate initial guesses for each particle's parameters. Those guesses will be fed into the least-squares fitter.<br/>
3. Full integration with pylorenzmie.<br/>
