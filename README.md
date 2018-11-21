# mie_video
A Python fitter to analyze holographic video microscopy flows with the Lorenz-Mie theory of light scattering.

**What does the fitter do?**<br/>
*One* fitter retrieves the size, refractive index, and three dimensional position of *one* particle across all frames in the video where the particle appears.<br/>
**How does it work?**<br/>
Upon initialization, Video_Fitter.localize tracks all particles in all frames of the input video file, determines using trackpy.link the distinct particle trajectories, and compiles them into a DataFrame. Using Video_Fitter.fit, the user chooses a trajectory to analyze, fits each frame to Lorenz-Mie theory, and compiles the fits into another DataFrame.

Additional dependencies are<br/>
For localization: https://github.com/davidgrier/tracker, https://github.com/markhannel/features, https://github.com/markhannel/lab_io <br/>
For fitting: https://github.com/markhannel/lorenzmie
