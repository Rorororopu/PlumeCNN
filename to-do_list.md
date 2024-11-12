# To-do List

## Tasks

1. Streamline
2. Calculate Gradient
3. Visualization (scatterplot)

## Problems

1. Interpolation
For 3D data, doing direct interpolation will use soo many memory that can't be completed by HPC.
So what we do is to first interpolate z-slice, then interpolate y slice.

2. Streamline
If number of seed points is a little more, it can't be correctly analyzed by the browser.
Somehow, the code have to be run twice to generate a readable result.
Somehow, even if you just changed the seed of random number, the code won't run.
Can't export image.
