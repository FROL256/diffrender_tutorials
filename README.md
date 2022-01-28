This is differentiable raster sandbox for get familiar with the basics of differentiable rendering  

# Make animation

1) cd rendered_opt
2) ffmpeg -framerate 25 -pattern_type glob -i '*.bmp' out.gif

# Installation of opt. tools (optional!)

Sorry for this sh*t, we need Eigen for optimisation methods for a while

1) Download Eigen 3.4

2) install it in your system:
   mkdir build && cd build
   cmake ..
   sudo make install

3) install https://github.com/kthohr/optim to include folder as header-only solution (dont forget to do 'git submodule update --init')

