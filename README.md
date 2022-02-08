This is differentiable raster sandbox for get familiar with the basics of differentiable rendering  

# Build
Use Cmake in a standart way

# Make animation

1) cd rendered_opt
2) ffmpeg -framerate 25 -pattern_type glob -i '*.bmp' out.gif

# Installation of complex opt. tools (optional, -DCOMPLEX_OPT=ON)

Sorry for this, we need Eigen for advanced optimisation methods for a while (use https://github.com/kthohr/optim)

1) Download Eigen 3.4

2) install it in your system:
   mkdir build && cd build
   cmake ..
   sudo make install

3) install https://github.com/kthohr/optim to 'include' folder as header-only solution or any other way (don't forget to do 'git submodule update --init')

