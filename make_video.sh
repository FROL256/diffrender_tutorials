ffmpeg -framerate 25 -i output/rendered_opt0/render_%04d.bmp -c:v libx264 output/video1.mp4
ffmpeg -framerate 25 -i output/rendered_opt1/render_%04d.bmp -c:v libx264 output/video2.mp4
ffmpeg -framerate 25 -i output/rendered_opt2/render_%04d.bmp -c:v libx264 output/video3.mp4
ffmpeg -i output/video1.mp4 -i output/video2.mp4 -i output/video3.mp4 -filter_complex "[0:v][1:v][2:v]hstack=3" output/output.gif
