ffmpeg -framerate 25 -pattern_type glob -i 'rendered_opt0/*.bmp' view0.mp4
ffmpeg -framerate 25 -pattern_type glob -i 'rendered_opt1/*.bmp' view1.mp4
ffmpeg -framerate 25 -pattern_type glob -i 'rendered_opt2/*.bmp' view2.mp4
ffmpeg -i view0.mp4 -i view1.mp4 -i view2.mp4 -filter_complex "[1:v][0:v]scale2ref=oh*mdar:ih[1v][0v];[2:v][0v]scale2ref=oh*mdar:ih[2v][0v];[0v][1v][2v]hstack=3,scale='2*trunc(iw/2)':'2*trunc(ih/2)'" views_all.mp4
