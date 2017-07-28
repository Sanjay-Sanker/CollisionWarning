from yolo import *


# img = plt.imread('test_images/test1.jpg');
# new = frame_func(img);
# plt.imshow(new);
# plt.show();

project_video_output = './test_video_output.mp4'
clip1 = VideoFileClip("./test_video.mp4")

lane_clip = clip1.fl_image(frame_func) #NOTE: this function expects color images!!
lane_clip.write_videofile(project_video_output, audio=False)