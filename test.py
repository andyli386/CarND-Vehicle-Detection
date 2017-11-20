import glob
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip

from lesson_functions import *
from sklearn.externals import joblib

dist_pickle = pickle.load(open("model_save/dist1.p", "rb"))

svc = dist_pickle["svc"]
clf = dist_pickle["clf"]
X_scaler = dist_pickle["X_scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

img = mpimg.imread('test_images/test6.jpg')
ystart = 400
ystop = 656
scale = 1.5
#print(pix_per_cell, cell_per_block, spatial_size)
def pipeline(image):
    bbox_list =  []
    for scale in range(1.0, 0.2, 3.0):
        box_list, draw_image = find_cars(image, ystart, ystop, scale, clf, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        bbox_list.extend(box_list)

#plt.imshow(out_img)
#plt.show()

#white_output = './project_video_output.mp4'
#clip = VideoFileClip("./project_video.mp4")
white_output = './test_video_output.mp4'
clip = VideoFileClip("./test_video.mp4")
white_clip = clip.fl_image(pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)