import glob
import pickle
from collections import deque

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.ndimage import label

from lesson_functions import *
from sklearn.externals import joblib

dist_pickle = pickle.load(open("model_save/dist1.p", "rb"))

svc = dist_pickle["svc"]
#clf = dist_pickle["clf"]
X_scaler = dist_pickle["X_scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

#img = mpimg.imread('test_images/test1.jpg')
#img = mpimg.imread('test_images/test2.jpg')
#img = mpimg.imread('test_images/test3.jpg')
#img = mpimg.imread('test_images/test4.jpg')
img = mpimg.imread('test_images/test5.jpg')
#img = mpimg.imread('test_images/test6.jpg')
ystart = 400
ystop = 656
#scale = 1.5
#print(pix_per_cell, cell_per_block, spatial_size)
def pipeline(image):
    bbox_list =  []
    for scale in range(10, 30, 3):
        scale /= 10
        #print(scale)
        draw_image, box_list = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        #print(box_list)
        bbox_list.extend(box_list)


    #print(bbox_list)

    heat = np.zeros_like(image[:, :, 0]).astype(np.float32)
    heat = add_heat(heat, bbox_list)
    heat = apply_threshold(heat, threshold=2)

    current_heatmap = np.clip(heat, 0, 255)

    history.append(current_heatmap)
    heatmap = np.zeros_like(current_heatmap).astype(np.float)
    for heat in history:
        heatmap += heat

    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    return draw_img

history = deque(maxlen=10)
#out_img = pipeline(img)
#plt.imshow(out_img)
#plt.show()
#white_output = './project_video_output.mp4'
#clip = VideoFileClip("./project_video.mp4")
white_output = './test_video_output.mp4'
clip = VideoFileClip("./test_video.mp4")
white_clip = clip.fl_image(pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)