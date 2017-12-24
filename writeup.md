**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction, Color Transform and append binned color features and historgrams of color on a labeled training set of images and train a classifier Linear SVM classifier
* Normalize features and randomize a selection for training and testing.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.
* As a challange identify both the lanes and vehicles. 

[//]: # (Image References)
[image1]: ./data/vehicles/image0.jpg
[image2]: ./data/vehicles/image1.jpg
[image3]: ./data/non-vehicles/extra3.jpg
[image4]: ./data/non-vehicles/extra3.jpg
[image5]: ./output_images/hog_images.jpg
[image6]: ./output_images/search_window_images.jpg
[image7]: ./output_images/test_window0.jpg
[image8]: ./output_images/test_window2.jpg
[image9]: ./output_images/heat_map_bounding_box.png
[image10]: ./examples/labels_map.png
[image11]: ./examples/output_bboxes.png
[video1]: ./output_videos/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The functions get_hog_features and extract_features are used for this step. Code for these functions is contained in the second code cell of the 'P5_Vehicle_Detection.ipynb' IPython notebook. These functions are reused from UDACITY training material.  

I start by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

Car | Non Car
:--------:|:------------:
![alt text][image1] | ![alt text][image3]
![alt text][image2] | ![alt text][image4]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=16`, `pixels_per_cell=(8, 8)` and `cells_per_block=(8, 8)`:


![alt text][image5]

I have explored `YUV` and `RGB` color spaces but the results were not proper , so i choose `YCrCb` color space and all the channels were processed. 

Besides HOG parameters i am extracting Spatial Bin and Color Historgram features to increase the accuracy of the trained model.

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and selected the following settings for my project

ColorSpace = `YCrCb`
orient = 16  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 8 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

Besides HOG parameters i am extracting Spatial Bin and Color Historgram features to increase the accuracy of the trained model. Following settings are used for extracting these features

spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins

I selected these parameters as during the lessons i have experimented a bit with these parameters and found that SVM model has best possible results with these settings. 

Color Space: With color space as either `YUV` or `RGB` the accuracy of the model was not more than 98%. So i switched to `YCrCb`, after this the accuracy increased to more than 99%. 

Orientation: I was able to get 99% training accuracy with 9 orientations but the car detection was not consistent and a few false targets were shown in the final output. So increased it a bit more to 11 i saw better results , after further trials i choose 16. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using LinearSVC. Training of the model was done in two stages.

1) Identify Optimal Parameters:
  To search for best parameters for maximum output accuracy i have used GridSearchCV. As i am using LinearSVC i only need to find optimal value for C. Code for this can be found in 5th code cell of 'P5_Vehicle_Detection.ipynb' IPython notebook .  

2) Use the Optimal Parameters train the final model.
   After finding the optimal parameters , using best_estimator_.C argument a new LinerSVC was created. The model was trained using 80% of the data. Trained model was tested using remaining 20% data. Code for this can be found in 6th code cell of 'P5_Vehicle_Detection.ipynb' IPython notebook .  

Training Data:
    Intially model was trained using GTI and KITTI dataset shared by UDACITY. Even though the model output was more than 99% white cars were not being detected properly and many false target were being detected by the model. To improve the model performance i have captured car and non-car the images from the project video. After this false targets have reduced and white car was being detected most of the time. However the size of the box was very small for white cars.
    Assuming that the model is still not able to detect the vehicle properly i searched for solution in the forums ,  I read in one of the forums that after training with the UDACITY dataset white car is being detected very well. I had downloaded UDACITY dataset1 and tried to extract the images but i was not succesfull in exacting the vehicle images properly from the dataset i might be doing something wrong. After trying for 2 days i gave up on the UDACITY dataset1.
    Finally i have increased the number of white car images to around 30 images after this the model was working properly. All these images were captured from the project video.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
The code for the sliding window search is reused from UDACITY course material.Functions slide_window and search_windows implemet the sliding window logic. Code for these functions is contained in the 7th code cell of the 'P5_Vehicle_Detection.ipynb' IPython notebook. 

I have tried to used HOG sub sampling method but i was not successfull , i did not explore further as there was limited time to finish the project. 

I have started the project by using search window size of 64x64 , with this configuration processing time was high , many false targets were seen and even though the cars were identified the resulting window was very small. So i have increase the scale and have strated searching image twice with different scaling. Finally 96x96 and 128x128 scales were selected , as any scales less than these resulted in false targets and smaller boxes.

![alt text][image6]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier? 

I took following steps to optimize the classifier:
- Used GridSearchCV to find the value of parameter C for which test image accuracy for highest. 
- Took several white car and black car image snapshots from the project video and used as training and test data.
- Took several trees and road image snapshots from the project video and used as training and test data.
- Used following parameters to extract the feautres from the images 
      1) HOG features of YCrCb 3-channel image with Orientation of 16 
      2) Spatial Binning with 32x32 size
      3) Color Histogram with 32 bins

Pipeline: Pipe line for test image processing is implemented in code cell 11 of the 'P5_Vehicle_Detection.ipynb' IPython notebook. Function process_image is used for test image processing. 
- Train the LINEARSVC with HOG, Spatial Bin and Color Histogram features. These features are extracted from the training images of size 64x64
- Search the image in which the cars have to be identified twice with 96x96 and 128x128 scale sizes. This search is done sequentially.
- After each search the resultant windows were used to create a blob of red color , i.e in an image, window where car was found is filled red rest of the image is blackened. 
- To increase the size of the box , if an area of picture has overlapping boxes showing the presence of the car then the overlapping area red intensitiy is increased so that the red blob size increases. 
- The boundaries of the red blob are found and these boundaries are used for drawing the box on in the final image.

Here are some example images:

:--------:|:------------:
![alt text][image7] | ![alt text][image8]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
Pipe line for video processing is implemented in code cell 13 of the 'P5_Vehicle_Detection.ipynb' IPython notebook. Function process_video is used for video processing. 
I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

With just the above implementation the box size was very small , to increase the box size i have increased the intensity of the blob where overlapping boxes are seen this implementation was taken from one of the UDACITY forum discussions and it can found in function add_heat of 8th Code cell of the 'P5_Vehicle_Detection.ipynb' IPython notebook.  

After this the final result was very wobbly , so i have implemented a `collections.deque` for size 15. All the previous 14 heatmap images are added to current heatmap and thresholding is applied. This thresolded heatmap is used for identifying the cars. 

Here's an example result showing the heatmap from a series of frames of video, the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image8]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced issue in identifying the white car properly , increasing the box size and identifying the car when the car is very far. To improve the car identification i have extracted car images from the project video , to increase the box size i have modified add_heat function to add extra weight where more than one overlapping boxes were found and have increase search scale size to 96x96 and 128x128. 

I was not very successful in tracking the vehicle when it is a bit far way , there was improvement when in introduced 2 search windows as mentioned above. Probably reducing the window size further and increasing the number search per image might help in identifying the far away vehicle. When i have implemented this the number of false targets have increased so my final code does not include this implementaion. 
