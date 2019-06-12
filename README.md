# Pupil-Detector
To detect the center and area of the pupil in the given video

##Approach

Detecting the pupil is broken down to
1. Detecting the face
2. Dectecting the eye region
3. Detecting the pupil

For detecting the face, a viola jones classifier is used. The eye region is extracted from the detected face patch by HoG 5 point facial landmark detector of dlib library. https://github.com/davisking/dlib-models/blob/master/shape_predictor_5_face_landmarks.dat.bz2

To extract the eye patch from the eye region suggested by HoG predictor, the distance between the two corners of eye is used. Thus obtained eye patch is preprocesed before giving to a blob detector. Thus obtained center points and radius of pupil is used to visualize the pupil center and the area in the result video.

##Discussion
The approach is able to detect pupil center effectively in most of the frames in both with and without spectacles test scenarios. 
The detections are missed mostly in the cases where the pupil is partially covered by the eyelid. The approach works in realtime with approximately 15 fps.

##TODO
The execution time of the approach can be optimized. The pupil partial occlusion by the eye lid cases coudl be handled by some dynamic filtering approaches or by optimizing a non -linear least squares approach to fit a circle over the occluded pupil regions. The tracking of the pupil could also improve the performance. Deep learning approaches are one of the obvious ways to explore,


##Dependencies

* python 3.6
* dlib
* OpenCV

#Run

python main.py <input_video>


