import sys
import argparse
import numpy as np
import cv2
import dlib
import math

class PupilDetector(object):
	"""
	To detect the center and area of the pupil in the given input video.
	"""
	def __init__(self, input_video = ""):
		# input video
		self._input_video    = input_video
		# the face cascade classifier
		self._face_cascade   = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		# eye feature predictor
		self._predictor      = dlib.shape_predictor('eye_predictor_landmarks.dat')
		# blob detector
		self._detector       = cv2.SimpleBlobDetector_create() 
		# eye patch positions in each frame
		self._eyes_pos       = []
		# array of keypoints of the eye patches
		self.main_key_points = []


	def shapeToNumpy(self, shape, dtype="int"):
		"""Convert shape type to numpy array.

    	Keyword arguments:
    	shape -- input shape
    	dtype -- type of array

    	Return:
		coords -- numpy array
    	"""
		# initialize the list of (x, y)-coordinates
		coords = np.zeros((shape.num_parts, 2), dtype=dtype)
		# loop over all facial landmarks and convert them
		# to a 2-tuple of (x, y)-coordinates
		for i in range(0, shape.num_parts):
			coords[i] = (shape.part(i).x, shape.part(i).y)

		# # return the list of (x, y)-coordinates
		return coords

	def detectEyes(self, gray_img):
		"""Detect eyes from input gray image after detecting the face.

    	Keyword arguments:
    	gray_img -- input gray image

    	Return:
		eyes -- array of eye patches detected
    	"""
    	# detect face
		faces = self._face_cascade.detectMultiScale(gray_img, 1.3, 5)

		self._eyes_pos = []
		eyes = []
		# detect eyes only if face detected
		if len(faces) != 0:
		    for (x, y, w, h) in faces:
		        faceRect = dlib.rectangle(x, y, x + w, y + h)
		        shape = self._predictor(gray_img, faceRect)
		        shape = self.shapeToNumpy(shape)
		        # extract the eye patch from the image
		        left_eye_distance = math.sqrt(math.pow((shape[0][1] - shape[1][1]), 2) + math.pow((shape[0][0] - shape[1][0]), 2))
		        right_eye_distance = math.sqrt(math.pow((shape[2][0] - shape[3][0]), 2) + math.pow((shape[2][1] - shape[3][1]), 2))

		        eyes.append(gray_img[shape[1][1]- int(left_eye_distance / 3): shape[0][1] + int(left_eye_distance / 3), 
		        	shape[1][0] + int(left_eye_distance / 5): (shape[0][0] - int(left_eye_distance / 5))])
		        self._eyes_pos.append([shape[1][0] + int(left_eye_distance / 5), shape[1][1] - int(left_eye_distance / 3)])
		        eyes.append(gray_img[shape[2][1]- int(right_eye_distance / 3) : shape[3][1] + int(right_eye_distance / 3), 
		        	shape[2][0] + int(right_eye_distance / 5):int(shape[3][0] - int(right_eye_distance / 5))])
		        self._eyes_pos.append([shape[2][0] + int(right_eye_distance / 5), shape[2][1] - int(right_eye_distance / 3)])
		return (eyes)

	def drawDetections(self, image):
		"""Draw the detected pupil center and the area to each input frame.

    	Keyword arguments:
    	image -- input frame

    	Return:
		image -- image drawn with detected pupil center and area 
    	"""
		for j, keypoints in enumerate(self.main_key_points):
		        for i, keypoint in enumerate(keypoints):
		        	#computing the area of the detected pupil
		            area = "{0:.2f}".format(math.pi * math.pow(keypoint.size/2, 2))
		            keypoint = cv2.KeyPoint(keypoint.pt[0] + self._eyes_pos[j][0] , keypoint.pt[1] + self._eyes_pos[j][1] , 
		            	keypoint.size, keypoint.angle, keypoint.response, keypoint.octave, keypoint.class_id)
		            cv2.circle(image, (int(keypoint.pt[0]), int(keypoint.pt[1])), 1, (0, 0, 255), -1)
		            cv2.putText(image, str(area), (int(keypoint.pt[0]), int(keypoint.pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
		            cv2.circle(image, (int(keypoint.pt[0]), int(keypoint.pt[1])), int(keypoint.size/2), (0, 0, 255), 1)
		return image


	def detectPupil(self, frame):
		"""Detected pupil after detecting the eyes in each frame .

    	Keyword arguments:
    	frame -- input frame

    	Return:
		detected_image -- the image frame with pupil detection drawn
    	"""
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# detect eyes
		eyes = self.detectEyes(gray)
		self.main_key_points = []

		for (i, eye) in enumerate(eyes):
			# blurring the image
			blur_img = cv2.bilateralFilter(eye, 5, 75, 75)
			# thresholding the blurred image
			thresholded_image = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)
			# image opening   
			inverted_img = cv2.dilate(thresholded_image, None, iterations=2) #2
			inverted_img = cv2.erode(inverted_img, None, iterations=3) #1   
			# Detect blobs.
			keypoints = self._detector.detect(inverted_img)
			self.main_key_points.append(keypoints)
		detected_image = self.drawDetections(frame)
		return (detected_image)

	def readFrames(self):
		"""
		To read each frames of the input video
    	"""
		cap = cv2.VideoCapture(self._input_video)
		while 1:
			ret, frame = cap.read()
			# if no frames
			if (ret == False):
				return
			frame = cv2.rotate(frame, 2)
			result = self.detectPupil(frame)
			cv2.imshow('Detected Image', result)
			k = cv2.waitKey(30) & 0xff
			if k == 27:
				break
		cap.release()
		cv2.destroyAllWindows()