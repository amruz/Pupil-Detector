import sys
import argparse
import numpy as np
import cv2
import dlib
import math

class PupilDetector(object):
	"""
	A class used to detect the center and area of the pupil in the given input
	...
	Attributes
	-------------
	face_cascade 

	Methods
	--------------
	shape_to_np 



	"""
	def __init__(self,input_video):
		self._input_video = input_video
		self._face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		self._predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
		self._hogFaceDetector = dlib.get_frontal_face_detector()




	def shape_to_np(self, shape, dtype="int"):
		# initialize the list of (x, y)-coordinates
		coords = np.zeros((shape.num_parts, 2), dtype=dtype)
		# loop over all facial landmarks and convert them
		# to a 2-tuple of (x, y)-coordinates
		for i in range(0, shape.num_parts):
			coords[i] = (shape.part(i).x, shape.part(i).y)

		# # return the list of (x, y)-coordinates
		return coords

	def DetectEyes(self,gray_img):
		faces = self._face_cascade.detectMultiScale(gray_img, 1.3, 5)

		eyes_pos = []
		eyes = []

		if len(faces) != 0:
		    #for faceRect in faceRects:
		    for (x,y,w,h) in faces:
		        faceRect = dlib.rectangle(x, y, x + w, y + h)
		        #print (faceRect)
		        shape = self._predictor(gray_img, faceRect)
		        shape = self.shape_to_np(shape)
		        #print (shape)
		        # eyes.append(gray_img[ shape[1][1]-20: (shape[0][1]+10), shape[1][0]: (shape[0][0])])
		        # eyes_pos.append([shape[1][0], shape[1][1]])
		        # eyes.append(gray_img[ shape[2][1]-20: (shape[3][1]+10), shape[2][0]: (shape[3][0])])
		        # eyes_pos.append([shape[2][0], shape[2][1]])
		        left_eye_distance = math.sqrt(math.pow((shape[0][1] - shape[1][1]),2) + math.pow((shape[0][0] - shape[1][0]),2))
		        right_eye_distance = math.sqrt(math.pow((shape[2][0] - shape[3][0]),2) + math.pow((shape[2][1] - shape[3][1]),2))

		        eyes.append(gray_img[shape[1][1]- int(left_eye_distance / 3): shape[0][1] + int(left_eye_distance / 3), shape[1][0] + int(left_eye_distance / 5): (shape[0][0] - int(left_eye_distance / 5))])
		        eyes_pos.append([shape[1][0] +int(left_eye_distance /  5), shape[1][1] - int(left_eye_distance / 3)])
		        eyes.append(gray_img[shape[2][1]-int(right_eye_distance / 3) : shape[3][1] + int(right_eye_distance / 3), shape[2][0] + int(right_eye_distance / 5):int(shape[3][0] - int(right_eye_distance / 5))])
		        eyes_pos.append([shape[2][0] +int(right_eye_distance / 5), shape[2][1] - int(right_eye_distance / 3)])

		return (eyes_pos,eyes)

	def DrawDetections(self, image, main_key_points, eyes_pos):
		for j,keypoints in enumerate(main_key_points):
		        for i, keypoint in enumerate(keypoints):
		            area = "{0:.2f}".format(2*3.14*(keypoint.size/2)*(keypoint.size/2))
		            keypoint = cv2.KeyPoint(keypoint.pt[0] + eyes_pos[j][0] , keypoint.pt[1] + eyes_pos[j][1], keypoint.size, keypoint.angle, keypoint.response, keypoint.octave, keypoint.class_id)
		            cv2.circle(image, (int(keypoint.pt[0]), int(keypoint.pt[1])), 1, (0, 0, 255), -1)
		            cv2.putText(image, str(area), (int(keypoint.pt[0]), int(keypoint.pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
		            cv2.circle(image, (int(keypoint.pt[0]), int(keypoint.pt[1])), int(keypoint.size/2), (0, 0, 255), 1)
		cv2.imshow('Result',image)


	def DetectPupil(self):
		cap = cv2.VideoCapture(self._input_video)
		#cap = cv2.VideoCapture(0)
		while 1:
			ret, img = cap.read()
			if (ret == False):
				return
			img = cv2.rotate(img, 2)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			eyes_pos, eyes = self.DetectEyes(gray) 
		        #thresholding
			main_key_points = []
			for i,eye in enumerate(eyes):
				blur = cv2.bilateralFilter(eye,5,75,75)
				th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,7,2)
				params = cv2.SimpleBlobDetector_Params()
				detector = cv2.SimpleBlobDetector_create(params)      
				inverted_img = cv2.dilate(th3, None, iterations=2) #2
				inverted_img = cv2.erode(inverted_img, None, iterations=3) #1   # Detect blobs.
				keypoints = detector.detect(inverted_img)
				main_key_points.append(keypoints)
			self.DrawDetections(img, main_key_points, eyes_pos) 
			k = cv2.waitKey(30) & 0xff
			if k == 27:
				break
		

		cap.release()
		cv2.destroyAllWindows()


def main(argv):
	inputfile = ''
	parser = argparse.ArgumentParser()
	parser.add_argument('input',metavar='i',help='Input video file')
	args = parser.parse_args()
	print (args)
	pupildetect = PupilDetector(args.input)
	pupildetect.DetectPupil()

main(sys.argv[1:])