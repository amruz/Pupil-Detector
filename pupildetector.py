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




	def shapeToNumpy(self, shape, dtype="int"):
		# initialize the list of (x, y)-coordinates
		coords = np.zeros((shape.num_parts, 2), dtype=dtype)
		# loop over all facial landmarks and convert them
		# to a 2-tuple of (x, y)-coordinates
		for i in range(0, shape.num_parts):
			coords[i] = (shape.part(i).x, shape.part(i).y)

		# # return the list of (x, y)-coordinates
		return coords

	def detectEyes(self,gray_img):
		faces = self._face_cascade.detectMultiScale(gray_img, 1.3, 5)

		eyes_pos = []
		eyes = []

		if len(faces) != 0:
		    for (x,y,w,h) in faces:
		        faceRect = dlib.rectangle(x, y, x + w, y + h)
		        shape = self._predictor(gray_img, faceRect)
		        shape = self.shapeToNumpy(shape)
		        left_eye_distance = math.sqrt(math.pow((shape[0][1] - shape[1][1]),2) + math.pow((shape[0][0] - shape[1][0]),2))
		        right_eye_distance = math.sqrt(math.pow((shape[2][0] - shape[3][0]),2) + math.pow((shape[2][1] - shape[3][1]),2))

		        eyes.append(gray_img[shape[1][1]- int(left_eye_distance / 3): shape[0][1] + int(left_eye_distance / 3), shape[1][0] + int(left_eye_distance / 5): (shape[0][0] - int(left_eye_distance / 5))])
		        eyes_pos.append([shape[1][0] +int(left_eye_distance /  5), shape[1][1] - int(left_eye_distance / 3)])
		        eyes.append(gray_img[shape[2][1]-int(right_eye_distance / 3) : shape[3][1] + int(right_eye_distance / 3), shape[2][0] + int(right_eye_distance / 5):int(shape[3][0] - int(right_eye_distance / 5))])
		        eyes_pos.append([shape[2][0] +int(right_eye_distance / 5), shape[2][1] - int(right_eye_distance / 3)])

		return (eyes_pos,eyes)

	def drawDetections(self, image, main_key_points, eyes_pos):
		for j,keypoints in enumerate(main_key_points):
		        for i, keypoint in enumerate(keypoints):
		            area = "{0:.2f}".format(2*3.14*(keypoint.size/2)*(keypoint.size/2))
		            keypoint = cv2.KeyPoint(keypoint.pt[0] + eyes_pos[j][0] , keypoint.pt[1] + eyes_pos[j][1] , keypoint.size, keypoint.angle, keypoint.response, keypoint.octave, keypoint.class_id)
		            cv2.circle(image, (int(keypoint.pt[0]), int(keypoint.pt[1])), 1, (0, 0, 255), -1)
		            cv2.putText(image, str(area), (int(keypoint.pt[0]), int(keypoint.pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
		            cv2.circle(image, (int(keypoint.pt[0]), int(keypoint.pt[1])), int(keypoint.size/2), (0, 0, 255), 1)
		return image


	def detectPupil(self,frame):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		eyes_pos, eyes = self.detectEyes(gray)
		main_key_points = []
		for (i,eye) in enumerate(eyes):
			blur_img = cv2.bilateralFilter(eye,5,75,75)
			th3 = cv2.adaptiveThreshold(blur_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,7,2)
			params = cv2.SimpleBlobDetector_Params()
			detector = cv2.SimpleBlobDetector_create(params)      
			inverted_img = cv2.dilate(th3, None, iterations=2) #2
			inverted_img = cv2.erode(inverted_img, None, iterations=3) #1   # Detect blobs.
			keypoints = detector.detect(inverted_img)
			main_key_points.append(keypoints)
		detected_image = self.drawDetections(img, main_key_points, eyes_pos)
		return (len(main_key_points), detected_image)

	def readFrames(self):
		cap = cv2.VideoCapture(self._input_video)
		while 1:
			ret, frame = cap.read()
			if (ret == False):
				return
			frame = cv2.rotate(frame, 2)
			_, result = self.detectPupil(frame)
			cv2.imshow('Detected Image', result)
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
	pupil_detect = PupilDetector(args.input)
	pupil_detect.readFrames()

main(sys.argv[1:])