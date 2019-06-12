import unittest
import cv2
from pupil_detector import PupilDetector

class TestPupilDetector(unittest.TestCase):
	"""
	Test class for pupil detector
	"""
	def setUp(self):
		"""
		Initialize params 
		"""
		self.pupil_detector  = PupilDetector()
		self._test_img       = cv2.imread('test.png',1)
		self._test_fail_img  = cv2.imread('test_fail.png',1)

	def test_detectPupil(self):
		"""
		Verifies the detection of pupil in test image and
		verifies the pupil centre in each eye patch
		"""
		self.pupil_detector.detectPupil(self._test_img)
		self.assertEqual(2, len(self.pupil_detector.main_key_points))
		self.assertEqual(9.368627548217773, self.pupil_detector.main_key_points[0][0].pt[0])
		self.assertEqual(9.0, self.pupil_detector.main_key_points[0][0].pt[1])
		self.assertEqual(11.5, self.pupil_detector.main_key_points[1][0].pt[0])
		self.assertEqual(8.0, self.pupil_detector.main_key_points[1][0].pt[1])  

		self.pupil_detector.detectPupil(self._test_fail_img)
		self.assertEqual(0, len(self.pupil_detector.main_key_points))

	def test_detectEyes(self):
		"""
		Verifies the detection of eyes in the image
		"""
		gray_img = cv2.cvtColor(self._test_img, cv2.COLOR_BGR2GRAY)
		eyes = self.pupil_detector.detectEyes(gray_img)
		self.assertEqual(2, len(eyes))

		gray_img = cv2.cvtColor(self._test_fail_img, cv2.COLOR_BGR2GRAY)
		eyes = self.pupil_detector.detectEyes(gray_img)
		self.assertEqual(0, len(eyes))


if __name__ == '__main__':
	unittest.main()