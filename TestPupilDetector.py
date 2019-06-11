import unittest
import pupildetector

class TestPupilDetector(unittest.TestCase):

	def __init__(self,img_file):
		self._test_img = cv2.imread(img_file,1)

	def test_detectPupil(self):
		self.assertEqual(0,0)

	def test_detectPupil(self):
		self.assertEqual(0,0)

	def test_detectEyes(self):
		self.assertEqual(0,0)

	def test_shapeToNumpy(self):
		self.assertEqual(0,0)


if __name__ == '__main__':
	unittest.main()