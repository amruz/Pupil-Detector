import sys
import argparse

from pupil_detector import PupilDetector

def main(argv):
	inputfile = ''
	parser = argparse.ArgumentParser()
	parser.add_argument('input',metavar='i',help='Input video file')
	args = parser.parse_args()
	pupil_detect = PupilDetector(args.input)
	pupil_detect.readFrames()

main(sys.argv[1:])
