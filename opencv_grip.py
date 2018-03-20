import numpy
import math
import cv2
from enum import Enum
from grip import GripPipeline
from networktables import NetworkTables

NetworkTables.initialize(server='roborio-85-frc.local')

cam=cv2.VideoCapture(0)
oPipe = GripPipeline()

table = NetworkTables.getTable('GRIP/myContoursReport')

success=True
while True:
	success,img=cam.read()
	if success:
		continue

        try:
		oPipe.process(img)
		for mxContour in oPipe.filter_contours_output:
        		x,y,w,h = cv2.boundingRect(mxContour)
        		table.putString('FoundValues', ' x={}  y={}  w={}  h={}').format(x,y,w,h)
			break
        except cv2.error as ex:
		print('Error: ', ex)
