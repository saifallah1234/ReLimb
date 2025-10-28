import cv2 as cv
from cvzone.PoseModule import PoseDetector

capture = cv.VideoCapture('ressources/walking_white.mp4')
detector = PoseDetector(staticMode=False,
                        modelComplexity=1,
                        smoothLandmarks=True,
                        enableSegmentation=False,
                        smoothSegmentation=True,
                        detectionCon=0.5,
                        trackCon=0.5)
while True:
    isTrue, frame = capture.read()
    frame = detector.findPose(frame)
    lmList, bboxInfo = detector.findPosition(frame, draw=True, bboxWithHands=False)
    if lmList:
        # Get the center of the bounding box around the body
        center = bboxInfo["center"]

        # Draw a circle at the center of the bounding box
        cv.circle(frame, center, 5, (255, 0, 255), cv.FILLED)
        cv.circle(
    frame,                   # Image to draw on
    lmList[27][0:2],            # Center coordinates (x, y)
    100,                   # Radius in pixels
    (0, 0, 255),           # Color in BGR -> here green
    thickness=3   )         # Border thickness (use -1 to fill the circle)


        # Calculate the distance between landmarks 11 and 15 and draw it on the image
        """length, frame, info = detector.findDistance(lmList[27][0:2],
                                                  lmList[28][0:2],
                                                  img=frame,
                                                  color=(255, 0, 0),
                                                  scale=10)
        print(f'info: {info} with length of: {length}')"""

        # Calculate the angle between landmarks 11, 13, and 15 and draw it on the image
        """angle, frame = detector.findAngle(lmList[11][0:2],
                                        lmList[13][0:2],
                                        lmList[15][0:2],
                                        img=frame,
                                        color=(0, 0, 255),
                                        scale=10)

        # Check if the angle is close to 50 degrees with an offset of 10
        isCloseAngle50 = detector.angleCheck(myAngle=angle,
                                             targetAngle=50,
                                             offset=10)

        # Print the result of the angle check
        print(isCloseAngle50)"""

    cv.imshow('Pose Detector', frame)
    if cv.waitKey(50) & 0xFF == ord('d'):
        break
capture.release()
cv.destroyAllWindows()