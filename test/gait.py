import cv2 as cv
from cvzone.PoseModule import PoseDetector
import numpy as np

capture = cv.VideoCapture('ressources/walking_white.mp4')
detector = PoseDetector()
previous_left_ankle_y, previous_right_ankle_y = None, None
asymmetry_scores = []

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break
    frame = cv.resize(frame, (640, 480))
    frame = detector.findPose(frame)
    lmList, bboxInfo = detector.findPosition(frame, draw=True, bboxWithHands=False)

    if lmList:
        # Get key landmarks for both legs
        left_hip, left_knee, left_ankle = lmList[23], lmList[25], lmList[27]
        right_hip, right_knee, right_ankle = lmList[24], lmList[26], lmList[28]

        # Calculate knee angles for both legs
        left_angle, _ = detector.findAngle(left_hip[0:2], left_knee[0:2], left_ankle[0:2], img=frame, color=(0,255,0))
        right_angle, _ = detector.findAngle(right_hip[0:2], right_knee[0:2], right_ankle[0:2], img=frame, color=(255,0,0))

        # Calculate difference in knee bending (asymmetry)
        diff = abs(left_angle - right_angle)
        asymmetry_scores.append(diff)

        cv.putText(frame, f"Asymmetry: {int(diff)} deg", (20, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Check if one leg is lagging (simple vertical movement check)
        if previous_left_ankle_y and previous_right_ankle_y:
            left_move = abs(left_ankle[1] - previous_left_ankle_y)
            right_move = abs(right_ankle[1] - previous_right_ankle_y)

            ratio = left_move / (right_move + 1e-5)
            cv.putText(frame, f"Step ratio: {ratio:.2f}", (20, 100),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            # Detect lameness (for example, ratio < 0.5 or > 1.5)
            if ratio < 0.5 or ratio > 1.5:
                cv.putText(frame, "⚠️ Possible limp detected", (20, 140),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        previous_left_ankle_y, previous_right_ankle_y = left_ankle[1], right_ankle[1]

    cv.imshow("Gait Analysis", frame)
    if cv.waitKey(1) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()

# Print average asymmetry
if asymmetry_scores:
    print("Average Asymmetry Score:", np.mean(asymmetry_scores))
