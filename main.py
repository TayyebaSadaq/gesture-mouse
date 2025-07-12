import cv2
import mediapipe as mp
from utils.hand_tracking import HandDetector
import pyautogui

# # initialise mediapipe
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
# mp.draw = mp.solutions.drawing_utils
detector = HandDetector()
 
# start the webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # # flip and convert to RGB
    frame = cv2.flip(frame, 1)
    # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # # process the frames
    # results = hands.process(rgb)

    # # if hands are detected
    # if results.multi_hand_landmarks:
    #     for hand_landmarks in results.multi_hand_landmarks:
    #         mp.draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    fingers, lm_list = detector.fingers_up(frame)
    print(f"Fingers up: {fingers}")
    
    if fingers == [0,1,0,0,0]: # only index finger up
        direction = detector.detect_scroll_direction(lm_list)
        if direction == "up":
            pyautogui.scroll(100)
        elif direction == "down":
            pyautogui.scroll(-100)
    
    # show the frame
    cv2.imshow("Gesture Mouse", frame)

    # quit on "q" press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()