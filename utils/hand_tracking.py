import mediapipe as mp
import cv2

class HandDetector:
    def __init__(self):
        # initialise mediapipe hand detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils
        
    def fingers_up(self, frame):
        # convert frame to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        # if hands are detected, carry on
        if results.multi_hand_landmarks:
            fingers_up = []
            
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                h, w, _ = frame.shape
                
                # Thumb: check if the tip is above the base
                if landmarks[4].x < landmarks[3].x:
                    fingers_up.append(1) # thumb up
                else:
                    fingers_up.append(0) # thumb down
                    
                # Index, middle. ring and pinky fingers check
                finger_tips = [8, 12, 16, 20]
                finger_bases = [6, 10, 14, 19]
                
                for tip, base in zip(finger_tips, finger_bases):
                    if landmarks[tip].y < landmarks[base].y:
                        fingers_up.append(1) # finger up
                    else:
                        fingers_up.append(0) # finger down
            
            return fingers_up
    
        return []