import mediapipe as mp
import cv2

class HandDetector:
    def __init__(self):
        # initialise mediapipe hand detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils
        self.prev_index_y = None
        
    def fingers_up(self, frame):
        # convert frame to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        lm_list = [] 
        
        # if hands are detected, carry on
        if results.multi_hand_landmarks:
            fingers_up = []
            
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                h, w, _ = frame.shape
                
                 # Save all landmarks to list
                for id, lm in enumerate(landmarks):
                    x, y = int(lm.x * w), int(lm.y * h)
                    lm_list.append((id, x, y))
                    
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
            
            return fingers_up, lm_list
    
        return [], []
    
    def detect_scroll_direction(self, lm_list):
        if len(lm_list) == 0:
            return None # no hand detected
        
        index_y = lm_list[8][2] # coords of index finger tip
        direction = None
        
        if self.prev_index_y is not None:
            dy = index_y - self.prev_index_y
            
            if abs(dy) > 20: # sensitivity threshold
                if dy > 0:
                    direction = "down"
                else:
                    direction = "up"
        
        self.prev_index_y = index_y
        return direction