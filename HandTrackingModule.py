import cv2
import mediapipe as mp
import time

class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = int(maxHands)
        self.detectionCon = float(detectionCon)
        self.trackCon = float(trackCon)
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=1, 
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        
        # Defining the landmark connection skeleton map manually 
        # This completely avoids using the buggy self.mpDraw utility
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index Finger
            (5, 9), (9, 10), (10, 11), (11, 12),   # Middle Finger
            (9, 13), (13, 14), (14, 15), (15, 16), # Ring Finger
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky & Palm base
        ]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks and draw:
            h, w, c = img.shape
            for handLms in self.results.multi_hand_landmarks:
                # 1. Manually extract landmark coordinates to draw
                coords = {}
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    coords[id] = (cx, cy)
                
                # 2. Draw Green Connection Lines manually using native OpenCV
                for connection in self.connections:
                    start_pt = coords.get(connection[0])
                    end_pt = coords.get(connection[1])
                    if start_pt and end_pt:
                        cv2.line(img, start_pt, end_pt, (0, 255, 0), 2)
                
                # 3. Draw Red Joints manually using native OpenCV
                for id, pt in coords.items():
                    cv2.circle(img, pt, 4, (0, 0, 255), -1)
                    
        return img

    def findPosition(self, img, handNo=0):
        lmList = []
        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
        return lmList

def main():
    pTime = 0
    cTime = 0
    
    # Standard capture. If it freezes, you can change 0 to: cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(0) 
    
    if not cap.isOpened():
        print("Error: Webcam could not be reached.")
        return

    detector = handDetector()
    
    while cap.isOpened():
        success, img = cap.read()
        if not success or img is None:
            continue
            
        img = cv2.flip(img, 1)
        img = detector.findHands(img, draw=True)
        lmList = detector.findPosition(img)
        
        # Ensure we have the full array before reading properties
        if len(lmList) > 4:
            print(f"Thumb Tip (ID 4): {lmList[4]}")
            
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        
        cv2.putText(
            img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3
        )
        
        cv2.imshow("Hand Tracking Module", img)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
