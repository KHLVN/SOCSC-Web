import cv2
import mediapipe as mp
import pyautogui

# Mediapipe Hand detection setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam feed
cap = cv2.VideoCapture(0)

# Function to calculate Euclidean distance
def distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

# Main loop
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally and convert to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks
        result = hands.process(rgb_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get landmarks
                landmarks = hand_landmarks.landmark
                
                # Extract coordinates of specific landmarks (index finger tip and thumb tip)
                h, w, _ = frame.shape
                index_tip = (int(landmarks[8].x * w), int(landmarks[8].y * h))
                thumb_tip = (int(landmarks[4].x * w), int(landmarks[4].y * h))
                
                # Draw points
                cv2.circle(frame, index_tip, 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, thumb_tip, 10, (0, 0, 255), cv2.FILLED)

                # Measure distance between index finger and thumb
                dist = distance(index_tip, thumb_tip)

                # Define threshold for gesture
                if dist < 40:  # Gesture: Fingers close together
                    pyautogui.press('right')  # Next slide
                elif index_tip[0] < w // 3:  # Gesture: Move left
                    pyautogui.press('left')  # Previous slide
                
        # Display the frame
        cv2.imshow("Presentation Control", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()