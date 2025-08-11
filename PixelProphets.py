import cv2
import os
from datetime import datetime

SAVE_DIR = "/Users/luisrafaelarguelles/Documents/Cambridge/Workshops/Python Intro/stedmund-summer-python-workshop/Project"
os.makedirs(SAVE_DIR, exist_ok=True)
buttons = {
    "Capture": (50, 50, 150, 100),
    "Quit":    (200, 50, 300, 100),
    "Rock Paper Scissors":    (350, 50, 600, 100),
    "Tic Tac Toe": (650, 50, 850, 100),
}

# Mouse
running = True
current_frame = None 
show_ttt = False  

def draw_buttons(frame):
    for label, (x1, y1, x2, y2) in buttons.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.putText(frame, label, (x1 + 5, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

def mouse_callback(event, x, y, flags, param):
    global running, current_frame, show_ttt
    if event == cv2.EVENT_LBUTTONDOWN:
        for label, (x1, y1, x2, y2) in buttons.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                print(f"{label} button clicked")
                if label == "Quit":
                    running = False
                elif label == "Capture" and current_frame is not None:
                    filename = f"photo_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                    filepath = os.path.join(SAVE_DIR, filename)
                    cv2.imwrite(filepath, current_frame) 
                    print(f"Saved: {filepath}")
                elif label == "Tic Tac Toe":
                    show_ttt = not show_ttt

def draw_tic_tac_toe(frame):
    """Draw a 3x3 grid centered on the frame."""
    h, w = frame.shape[:2]
    size = int(min(w, h) * 0.6)        
    cx, cy = w // 2, h // 2       
    half = size // 2
    x1, y1 = cx - half, cy - half      
    x2, y2 = cx + half, cy + half      

    step = size // 3
    color = (255, 255, 255)               
    thickness = 3

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    cv2.line(frame, (x1 + step, y1), (x1 + step, y2), color, thickness)
    cv2.line(frame, (x1 + 2 * step, y1), (x1 + 2 * step, y2), color, thickness)

    cv2.line(frame, (x1, y1 + step), (x2, y1 + step), color, thickness)
    cv2.line(frame, (x1, y1 + 2 * step), (x2, y1 + 2 * step), color, thickness)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
cv2.namedWindow("Webcam")
cv2.setMouseCallback("Webcam", mouse_callback)

if not cap.isOpened():
    print("Error: Could not access the camera.")
else:
    print("Click the on-screen buttons. You can also press 'c' to capture or 'q' to quit.")
    while running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        current_frame = frame.copy()

        draw_buttons(frame)

        if show_ttt:
            draw_tic_tac_toe(frame)

        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and current_frame is not None:
            filename = f"photo_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, current_frame)
            print(f"Saved: {filepath}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()












