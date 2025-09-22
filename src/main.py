import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import cv2
import os

MODEL_PATH = os.path.join("model", "gesture_recognizer.task")

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
gestures = None

def print_result(result, output_image, timestamp_ms):
    global gestures

    if len(result.gestures) != 0 and result.gestures[0][0].category_name != "None":
        gestures = result.hand_landmarks
        print(
            "gesture recognition result: {}".format(result.gestures[0][0].category_name)
        )
        

def draw_hands(image):
    global gestures

    for gesture in gestures:
        landmarks = landmark_pb2.NormalizedLandmarkList()
        landmarks.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in gesture
        ])

        mp_drawing.draw_landmarks(
            frame, landmarks, mp_hands.HAND_CONNECTIONS
        )

    gestures = None


# Create a gesture recognizer instance with the video mode:
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
)

recognizer = GestureRecognizer.create_from_options(options)
cap = cv2.VideoCapture(0)
stamp = 0

while True:
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (400, 400))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    recognition_result = recognizer.recognize_async(mp_image, stamp)
    
    if gestures:
        draw_hands(frame)
        
    cv2.imshow("camera", frame)
    stamp += 1

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
