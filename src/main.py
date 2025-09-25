import mediapipe as mp
from playsound3 import playsound
from mediapipe.framework.formats import landmark_pb2
import cv2
import os

MODEL_PATH = os.path.join("model", "gesture_recognizer.task")
SOUND_PATH = "sounds"

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

current_gesture = None
gesture = None


def get_result(result, output_image, timestamp_ms):
    global gesture, prev_gesture
    
    if len(result.gestures) != 0 and result.gestures[0][0].category_name != "None":
        gesture = result


def draw_hands(image):
    global gesture

    for hand_landmarks in gesture.hand_landmarks:
        landmarks = landmark_pb2.NormalizedLandmarkList()
        landmarks.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in hand_landmarks
            ]
        )

        mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    gesture = None


def process_gesture(gesture_type):
    match gesture_type:
        case "Closed_Fist":
            playsound(os.path.join(SOUND_PATH, "game-start.mp3"), block=False)
            print("Closed")

        case "Open_Palm":
            playsound(os.path.join(SOUND_PATH, "gun-shot.mp3"), block=False)
            print("Open")

        case "Pointing_Up":
            playsound(os.path.join(SOUND_PATH, "yay.mp3"), block=False)
            print("Point Up")

        case "Thumb_Down":
            print("Thumb Down")

        case "Thumb_Up":
            print("Thumb Up")

        case "Victory":
            print("Victory")

        case "ILoveYou":
            print("I LOVE U")

        case _:
            print("BAD BAD BAD")


# Create a gesture recognizer instance with the video mode:
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=get_result,
)

recognizer = GestureRecognizer.create_from_options(options)
cap = cv2.VideoCapture(0)
stamp = 0

# Repeatedly reading the webcam stream and processing the frame
while True:
    ret, frame = cap.read()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    recognition_result = recognizer.recognize_async(mp_image, stamp)

    # If a gesture is detected, we process the gesture type and draw the hand
    if gesture:
        # If the gesture is different than the previous gesture, we run the process function
        gesture_type = gesture.gestures[0][0].category_name
        if gesture_type != current_gesture or not current_gesture:
            process_gesture(gesture_type)

        current_gesture = gesture.gestures[0][0].category_name

        
        # Write the current gesture in the window and draw the hand connections
        cv2.putText(
            frame,
            current_gesture,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            4,
            cv2.LINE_AA,
        )

        draw_hands(frame)

    cv2.imshow("camera", frame)
    stamp += 1

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
