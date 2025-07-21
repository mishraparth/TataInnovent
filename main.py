# main.py (Final Version with Voice Commands)

import cv2
import mediapipe as mp
import math
import threading
from playsound import playsound
import speech_recognition as sr  # NEW: For voice commands

# --- Constants ---
EAR_THRESHOLD = 0.20
CONSECUTIVE_FRAMES_THRESHOLD = 10
ALARM_SOUND_PATH = "assets/sounds/alert.mp3"

# --- Visuals ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (255, 255, 255)
AWAKE_COLOR = (0, 255, 0)
DROWSY_COLOR = (0, 0, 255)
DISABLED_COLOR = (80, 80, 80)

# --- Global State Variables ---
consecutive_frames_counter = 0
alarm_on = False
alert_system_enabled = True  # NEW: Flag to control the alert system


# --- Helper Functions ---
def euclidean_distance(point1, point2):
    x1, y1 = point1;
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_ear(eye, frame_width, frame_height):
    p1 = (int(eye[0].x * frame_width), int(eye[0].y * frame_height))
    p2 = (int(eye[1].x * frame_width), int(eye[1].y * frame_height))
    p3 = (int(eye[2].x * frame_width), int(eye[2].y * frame_height))
    p4 = (int(eye[3].x * frame_width), int(eye[3].y * frame_height))
    p5 = (int(eye[4].x * frame_width), int(eye[4].y * frame_height))
    p6 = (int(eye[5].x * frame_width), int(eye[5].y * frame_height))
    vertical_dist1 = euclidean_distance(p2, p6)
    vertical_dist2 = euclidean_distance(p3, p5)
    horizontal_dist = euclidean_distance(p1, p4)
    if horizontal_dist == 0: return 0.0
    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
    return ear


def play_alarm_sound():
    playsound(ALARM_SOUND_PATH)


# --- NEW: VOICE COMMANDS FUNCTION ---
def process_voice_command():
    global alert_system_enabled
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for a command (e.g., 'enable alert' or 'disable alert')...")
        r.pause_threshold = 1.5
        audio = r.listen(source)

    try:
        print("Recognizing...")
        command = r.recognize_google(audio).lower()
        print(f"You said: {command}")

        if "enable" in command:
            alert_system_enabled = True
            print("Alert system has been ENABLED.")
        elif "disable" in command:
            alert_system_enabled = False
            print("Alert system has been DISABLED.")

    except sr.UnknownValueError:
        print("Could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")


# --- Main Program ---
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    frame_height, frame_width, _ = frame.shape
    results = face_mesh.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    status_text = "STATUS: NO FACE DETECTED"
    status_color = DISABLED_COLOR

    if results.multi_face_landmarks and alert_system_enabled:
        face_landmarks = results.multi_face_landmarks[0].landmark
        left_eye_lm = [face_landmarks[i] for i in LEFT_EYE_INDICES]
        right_eye_lm = [face_landmarks[i] for i in RIGHT_EYE_INDICES]

        # Draw eye landmarks
        for lm_list in [left_eye_lm, right_eye_lm]:
            for landmark in lm_list:
                point = (int(landmark.x * frame_width), int(landmark.y * frame_height))
                cv2.circle(frame, point, 1, AWAKE_COLOR, -1)

        left_ear = calculate_ear(left_eye_lm, frame_width, frame_height)
        right_ear = calculate_ear(right_eye_lm, frame_width, frame_height)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            consecutive_frames_counter += 1
        else:
            consecutive_frames_counter = 0
            alarm_on = False

        status_text = "STATUS: AWAKE"
        status_color = AWAKE_COLOR

        if consecutive_frames_counter >= CONSECUTIVE_FRAMES_THRESHOLD:
            status_text = "STATUS: DROWSY"
            status_color = DROWSY_COLOR
            if not alarm_on:
                alarm_on = True
                alarm_thread = threading.Thread(target=play_alarm_sound)
                alarm_thread.start()

        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (frame_width - 150, 65), FONT, 0.7, TEXT_COLOR, 2)
    elif not alert_system_enabled:
        status_text = "ALERTS DISABLED"
        status_color = DISABLED_COLOR

    # Draw the main status bar
    cv2.rectangle(frame, (0, 0), (frame_width, 40), status_color, -1)
    cv2.putText(frame, status_text, (10, 25), FONT, 0.7, TEXT_COLOR, 2)

    # NEW: Draw a smaller status bar for the alert system state
    alert_sys_text = "Alerts: ON" if alert_system_enabled else "Alerts: OFF"
    alert_sys_color = AWAKE_COLOR if alert_system_enabled else DROWSY_COLOR
    cv2.rectangle(frame, (0, 45), (150, 80), alert_sys_color, -1)
    cv2.putText(frame, alert_sys_text, (10, 65), FONT, 0.7, TEXT_COLOR, 2)
    cv2.putText(frame, "[C] to command", (10, frame_height - 10), FONT, 0.5, (255, 255, 0), 2)

    cv2.imshow('AI Safety Co-Pilot', frame)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):  # NEW: Listen for 'c' key to trigger voice command
        process_voice_command()

cap.release()
cv2.destroyAllWindows()
