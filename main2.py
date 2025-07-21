# main.py (Final Version with On-Screen Messages)

import cv2
import mediapipe as mp
import math
import threading
from playsound import playsound
import speech_recognition as sr

# --- Constants ---
EAR_THRESHOLD = 0.18
CONSECUTIVE_FRAMES_THRESHOLD = 20
ALARM_SOUND_PATH = "assets/sounds/alert.mp3"  # Using .wav is more reliable

# --- Visuals ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (255, 255, 255)
AWAKE_COLOR = (0, 255, 0)
DROWSY_COLOR = (0, 0, 255)
DISABLED_COLOR = (80, 80, 80)
MESSAGE_COLOR = (0, 255, 255)  # Yellow for messages

# --- Global State Variables ---
consecutive_frames_counter = 0
alarm_on = False
alert_system_enabled = True
# NEW: Variables for the on-screen message system
display_message = ""
message_timer = 0


# --- Helper Functions ---
def euclidean_distance(point1, point2):
    x1, y1 = point1;
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_ear(eye, frame_width, frame_height):
    p1 = (int(eye[0].x * frame_width), int(eye[0].y * frame_height));
    p2 = (int(eye[1].x * frame_width), int(eye[1].y * frame_height))
    p3 = (int(eye[2].x * frame_width), int(eye[2].y * frame_height));
    p4 = (int(eye[3].x * frame_width), int(eye[3].y * frame_height))
    p5 = (int(eye[4].x * frame_width), int(eye[4].y * frame_height));
    p6 = (int(eye[5].x * frame_width), int(eye[5].y * frame_height))
    vertical_dist1 = math.hypot(p2[0] - p6[0], p2[1] - p6[1]);
    vertical_dist2 = math.hypot(p3[0] - p5[0], p3[1] - p5[1])
    horizontal_dist = math.hypot(p1[0] - p4[0], p1[1] - p4[1])
    return 0.0 if horizontal_dist == 0 else (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)


def play_alarm_sound():
    try:
        playsound(ALARM_SOUND_PATH)
    except Exception as e:
        print(f"Error playing sound: {e}")


# --- VOICE COMMANDS FUNCTION (UPDATED) ---
def process_voice_command():
    global alert_system_enabled, display_message, message_timer
    r = sr.Recognizer();
    r.pause_threshold = 1.5

    display_message = "Listening...";
    message_timer = 150  # Show "Listening..." on screen
    print("Listening...")

    with sr.Microphone() as source:
        try:
            audio = r.listen(source, timeout=5)
        except sr.WaitTimeoutError:
            display_message = "Listening timed out.";
            message_timer = 150
            return

    try:
        display_message = "Recognizing...";
        message_timer = 150  # Show "Recognizing..." on screen
        print("Recognizing...")
        command = r.recognize_google(audio).lower()

        display_message = f"You said: {command}";
        message_timer = 150  # Show the recognized command
        print(f"You said: {command}")

        if "enable" in command:
            alert_system_enabled = True
        elif "disable" in command:
            alert_system_enabled = False
        elif "sarthi" in command:
            display_message = "feature under development";
            message_timer = 150
        elif "help" in command:
            display_message = "feature under development";
            message_timer = 150
        elif "radio" in command:
            display_message = "feature under development";
            message_timer = 150


    except sr.UnknownValueError:
        display_message = "Could not understand audio.";
        message_timer = 150
    except sr.RequestError:
        display_message = "Speech service unavailable.";
        message_timer = 150


# --- Main Program ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    frame = cv2.resize(frame, (854, 480))
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    avg_ear = None
    status_text = "STATUS: NO FACE DETECTED"
    status_color = DISABLED_COLOR

    if not results.multi_face_landmarks:
        consecutive_frames_counter = 0;
        alarm_on = False

    if results.multi_face_landmarks and alert_system_enabled:
        face_landmarks = results.multi_face_landmarks[0].landmark
        left_eye_lm = [face_landmarks[i] for i in LEFT_EYE_INDICES];
        right_eye_lm = [face_landmarks[i] for i in RIGHT_EYE_INDICES]
        left_ear = calculate_ear(left_eye_lm, frame_width, frame_height);
        right_ear = calculate_ear(right_eye_lm, frame_width, frame_height)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            consecutive_frames_counter += 1
        else:
            consecutive_frames_counter = 0;
            alarm_on = False

        landmark_color = AWAKE_COLOR
        status_text = "STATUS: AWAKE"
        status_color = AWAKE_COLOR

        if consecutive_frames_counter >= CONSECUTIVE_FRAMES_THRESHOLD:
            landmark_color = DROWSY_COLOR
            status_text = "STATUS: DROWSY"
            status_color = DROWSY_COLOR
            if not alarm_on:
                alarm_on = True
                threading.Thread(target=play_alarm_sound).start()
                consecutive_frames_counter = 0

        for lm_list in [left_eye_lm, right_eye_lm]:
            for landmark in lm_list:
                point = (int(landmark.x * frame_width), int(landmark.y * frame_height));
                cv2.circle(frame, point, 1, landmark_color, -1)

    elif not alert_system_enabled:
        status_text = "ALERTS DISABLED";
        status_color = DISABLED_COLOR

    # --- Draw UI ---
    cv2.rectangle(frame, (0, 0), (frame_width, 40), status_color, -1)
    cv2.putText(frame, status_text, (10, 25), FONT, 0.7, TEXT_COLOR, 2)
    if avg_ear is not None:
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (frame_width - 120, 25), FONT, 0.7, TEXT_COLOR, 2)

    # NEW: Draw the message on screen if the timer is active
    if message_timer > 0:
        text_size = cv2.getTextSize(display_message, FONT, 0.7, 2)[0]
        text_x = (frame_width - text_size[0]) // 2
        cv2.putText(frame, display_message, (text_x, frame_height - 20), FONT, 0.7, MESSAGE_COLOR, 2)
        message_timer -= 1
    else:
        cv2.putText(frame, "[C] to Command | [Q] to Quit", (10, frame_height - 10), FONT, 0.5, MESSAGE_COLOR, 1)

    cv2.imshow('AI Safety Co-Pilot "Saarthi"', frame)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        process_voice_command()

cap.release()
cv2.destroyAllWindows()
