# import cv2
# import mediapipe as mp
# import pickle
# import subprocess
# import numpy as np
# import os
# import platform
# import time
# import grp

# # -------------------- Fix for Qt display issues --------------------
# if platform.system() == "Linux":
#     os.environ["QT_QPA_PLATFORM"] = "xcb"

# # -------------------- Setup --------------------
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
# mp_draw = mp.solutions.drawing_utils

# cap = None

# GESTURE_DATA_FILE = "gesture_data.pkl"
# GESTURE_APP_MAP_FILE = "gesture_app_map.pkl"

# # -------------------- Camera Initialization --------------------
# def initialize_camera():
#     global cap
#     if cap is not None and cap.isOpened():
#         return True

#     print("🎥 Initializing camera...")

#     # Check video group on Linux
#     if platform.system() == "Linux":
#         try:
#             video_gid = grp.getgrnam('video').gr_gid
#             user_groups = os.getgroups()
#             if video_gid not in user_groups:
#                 print("⚠️  Warning: User not in 'video' group")
#                 print("   Run: sudo usermod -a -G video $USER")
#                 print("   Then log out and log back in")
#         except KeyError:
#             print("⚠️  'video' group not found. Skipping group check.")

#     backends = []
#     os_name = platform.system()
#     if os_name == "Windows":
#         backends = [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "Media Foundation"), (cv2.CAP_ANY, "Default")]
#     elif os_name == "Linux":
#         backends = [(cv2.CAP_V4L2, "V4L2"), (cv2.CAP_GSTREAMER, "GStreamer"), (cv2.CAP_ANY, "Default")]
#     elif os_name == "Darwin":
#         backends = [(cv2.CAP_AVFOUNDATION, "AVFoundation"), (cv2.CAP_ANY, "Default")]

#     for backend_id, backend_name in backends:
#         print(f"Trying backend: {backend_name}")
#         for cam_idx in range(3):
#             temp_cap = cv2.VideoCapture(cam_idx, backend_id)
#             if temp_cap.isOpened():
#                 ret, frame = temp_cap.read()
#                 if ret and frame is not None:
#                     cap = temp_cap
#                     print(f"✅ Camera opened successfully! Index: {cam_idx}, Backend: {backend_name}")
#                     return True
#                 temp_cap.release()
#     print("❌ Failed to open camera.")
#     return False

# # -------------------- Gesture Utilities --------------------
# # def save_gesture(name, landmarks):
# #     """Save multiple normalized samples per gesture."""
# #     lm_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
# #     lm_array -= lm_array[0]  # normalize relative to wrist
# #     max_dist = np.max(np.linalg.norm(lm_array, axis=1))
# #     if max_dist > 0:
# #         lm_array /= max_dist
# #     lm_array = lm_array.flatten()

# #     data = load_gestures()
# #     if name not in data:
# #         data[name] = []
# #     data[name].append(lm_array)
# #     pickle.dump(data, open(GESTURE_DATA_FILE, "wb"))
# #     print(f"[Saved] Gesture '{name}' sample {len(data[name])} recorded.")


# def save_gesture(name, landmarks):
#     data = load_gestures()
#     # Convert Mediapipe landmarks → simple (x, y, z) array
#     coords = [(lm.x, lm.y, lm.z) for lm in landmarks]
#     data[name] = coords
#     with open(GESTURE_DATA_FILE, "wb") as f:
#         pickle.dump(data, f)
#     print(f"✅ Gesture '{name}' saved successfully!")



# def load_gestures():
#     if os.path.exists(GESTURE_DATA_FILE):
#         try:
#             return pickle.load(open(GESTURE_DATA_FILE, "rb"))
#         except Exception:
#             print("⚠️ Corrupted gesture file. Recreating...")
#             os.remove(GESTURE_DATA_FILE)
#     return {}

# def save_app_map(app_map):
#     pickle.dump(app_map, open(GESTURE_APP_MAP_FILE, "wb"))

# def load_app_map():
#     if os.path.exists(GESTURE_APP_MAP_FILE):
#         return pickle.load(open(GESTURE_APP_MAP_FILE, "rb"))
#     return {}

# def match_gesture(frame_landmarks, saved_gestures, threshold=0.3):
#     """Compare current frame landmarks with saved gestures."""
#     frame = np.array([[lm.x, lm.y, lm.z] for lm in frame_landmarks])
#     frame -= frame[0]
#     max_dist = np.max(np.linalg.norm(frame, axis=1))
#     if max_dist > 0:
#         frame /= max_dist
#     frame = frame.flatten()

#     best_match = None
#     best_dist = float("inf")

#     for name, samples in saved_gestures.items():
#         for s in samples:
#             dist = np.linalg.norm(frame - s)
#             if dist < best_dist:
#                 best_dist = dist
#                 best_match = name

#     if best_dist < threshold:
#         return best_match
#     return None

# def launch_app(command):
#     """Launch the app command cross-platform."""
#     try:
#         subprocess.Popen(command, shell=True, start_new_session=True)
#         print(f"[Launched] {command}")
#     except Exception as e:
#         print(f"Failed to launch app: {e}")

# # -------------------- Training Mode --------------------
# def train_gestures(samples_per_gesture=15):
#     if not initialize_camera():
#         return

#     app_map = load_app_map()

#     print("=== TRAINING MODE ===")
#     print("Press 't' to record a gesture")
#     print("Press 'q' to quit training")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(frame_rgb)

#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#         cv2.imshow("Gesture Training", frame)
#         key = cv2.waitKey(1) & 0xFF

#         if key == ord('t') and results.multi_hand_landmarks:
#             name = input("Enter gesture name: ").strip()
#             print(f"Recording {samples_per_gesture} samples for '{name}'...")

#             for i in range(samples_per_gesture):
#                 ret, frame = cap.read()
#                 if not ret:
#                     continue
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 res = hands.process(frame_rgb)
#                 if res.multi_hand_landmarks:
#                     save_gesture(name, res.multi_hand_landmarks[0].landmark)

#                 cv2.putText(frame, f"Recording {i+1}/{samples_per_gesture}", (10, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                 cv2.imshow("Gesture Training", frame)
#                 cv2.waitKey(150)

#             command = input(f"Enter command to launch for '{name}': ").strip()
#             app_map[name] = command
#             save_app_map(app_map)
#             print(f"[Mapping Saved] Gesture '{name}' -> {command}")

#         elif key == ord('q'):
#             break

#     if cap:
#         cap.release()
#     cv2.destroyAllWindows()

# # -------------------- Run Mode --------------------
# def run_gestures(gesture_cooldown=2.0, recognition_threshold=0.35, window_size=5):
#     if not initialize_camera():
#         return

#     saved_gestures = load_gestures()
#     app_map = load_app_map()

#     if not saved_gestures or not app_map:
#         print("❌ No gestures found. Train gestures first.")
#         return

#     last_trigger_time = {}
#     gesture_window = []

#     print("=== GESTURE CONTROL RUNNING ===")
#     print("Press 'q' to quit")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(frame_rgb)

#         current_gesture = None
#         if results.multi_hand_landmarks:
#             hand_landmarks = results.multi_hand_landmarks[0]
#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             detected = match_gesture(hand_landmarks.landmark, saved_gestures, threshold=recognition_threshold)
#             if detected:
#                 gesture_window.append(detected)
#                 if len(gesture_window) > window_size:
#                     gesture_window.pop(0)
#                 current_gesture = max(set(gesture_window), key=gesture_window.count)
#             else:
#                 gesture_window.clear()

#         # Trigger app with cooldown
#         if current_gesture and current_gesture in app_map:
#             now = time.time()
#             if now - last_trigger_time.get(current_gesture, 0) > gesture_cooldown:
#                 launch_app(app_map[current_gesture])
#                 last_trigger_time[current_gesture] = now

#         # Display
#         text = f"Gesture: {current_gesture}" if current_gesture else "No gesture detected"
#         cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#         cv2.imshow("Gesture Control", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     if cap:
#         cap.release()
#     cv2.destroyAllWindows()

# # -------------------- Test Mode --------------------
# def test_gestures():
#     saved_gestures = load_gestures()
#     app_map = load_app_map()
#     print("=== TEST MODE ===")
#     print(f"Loaded {len(saved_gestures)} gestures:")
#     for name in saved_gestures:
#         print(f" - {name}")
#     print(f"Loaded {len(app_map)} app mappings:")
#     for gesture, command in app_map.items():
#         print(f" - {gesture} -> {command}")

# # -------------------- Camera Test --------------------
# def test_camera_only():
#     if not initialize_camera():
#         return
#     print("=== CAMERA TEST === (Press 'q' to exit)")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             continue
#         cv2.imshow("Camera Test", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     if cap:
#         cap.release()
#     cv2.destroyAllWindows()

# # -------------------- Main --------------------
# if __name__ == "__main__":
#     mode = input("Enter mode (train/run/test/camera): ").strip().lower()
#     if mode == "train":
#         train_gestures()
#     elif mode == "run":
#         run_gestures()
#     elif mode == "test":
#         test_gestures()
#     elif mode == "camera":
#         test_camera_only()
#     else:
#         print("Invalid mode. Choose 'train', 'run', 'test', or 'camera'.")

#     if cap is not None:
#         cap.release()
#     cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
import platform
import json

# === GLOBAL VARIABLES ===
GESTURE_DATA_FILE = "gesture_data.pkl"
APP_MAP_FILE = "app_map.json"
cap = None

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# =====================================
# CAMERA INITIALIZATION
# =====================================
def initialize_camera():
    global cap
    if cap is not None and cap.isOpened():
        return True

    print("🎥 Initializing camera...")

    os_name = platform.system()
    backends = []

    if os_name == "Windows":
        backends = [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "Media Foundation"), (cv2.CAP_ANY, "Default")]
    elif os_name == "Linux":
        backends = [(cv2.CAP_V4L2, "V4L2"), (cv2.CAP_GSTREAMER, "GStreamer"), (cv2.CAP_ANY, "Default")]
    else:
        backends = [(cv2.CAP_AVFOUNDATION, "AVFoundation"), (cv2.CAP_ANY, "Default")]

    for backend_id, backend_name in backends:
        for cam_idx in range(0, 3):
            temp_cap = cv2.VideoCapture(cam_idx, backend_id)
            if temp_cap.isOpened():
                ret, frame = temp_cap.read()
                if ret:
                    cap = temp_cap
                    print(f"✅ Camera opened (index {cam_idx}, backend {backend_name})")
                    return True
    print("❌ Camera initialization failed.")
    return False

# =====================================
# DATA UTILS
# =====================================
def load_gestures():
    if os.path.exists(GESTURE_DATA_FILE):
        with open(GESTURE_DATA_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def load_app_map():
    if os.path.exists(APP_MAP_FILE):
        with open(APP_MAP_FILE, "r") as f:
            return json.load(f)
    return {}  # Start empty for fully dynamic commands

def save_gesture(name, landmarks):
    data = load_gestures()
    coords = [(lm.x, lm.y, lm.z) for lm in landmarks]
    data.setdefault(name, []).append(coords)
    with open(GESTURE_DATA_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"✅ Saved sample for gesture: {name} ({len(data[name])} samples)")

    # Ask for dynamic command
    app_map = load_app_map()
    if name not in app_map:
        cmd = input(f"💻 Enter system command for gesture '{name}': ").strip()

        # Auto-fix for Chrome to suppress GCM warnings
        if "chrome" in cmd.lower() or "google-chrome" in cmd.lower():
            if "--disable-features=GCM" not in cmd:
                cmd += " --disable-features=GCM"

        app_map[name] = cmd
        with open(APP_MAP_FILE, "w") as f:
            json.dump(app_map, f, indent=2)
        print(f"✅ Saved command for '{name}': {cmd}")

def normalize_landmarks(landmarks):
    arr = np.array([(lm.x, lm.y, lm.z) for lm in landmarks])
    base = arr[0]
    arr -= base
    norm = np.linalg.norm(arr)
    if norm != 0:
        arr /= norm
    return arr.flatten()

# =====================================
# GESTURE MATCHING
# =====================================
def match_gesture(current_landmarks, saved_gestures):
    current = normalize_landmarks(current_landmarks)
    best_match, best_score = None, float("inf")

    for name, samples in saved_gestures.items():
        for sample in samples:
            saved = np.array(sample)
            base = saved[0]
            saved -= base
            norm = np.linalg.norm(saved)
            if norm != 0:
                saved /= norm
            saved = saved.flatten()

            if len(saved) == len(current):
                score = np.linalg.norm(saved - current)
                if score < best_score:
                    best_match, best_score = name, score

    return best_match if best_score < 0.25 else None

# =====================================
# TRAIN MODE
# =====================================
def train_gestures():
    if not initialize_camera():
        return

    gesture_name = input("👉 Enter gesture name to train: ").strip()
    print(f"🖐️ Show gesture '{gesture_name}' to camera...")
    print("Press SPACE to capture sample, ESC to finish training.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Training: {gesture_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Gesture Training", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    save_gesture(gesture_name, hand_landmarks.landmark)
            else:
                print("❌ No hand detected! Try again.")

    cv2.destroyAllWindows()
    print(f"✅ Training complete for gesture: {gesture_name}")

# =====================================
# RUN MODE
# =====================================
def launch_app(cmd):
    print(f"🚀 Launching: {cmd}")
    os.system(cmd)

def run_gestures():
    if not initialize_camera():
        return

    saved_gestures = load_gestures()
    app_map = load_app_map()

    if not saved_gestures:
        print("❌ No gestures found. Train gestures first.")
        return

    print("=== 🖐️ GESTURE CONTROL RUNNING ===")
    print("Press 'q' to quit.")

    last_gesture = None
    last_time = 0
    cooldown = 3.0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        gesture_name = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture_name = match_gesture(hand_landmarks.landmark, saved_gestures)

        if gesture_name:
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if gesture_name != last_gesture or (time.time() - last_time) > cooldown:
                if gesture_name in app_map:
                    launch_app(app_map[gesture_name])
                else:
                    print(f"⚠️ No command mapped for gesture '{gesture_name}'. Train it first.")
                last_gesture = gesture_name
                last_time = time.time()
        else:
            cv2.putText(frame, "No gesture detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Gesture Control", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

# =====================================
# MAIN ENTRY
# =====================================
if __name__ == "__main__":
    mode = input("Enter mode (train/run): ").strip().lower()
    if mode == "train":
        train_gestures()
    elif mode == "run":
        run_gestures()
    else:
        print("Invalid mode. Use 'train' or 'run'.")
