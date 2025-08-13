import cv2
import os
import threading
#import time

# === Camera device paths ===
CAMERA_DEVICES = {
    "Cam0": "/dev/video0",
    "Cam1": "/dev/video2",
    "Cam2": "/dev/video4"
}

# === Output folders ===
OUTPUT_DIR = "captured_images"
for cam_name in CAMERA_DEVICES:
    os.makedirs(os.path.join(OUTPUT_DIR, cam_name), exist_ok=True)

# === Global state ===
frames = {}
selected_camera = list(CAMERA_DEVICES.keys())[0]  # Default selected camera
running = True
img_counters = {name: 0 for name in CAMERA_DEVICES}

#camera control state
cam_caps = {} 
auto_focus = {}
auto_exposure = {}
focus_val = {}
exposure_val = {}

# === Capture Thread Function ===
def camera_thread(cam_name, device_path):
    global frames, running
    cap = cv2.VideoCapture(device_path)
    if not cap.isOpened():
        print(f"❌ Failed to open {device_path} ({cam_name})")
        return

# Set camera properties for better image clarity
    # cap.set(cv2.CAP_PROP_FPS, 60)  # Set FPS to 30 for smoother capture
    # cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus (manual focus needed for sharpness)
    # cap.set(cv2.CAP_PROP_FOCUS, 150)  # Set the focus level (try values between 0-255)
    # cap.set(cv2.CAP_PROP_EXPOSURE, -3)  # Set exposure level (-4 might help reduce blur in bright conditions)
    # cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.75)  # Adjust brightness (range 0 to 1)
    # cap.set(cv2.CAP_PROP_CONTRAST, 0.75)  # Adjust contrast (range 0 to 1)

    while running:
        ret, frame = cap.read()
        if ret:
            frames[cam_name] = frame
    cap.release()
    while running:
        ret, frame = cap.read()
        if ret:
            frames[cam_name] = frame
    cap.release()

# === Start Threads for Each Camera ===
threads = []
for name, path in CAMERA_DEVICES.items():
    t = threading.Thread(target=camera_thread, args=(name, path), daemon=True)
    t.start()
    threads.append(t)

print("✅ Cameras initialized")
print("Press [1/2/3] to select camera")
print("Press [c] to capture image from selected camera")
print("Press [q] to quit")

# === Main Display & Control Loop ===
while True:
    frame = frames.get(selected_camera)
    if frame is not None:
        display = frame.copy()
        cv2.putText(display, f"Selected: {selected_camera}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera Viewer", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('1'):
        selected_camera = "Cam0"
    elif key == ord('2'):
        selected_camera = "Cam1"
    elif key == ord('3'):
        selected_camera = "Cam2"
    elif key == ord(' ') and frame is not None:
        img_idx = img_counters[selected_camera]
        filename = os.path.join(OUTPUT_DIR, selected_camera, f"image_{img_idx:03d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"📸 Saved: {filename}")
        img_counters[selected_camera] += 1

# === Cleanup ===
running = False
cv2.destroyAllWindows()