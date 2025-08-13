import cv2
import os
import threading
import time
import platform

# ========= Config =========
CAMERA_DEVICES = {
    "Cam0": "/dev/video0",
    "Cam1": "/dev/video2",
    "Cam2": "/dev/video4"
}
OUTPUT_DIR = "captured_images"
RESOLUTION = (1920, 1080)   # (w, h) – tune to your camera
TARGET_FPS = 30             # try 30; increase if supported
CAPTURE_BURST = 5           # take N frames and keep the sharpest
BLUR_THRESHOLD = 120.0      # Laplacian var below this → likely blurry

# ========= Globals =========
frames = {}
locks = {name: threading.Lock() for name in CAMERA_DEVICES}
selected_camera = list(CAMERA_DEVICES.keys())[0]
running = True
img_counters = {name: 0 for name in CAMERA_DEVICES}
caps = {}
af_enabled = {name: False for name in CAMERA_DEVICES}    # start manual
focus_value = {name: 100 for name in CAMERA_DEVICES}     # 0–255 typical
exposure_value = {name: -6 for name in CAMERA_DEVICES}   # driver dependent

for cam_name in CAMERA_DEVICES:
    os.makedirs(os.path.join(OUTPUT_DIR, cam_name), exist_ok=True)

def warmup(cap, frames_to_drop=8):
    for _ in range(frames_to_drop):
        cap.grab()
        time.sleep(0.005)

def lap_var(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def set_camera_params(name, cap):
    # Lower latency
    try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception: pass

    # Resolution / FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    # MJPG often allows higher FPS than YUYV
    try:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    except Exception:
        pass

    # Manual focus (toggle AF later with 'a' if needed)
    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        af_enabled[name] = False
        cap.set(cv2.CAP_PROP_FOCUS, focus_value[name])
    except Exception:
        pass

    # Manual exposure (values vary by driver; start with negative)
    try:
        # Force manual exposure: different backends want different values
        for v in [0.25, 1, 0]:   # V4L2-style, DirectShow-style fallbacks
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, v)
            time.sleep(0.02)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value[name])
    except Exception:
        pass

    warmup(cap, frames_to_drop=10)

def camera_thread(cam_name, device_path):
    global frames, running
    cap = cv2.VideoCapture(device_path)
    if not cap.isOpened():
        print(f"❌ Failed to open {device_path} ({cam_name})")
        return

    set_camera_params(cam_name, cap)
    caps[cam_name] = cap

    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.005)
            continue
        with locks[cam_name]:
            frames[cam_name] = frame

    cap.release()

# === Start threads ===
threads = []
for name, path in CAMERA_DEVICES.items():
    t = threading.Thread(target=camera_thread, args=(name, path), daemon=True)
    t.start()
    threads.append(t)

print("✅ Cameras initialized")
print("[1/2/3] select | [a] AF toggle | [ [ / ] ] focus | [ - / = ] exposure | [c] capture | [q] quit")

# === UI loop ===
while True:
    with locks[selected_camera]:
        frame = frames.get(selected_camera)

    if frame is not None:
        sh = lap_var(frame)
        disp = frame.copy()
        hud = f"{selected_camera} | sharp:{sh:.0f} | AF:{'ON' if af_enabled[selected_camera] else 'OFF'} " \
              f"| F:{focus_value[selected_camera]} | EXP:{exposure_value[selected_camera]}"
        cv2.putText(disp, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Camera Viewer", disp)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('1'):
        selected_camera = "Cam0";  warmup(caps.get(selected_camera), 6)
    elif k == ord('2'):
        selected_camera = "Cam1";  warmup(caps.get(selected_camera), 6)
    elif k == ord('3'):
        selected_camera = "Cam2";  warmup(caps.get(selected_camera), 6)
    elif k == ord('a') and selected_camera in caps:
        # Toggle AF
        af_enabled[selected_camera] = not af_enabled[selected_camera]
        caps[selected_camera].set(cv2.CAP_PROP_AUTOFOCUS, 1 if af_enabled[selected_camera] else 0)
        time.sleep(0.05); warmup(caps[selected_camera], 8)
    elif k in (ord('['), ord(']')) and selected_camera in caps and not af_enabled[selected_camera]:
        # Manual focus
        step = -5 if k == ord('[') else 5
        focus_value[selected_camera] = max(0, min(255, focus_value[selected_camera] + step))
        caps[selected_camera].set(cv2.CAP_PROP_FOCUS, focus_value[selected_camera])
        time.sleep(0.02); warmup(caps[selected_camera], 6)
    elif k in (ord('-'), ord('=')) and selected_camera in caps:
        # Exposure
        step = -1 if k == ord('-') else 1
        exposure_value[selected_camera] += step
        caps[selected_camera].set(cv2.CAP_PROP_EXPOSURE, exposure_value[selected_camera])
        time.sleep(0.02); warmup(caps[selected_camera], 6)
    elif k == ord('c') and frame is not None:
        # Best-of-burst capture to avoid blur
        best = None; best_sh = -1.0
        for _ in range(CAPTURE_BURST):
            warmup(caps[selected_camera], 2)
            with locks[selected_camera]:
                f = frames.get(selected_camera)
            if f is None: continue
            s = lap_var(f)
            if s > best_sh:
                best_sh, best = s, f
        if best is not None:
            idx = img_counters[selected_camera]
            out = os.path.join(OUTPUT_DIR, selected_camera, f"image_{idx:03d}.jpg")
            cv2.imwrite(out, best, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"📸 Saved: {out} | sharpness={best_sh:.1f}" +
                  ("  (⚠ looks blurry)" if best_sh < BLUR_THRESHOLD else ""))
            img_counters[selected_camera] += 1

# === Cleanup ===
running = False
cv2.destroyAllWindows()
