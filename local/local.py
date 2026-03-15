"""
Local Camera — YOLO Person Detection + Best Frame Selection

Collects frames during a person event, picks the BEST one:
  - Largest person bbox (closest to camera)
  - Highest YOLO confidence
  - Least blur (sharpness score)
Then sends that single best frame to VPS.

Install:
    pip install opencv-python requests imagehash Pillow ultralytics

Run:
    export VPS_URL="http://your-vps-ip:8899"
    python3 local.py
"""

import cv2
import time
import os
import json
import requests
import threading
import numpy as np
import imagehash
from PIL import Image
from datetime import datetime
from queue import Queue
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
RTSP_URL = os.getenv("RTSP_URL", "")
VPS_URL = os.getenv("VPS_URL", "")
CAMERA_NAME = os.getenv("CAMERA_NAME", "Front Door")

OUTPUT_DIR = "motion_clips"
FRAME_DIR = "Frame_Sample"
BEST_FRAME_DIR = "Best_Frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(BEST_FRAME_DIR, exist_ok=True)

# Detection
CHECK_EVERY_SEC = 0.2        # Run YOLO every 0.2 seconds
PERSON_CONF = 0.4            # YOLO confidence for person
RECORD_AFTER = 3             # Stop 3s after last person seen

# Best frame selection
COLLECT_SECONDS = 2.0        # Collect frames for 2 seconds before picking best
MIN_SEND_INTERVAL = 5.0      # Min seconds between sends to VPS

# Sending
TARGET_WIDTH = 1920
JPEG_QUALITY = 85

# YOLO classes
PERSON_CLASS = 0
VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}


# ──────────────────────────────────────────────
# BACKGROUND SENDER
# ──────────────────────────────────────────────

send_queue = Queue()

def sender_worker():
    while True:
        item = send_queue.get()
        if item is None:
            break
        try:
            jpg_bytes, meta = item
            resp = requests.post(
                f"{VPS_URL}/api/analyze",
                files={"image": ("frame.jpg", jpg_bytes, "image/jpeg")},
                data={"metadata": json.dumps(meta, default=str)},
                timeout=60,
            )
            if resp.status_code == 200:
                r = resp.json()
                print(f"  🌐 VPS: {r.get('summary', r.get('status', ''))[:80]}")
            else:
                print(f"  ⚠ VPS error: {resp.status_code}")
        except Exception as e:
            print(f"  ⚠ VPS failed: {e}")
        send_queue.task_done()

threading.Thread(target=sender_worker, daemon=True).start()


# ──────────────────────────────────────────────
# FRAME QUALITY SCORING
# ──────────────────────────────────────────────

def sharpness_score(frame):
    """Laplacian variance — higher = sharper image."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def person_area(bbox):
    """Area of person bounding box — bigger = closer to camera."""
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def score_frame(frame, persons):
    """
    Score a frame based on:
      - Sharpness (40% weight) — less blur
      - Largest person size (40% weight) — closest to camera
      - Highest confidence (20% weight) — YOLO certainty
    Returns total score (higher = better frame)
    """
    if not persons:
        return 0

    # Sharpness
    sharp = sharpness_score(frame)
    sharp_normalized = min(sharp / 500.0, 1.0)  # Normalize to 0-1

    # Largest person
    max_area = max(person_area(p["bbox"]) for p in persons)
    frame_area = frame.shape[0] * frame.shape[1]
    size_normalized = min(max_area / (frame_area * 0.5), 1.0)  # Normalize

    # Highest confidence
    max_conf = max(p["conf"] for p in persons)

    # Weighted score
    score = (sharp_normalized * 0.4) + (size_normalized * 0.4) + (max_conf * 0.2)
    return score


def compress_frame(frame):
    h, w = frame.shape[:2]
    if w > TARGET_WIDTH:
        scale = TARGET_WIDTH / w
        frame = cv2.resize(frame, (TARGET_WIDTH, int(h * scale)))
    _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return jpg.tobytes()


# ──────────────────────────────────────────────
# LOAD YOLO
# ──────────────────────────────────────────────

print("=" * 55)
print("  Local Camera — YOLO + Best Frame Selection")
print("=" * 55)

print("Loading YOLO11n...")
yolo = YOLO("yolo11n.pt")
print("✓ YOLO loaded")

# Test VPS
try:
    r = requests.get(f"{VPS_URL}/health", timeout=5)
    data = r.json()
    print(f"✓ VPS online — {data.get('known_faces', 0)} known, {data.get('unknown_tracked', 0)} tracked")
except:
    print(f"⚠ VPS offline at {VPS_URL}")

# Connect camera
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to connect to camera!")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
skip_frames = max(1, int(fps * CHECK_EVERY_SEC))

print(f"✓ Camera: {width}x{height} @ {fps:.0f}fps")
print(f"✓ YOLO check every {CHECK_EVERY_SEC}s")
print(f"✓ Collects {COLLECT_SECONDS}s of frames, sends best one")
print(f"✓ Sending to {VPS_URL}")
print("=" * 55)
print("Watching...\n")

# State
recording = False
writer = None
last_person_time = 0
frame_count = 0
total_sent = 0
total_events = 0

# Frame collection during event
collecting = False
collect_start = 0
collected_frames = []  # [(frame, persons, vehicles, score)]
last_send_time = 0
event_sent = False     # Has this event already sent a frame?


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────

while True:
    # Flush buffer
    for _ in range(3):
        cap.grab()
    ret, frame = cap.read()

    if not ret:
        print("Lost connection, reconnecting...")
        cap.release()
        time.sleep(2)
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        continue

    frame_count += 1

    # Record if active
    if recording:
        writer.write(frame)

    # Only run YOLO every N frames
    if frame_count % skip_frames != 0:
        # Check recording timeout
        if recording and time.time() - last_person_time > RECORD_AFTER:
            writer.release()
            writer = None
            recording = False
            print(f"⏹  Clip saved. (Events: {total_events} | Sent: {total_sent})\n")
        continue

    # ── Run YOLO ──
    results = yolo(frame, conf=PERSON_CONF, verbose=False)[0]

    persons = []
    vehicles = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == PERSON_CLASS:
            persons.append({"bbox": [x1, y1, x2, y2], "conf": conf})
        elif cls in VEHICLE_CLASSES:
            vehicles.append({"label": VEHICLE_CLASSES[cls], "conf": conf})

    now = time.time()

    # ── Person detected ──
    if persons:
        last_person_time = now

        # Start recording
        if not recording:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = os.path.join(OUTPUT_DIR, f"person_{ts}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(fname, fourcc, fps, (width, height))
            recording = True
            writer.write(frame)
            event_sent = False
            total_events += 1
            print(f"\n🔴 Person detected! Recording → {fname}")

        # Log detections
        for p in persons:
            print(f"  👤 Person ({p['conf']:.0%}) at {p['bbox']}")
        for v in vehicles:
            print(f"  🚗 {v['label']} ({v['conf']:.0%})")

        # ── Frame collection ──
        if not collecting and not event_sent and (now - last_send_time >= MIN_SEND_INTERVAL):
            # Start collecting frames
            collecting = True
            collect_start = now
            collected_frames = []
            print(f"  📷 Collecting frames for {COLLECT_SECONDS}s...")

        if collecting:
            # Score this frame and add to collection
            score = score_frame(frame, persons)
            collected_frames.append({
                "frame": frame.copy(),
                "persons": persons,
                "vehicles": vehicles,
                "score": score,
                "timestamp": datetime.now().isoformat(),
            })

            # Check if collection period is over
            if now - collect_start >= COLLECT_SECONDS:
                collecting = False

                # Pick the BEST frame
                best = max(collected_frames, key=lambda x: x["score"])
                best_frame = best["frame"]
                best_persons = best["persons"]
                best_vehicles = best["vehicles"]
                best_score = best["score"]

                print(f"  🏆 Best frame selected (score: {best_score:.3f}) from {len(collected_frames)} candidates")
                print(f"     Persons: {len(best_persons)} | Sharpness: {sharpness_score(best_frame):.0f}")

                # Save best frame locally
                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                best_path = os.path.join(BEST_FRAME_DIR, f"best_{ts_str}.jpg")
                cv2.imwrite(best_path, best_frame)

                # Also save to Frame_Sample
                cv2.imwrite(os.path.join(FRAME_DIR, f"frame_{ts_str}.jpg"), best_frame)

                # Compress and send to VPS
                jpg = compress_frame(best_frame)
                meta = {
                    "camera": CAMERA_NAME,
                    "timestamp": best["timestamp"],
                    "persons_detected": len(best_persons),
                    "vehicles_detected": len(best_vehicles),
                    "person_bboxes": [p["bbox"] for p in best_persons],
                    "frame_score": round(best_score, 3),
                    "candidates_evaluated": len(collected_frames),
                }
                send_queue.put((jpg, meta))
                total_sent += 1
                last_send_time = now
                event_sent = True
                print(f"  📤 Best frame sent to VPS (#{total_sent}, {len(jpg)//1024}KB)")

                # Clear collection
                collected_frames = []

    # ── No person — check timeouts ──
    else:
        # If we were collecting and person disappeared, send what we have
        if collecting and collected_frames:
            collecting = False
            best = max(collected_frames, key=lambda x: x["score"])
            best_frame = best["frame"]

            print(f"  🏆 Person left — sending best from {len(collected_frames)} frames (score: {best['score']:.3f})")

            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(BEST_FRAME_DIR, f"best_{ts_str}.jpg"), best_frame)
            cv2.imwrite(os.path.join(FRAME_DIR, f"frame_{ts_str}.jpg"), best_frame)

            jpg = compress_frame(best_frame)
            meta = {
                "camera": CAMERA_NAME,
                "timestamp": best["timestamp"],
                "persons_detected": len(best["persons"]),
                "vehicles_detected": len(best["vehicles"]),
                "person_bboxes": [p["bbox"] for p in best["persons"]],
                "frame_score": round(best["score"], 3),
                "candidates_evaluated": len(collected_frames),
            }
            send_queue.put((jpg, meta))
            total_sent += 1
            last_send_time = time.time()
            event_sent = True
            print(f"  📤 Best frame sent to VPS (#{total_sent}, {len(jpg)//1024}KB)")
            collected_frames = []

    # Check recording timeout
    if recording and time.time() - last_person_time > RECORD_AFTER:
        writer.release()
        writer = None
        recording = False
        collecting = False
        collected_frames = []
        print(f"⏹  Clip saved. (Events: {total_events} | Sent: {total_sent})\n")

cap.release()
send_queue.put(None)