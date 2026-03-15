"""
Camera Security Server — Async ML + Hermes Architecture

Pipeline
1. Receives images over FastAPI
2. Runs local ML with hot models in RAM (YOLO11, conditional YOLO11-Pose, InsightFace, optional weapon YOLO)
3. Routes known clean identities to direct Python Notion logging (no Hermes call)
4. Queues unknown/suspicious scenes for background Hermes vision processing
5. Archives known sightings under known_object/<person>/{normal,threat}
"""

import asyncio
import json
import logging
import os
import pickle
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from insightface.app import FaceAnalysis
from ultralytics import YOLO

load_dotenv()

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
KNOWN_OBJECT_DIR = BASE_DIR / "known_object"
FACE_LOG_DIR = BASE_DIR / "face_logs"
UPLOAD_DIR = BASE_DIR / "camera_uploads"
LOG_DIR = BASE_DIR / "analysis_logs"
UNKNOWN_DB_FILE = BASE_DIR / "unknown_faces.pkl"
ENCODINGS_CACHE = BASE_DIR / "known_encodings.pkl"
SERVER_LOG = BASE_DIR / "server.log"
WEAPON_MODEL_PATH = BASE_DIR / "weapon_yolo.pt"

FACE_SIMILARITY = float(os.getenv("FACE_SIMILARITY", "0.4"))
UNKNOWN_MATCH = float(os.getenv("UNKNOWN_MATCH", "0.4"))
AUTO_ADD_AFTER = int(os.getenv("AUTO_ADD_AFTER", "5"))
PERSON_CONF = float(os.getenv("PERSON_CONF", "0.5"))
WEAPON_CONF = float(os.getenv("WEAPON_CONF", "0.3"))
API_PORT = int(os.getenv("PORT", "8899"))
VPS_HOST = os.getenv("VPS_HOST", f"http://localhost:{API_PORT}")
NOTION_DB_ID = os.getenv("SECURITY_NOTION_DB_ID", "")
NOTION_API_KEY = os.getenv("NOTION_API_KEY", "")
NOTION_VERSION = os.getenv("NOTION_VERSION", "2022-06-28")
HERMES_PROVIDER = os.getenv("HERMES_PROVIDER", "nous")
HERMES_VISION_MODEL = os.getenv("HERMES_VISION_MODEL", "gemini-3-flash")
HERMES_TIMEOUT = int(os.getenv("HERMES_TIMEOUT", "300"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "60"))
JOB_QUEUE_MAXSIZE = int(os.getenv("JOB_QUEUE_MAXSIZE", "512"))
PIPELINE_NAME = os.getenv("PIPELINE_NAME", "camera-security-vps")
SOURCE_PATH = "/api/analyze"
DIRECT_KNOWN_ROUTE = "known_identity"
HERMES_ROUTE = "vision_threat"
THREAT_LEVELS = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]
THREAT_BUCKET_LEVELS = {"MEDIUM", "HIGH", "CRITICAL", "UNKNOWN", "TIMEOUT", "ERROR"}
PERSON_CLASS = 0
VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}

# Pose keypoints
NOSE = 0
L_EYE = 1
R_EYE = 2
L_SHOULDER = 5
R_SHOULDER = 6
L_WRIST = 9
R_WRIST = 10
L_HIP = 11
R_HIP = 12
L_ANKLE = 15
R_ANKLE = 16

DB_LOCK = threading.Lock()
LOG_LOCK = threading.Lock()
COOLDOWN_LOCK = threading.Lock()
JOB_STATUS_LOCK = threading.Lock()
WORKER_LOCK = threading.Lock()
HERMES_QUEUE: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=JOB_QUEUE_MAXSIZE)
JOB_STATUS: Dict[str, Dict[str, Any]] = {}
RECENT_PROCESSING: Dict[str, float] = {}
WORKER_THREAD: Optional[threading.Thread] = None

for d in [KNOWN_FACES_DIR, KNOWN_OBJECT_DIR, FACE_LOG_DIR, UPLOAD_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(SERVER_LOG)],
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Camera Security Server")
app.mount("/images", StaticFiles(directory=str(UPLOAD_DIR)), name="images")

# ──────────────────────────────────────────────
# LOAD MODELS
# ──────────────────────────────────────────────
print("=" * 60)
print("  Camera Security Server — Loading models...")
print("=" * 60)
print("[1/3] YOLO11...")
yolo = YOLO(str(BASE_DIR / "yolo11n.pt"))
print("  ✓ YOLO11")
print("[2/3] YOLO11-Pose...")
yolo_pose = YOLO(str(BASE_DIR / "yolo11n-pose.pt"))
print("  ✓ YOLO11-Pose")
print("[3/3] InsightFace...")
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(640, 640))
print("  ✓ InsightFace")
weapon_det = None
if WEAPON_MODEL_PATH.exists():
    weapon_det = YOLO(str(WEAPON_MODEL_PATH))
    print("  ✓ Weapon detector")
else:
    print("  ⚠ No weapon model")

# ──────────────────────────────────────────────
# FACE DATABASE
# ──────────────────────────────────────────────
def load_known() -> Dict[str, List[np.ndarray]]:
    if ENCODINGS_CACHE.exists():
        cache_time = ENCODINGS_CACHE.stat().st_mtime
        image_times = [
            (Path(root) / file_name).stat().st_mtime
            for root, _, files in os.walk(KNOWN_FACES_DIR)
            for file_name in files
            if file_name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if image_times and cache_time > max(image_times):
            with ENCODINGS_CACHE.open("rb") as f:
                return pickle.load(f)

    db: Dict[str, List[np.ndarray]] = {}
    for name in sorted(os.listdir(KNOWN_FACES_DIR)):
        person_dir = KNOWN_FACES_DIR / name
        if not person_dir.is_dir():
            continue
        embeddings: List[np.ndarray] = []
        for img_name in sorted(os.listdir(person_dir)):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(str(person_dir / img_name))
            if img is None:
                continue
            faces = face_app.get(img)
            if faces:
                largest = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                embeddings.append(largest.embedding)
                print(f"    ✓ {name}/{img_name}")
        if embeddings:
            db[name] = embeddings

    with ENCODINGS_CACHE.open("wb") as f:
        pickle.dump(db, f)
    return db


def rebuild_known() -> Dict[str, List[np.ndarray]]:
    if ENCODINGS_CACHE.exists():
        ENCODINGS_CACHE.unlink()
    return load_known()


def load_unknown() -> Dict[str, Dict[str, Any]]:
    if UNKNOWN_DB_FILE.exists():
        with UNKNOWN_DB_FILE.open("rb") as f:
            return pickle.load(f)
    return {}


def save_unknown(db: Dict[str, Dict[str, Any]]) -> None:
    with UNKNOWN_DB_FILE.open("wb") as f:
        pickle.dump(db, f)


def cosim(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


def match_known_face(emb: np.ndarray, db: Dict[str, List[np.ndarray]]) -> Tuple[Optional[str], float]:
    best_name, best_score = None, 0.0
    for name, embeddings in db.items():
        for known_emb in embeddings:
            score = cosim(emb, known_emb)
            if score > best_score:
                best_score, best_name = score, name
    return (best_name, best_score) if best_score >= FACE_SIMILARITY else (None, best_score)


def match_unknown_face(emb: np.ndarray, db: Dict[str, Dict[str, Any]]) -> Tuple[Optional[str], float]:
    best_uid, best_score = None, 0.0
    for uid, data in db.items():
        score = cosim(emb, data["embedding"])
        if score > best_score:
            best_score, best_uid = score, uid
    return (best_uid, best_score) if best_score >= UNKNOWN_MATCH else (None, best_score)


def next_person_id(base_dir: Path) -> str:
    existing = [
        d.name
        for d in base_dir.iterdir()
        if d.is_dir() and d.name.startswith("person_") and d.name.split("_")[-1].isdigit()
    ]
    if not existing:
        return "person_001"
    nums = [int(name.split("_")[1]) for name in existing]
    return f"person_{max(nums) + 1:03d}"


print("\nLoading face databases...")
known_db = load_known()
unknown_db = load_unknown()
print(f"  Known: {len(known_db)} people | Unknown tracked: {len(unknown_db)}")

# ──────────────────────────────────────────────
# POSE ANALYSIS
# ──────────────────────────────────────────────
def analyze_pose(kps: np.ndarray, pbbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
    result = {
        "crouching": False,
        "running": False,
        "arms_raised": False,
        "hiding_face": False,
        "bent_over": False,
        "behavior_score": 0,
        "behavior_label": "standing",
    }
    if kps is None or len(kps) < 17:
        return result

    kp = kps[:, :2]
    conf = kps[:, 2] if kps.shape[1] > 2 else np.ones(17)
    px1, py1, px2, py2 = pbbox
    ph = py2 - py1
    pw = px2 - px1
    if ph < 20:
        return result

    if conf[L_SHOULDER] > 0.3 and conf[L_ANKLE] > 0.3:
        torso = abs(kp[L_ANKLE][1] - kp[L_SHOULDER][1])
        if torso < ph * 0.35:
            result["crouching"] = True
            result["behavior_score"] += 3

    for sh, wr in [(L_SHOULDER, L_WRIST), (R_SHOULDER, R_WRIST)]:
        if conf[sh] > 0.3 and conf[wr] > 0.3 and kp[wr][1] < kp[sh][1] - 20:
            result["arms_raised"] = True
            result["behavior_score"] += 1

    if conf[L_ANKLE] > 0.3 and conf[R_ANKLE] > 0.3:
        if abs(kp[L_ANKLE][0] - kp[R_ANKLE][0]) > pw * 0.6:
            result["running"] = True
            result["behavior_score"] += 2

    if conf[NOSE] > 0.3 and conf[L_HIP] > 0.3:
        hip_y = (kp[L_HIP][1] + kp[R_HIP][1]) / 2 if conf[R_HIP] > 0.3 else kp[L_HIP][1]
        if kp[NOSE][1] - hip_y > -ph * 0.1:
            result["bent_over"] = True
            result["behavior_score"] += 2

    face_visible = conf[NOSE] > 0.3 or conf[L_EYE] > 0.3 or conf[R_EYE] > 0.3
    body_visible = conf[L_SHOULDER] > 0.3 or conf[R_SHOULDER] > 0.3
    if not face_visible and body_visible:
        result["hiding_face"] = True
        result["behavior_score"] += 4

    for label in ["crouching", "running", "bent_over", "arms_raised", "hiding_face"]:
        if result[label]:
            result["behavior_label"] = label.replace("_", " ")
            break
    return result


# ──────────────────────────────────────────────
# AGE CORRECTION
# ──────────────────────────────────────────────
def correct_age(raw: Optional[float], fbbox: List[int], pbbox: Tuple[int, int, int, int]) -> Tuple[Optional[int], str]:
    if raw is None:
        return None, "unknown"
    person_h = pbbox[3] - pbbox[1]
    face_h = fbbox[3] - fbbox[1]
    if person_h < 10 or face_h < 5:
        return int(raw), "uncertain"

    ratio = face_h / person_h
    age = int(raw)
    if ratio > 0.28:
        if raw > 15:
            age = max(6, int(raw * 0.35))
        return min(age, 12), "child"
    if ratio > 0.22:
        if raw > 20:
            age = max(10, int(raw * 0.55))
        return min(age, 17), "teen_or_child"
    return age, "adult"


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def unique_list(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def chunk_text(text: str, chunk_size: int = 1800) -> List[Dict[str, Any]]:
    if not text:
        text = ""
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]
    return [{"type": "text", "text": {"content": chunk}} for chunk in chunks[:50]]


def build_source_tag_text(agent_route: str, image_category: str, job_id: Optional[str] = None) -> str:
    lines = [
        f"source_path={SOURCE_PATH}",
        f"agent_route={agent_route}",
        f"pipeline={PIPELINE_NAME}",
        f"image_category={image_category}",
    ]
    if job_id:
        lines.append(f"job_id={job_id}")
    return "\n".join(lines)


def image_category_from_result(route: str, hermes_result: Optional[Dict[str, Any]]) -> str:
    if route != HERMES_ROUTE:
        return "normal"
    if not hermes_result:
        return "threat"
    threat_level = str(hermes_result.get("threat_level", "UNKNOWN")).upper()
    score = hermes_result.get("score")
    if threat_level in THREAT_BUCKET_LEVELS:
        return "threat"
    if isinstance(score, (int, float)) and score >= 4:
        return "threat"
    return "normal"


def archive_known_sighting(person_name: str, source_image_path: str, category: str, ts_str: str) -> Optional[str]:
    category = "threat" if category == "threat" else "normal"
    person_dir = KNOWN_OBJECT_DIR / person_name / category
    person_dir.mkdir(parents=True, exist_ok=True)
    destination = person_dir / f"{ts_str}.jpg"
    try:
        shutil.copy2(source_image_path, destination)
        return str(destination)
    except Exception as exc:
        logger.error(f"Failed to archive image for {person_name}: {exc}")
        return None


def append_log_entry(entry: Dict[str, Any]) -> None:
    log_file = LOG_DIR / "log.jsonl"
    with LOG_LOCK:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")


def build_persons_summary(ml_result: Dict[str, Any]) -> List[str]:
    people = []
    for index, person in enumerate(ml_result.get("persons", []), start=1):
        face_info = person.get("face_info") or {}
        pose = person.get("pose") or {}
        summary = f"Person {index}: {person.get('face_status', 'unknown')}"
        if person.get("name"):
            summary += f", {person['name']}"
        if person.get("on_cooldown"):
            summary += ", cooldown=1"
        if face_info:
            summary += f", {face_info.get('gender', '?')} ~{face_info.get('age', '?')}y"
        summary += f", behavior: {pose.get('behavior_label', 'standing')}"
        if person.get("weapons_near"):
            summary += f", weapons near: {', '.join(person['weapons_near'])}"
        if person.get("sighting_count"):
            summary += f", sightings: {person['sighting_count']}"
        people.append(summary)
    return people


def parse_json_result(response_text: str) -> Optional[Dict[str, Any]]:
    for line in reversed(response_text.splitlines()):
        if "JSON_RESULT:" not in line:
            continue
        _, payload = line.split("JSON_RESULT:", 1)
        payload = payload.strip().strip("`")
        try:
            data = json.loads(payload)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    match = re.search(r"JSON_RESULT:\s*(\{.*\})", response_text, re.DOTALL)
    if not match:
        return None
    payload = match.group(1).strip().strip("`")
    try:
        data = json.loads(payload)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def infer_threat_level_from_text(text: str, default: str = "UNKNOWN") -> str:
    upper = text.upper()
    for level in THREAT_LEVELS:
        if level in upper:
            return level
    return default


def prune_cooldowns() -> None:
    now = time.monotonic()
    expired = [key for key, ts in RECENT_PROCESSING.items() if now - ts > COOLDOWN_SECONDS]
    for key in expired:
        RECENT_PROCESSING.pop(key, None)


def identity_on_cooldown(identity_key: Optional[str]) -> bool:
    if not identity_key:
        return False
    with COOLDOWN_LOCK:
        prune_cooldowns()
        ts = RECENT_PROCESSING.get(identity_key)
        if ts is None:
            return False
        return (time.monotonic() - ts) < COOLDOWN_SECONDS


def record_identity_processing(identity_key: Optional[str]) -> None:
    if not identity_key:
        return
    with COOLDOWN_LOCK:
        prune_cooldowns()
        RECENT_PROCESSING[identity_key] = time.monotonic()


def notion_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def create_notion_page(title: str, threat_level: str, score: int, description: str, camera: str, image_url: str, timestamp: str) -> Dict[str, Any]:
    if not NOTION_API_KEY:
        return {"ok": False, "error": "NOTION_API_KEY missing"}

    payload = {
        "parent": {"database_id": NOTION_DB_ID},
        "properties": {
            "Alert": {"title": chunk_text(title, 200)},
            "Threat Level": {"select": {"name": threat_level}},
            "Score": {"number": score},
            "Description": {"rich_text": chunk_text(description)},
            "Camera": {"rich_text": chunk_text(camera, 200)},
            "Image": {"url": image_url},
            "Time": {"date": {"start": timestamp}},
        },
    }
    try:
        response = requests.post(
            "https://api.notion.com/v1/pages",
            headers=notion_headers(),
            json=payload,
            timeout=15,
        )
        if response.ok:
            data = response.json()
            return {"ok": True, "page_id": data.get("id"), "status_code": response.status_code}
        return {"ok": False, "status_code": response.status_code, "error": response.text[:1000]}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def direct_log_known_to_notion(ml_result: Dict[str, Any], image_url: str, camera: str) -> Dict[str, Any]:
    identities = ", ".join(ml_result.get("direct_known_identities", [])) or ", ".join(ml_result.get("known_identities", [])) or "unknown"
    description_parts = [
        build_source_tag_text(DIRECT_KNOWN_ROUTE, "normal"),
        f"route_reason={ml_result.get('route_reason', '')}",
        f"known_identities={', '.join(ml_result.get('known_identities', [])) or 'None'}",
        f"auto_added_identities={', '.join(ml_result.get('auto_added_identities', [])) or 'None'}",
        f"vehicles={', '.join(v['label'] for v in ml_result.get('vehicles', [])) or 'None'}",
        "people_summary:",
        *build_persons_summary(ml_result),
    ]
    description = "\n".join(description_parts)
    title = f"Known: {identities} — {camera}"
    notion_result = create_notion_page(
        title=title,
        threat_level="NONE",
        score=0,
        description=description,
        camera=camera,
        image_url=image_url,
        timestamp=ml_result.get("timestamp") or datetime.now().isoformat(),
    )
    return {
        "agent": DIRECT_KNOWN_ROUTE,
        "status": "ok" if notion_result.get("ok") else "error",
        "notion_logged": notion_result.get("ok", False),
        "telegram_sent": False,
        "threat_level": "NONE",
        "score": 0,
        "summary": "known person logged directly from Python",
        "category": "normal",
        "notion_result": notion_result,
    }


def update_job_status(job_id: str, **fields: Any) -> None:
    with JOB_STATUS_LOCK:
        current = JOB_STATUS.setdefault(job_id, {"job_id": job_id})
        current.update(fields)


def start_background_worker() -> None:
    global WORKER_THREAD
    with WORKER_LOCK:
        if WORKER_THREAD and WORKER_THREAD.is_alive():
            return
        WORKER_THREAD = threading.Thread(target=background_worker_loop, name="hermes-worker", daemon=True)
        WORKER_THREAD.start()
        logger.info("✓ Hermes background worker started")


def queue_hermes_job(ml_result: Dict[str, Any], image_url: str, camera: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "queued_at": datetime.now().isoformat(),
        "camera": camera,
        "image_url": image_url,
        "ml_result": ml_result,
        "source_path": SOURCE_PATH,
    }
    update_job_status(
        job_id,
        status="queued",
        queued_at=job["queued_at"],
        route=ml_result.get("route"),
        camera=camera,
        image_url=image_url,
    )
    HERMES_QUEUE.put_nowait(job)
    return job_id


def background_worker_loop() -> None:
    while True:
        job = HERMES_QUEUE.get()
        job_id = job["job_id"]
        ml_result = job["ml_result"]
        image_url = job["image_url"]
        camera = job["camera"]
        update_job_status(job_id, status="processing", started_at=datetime.now().isoformat())
        try:
            hermes_result = asyncio.run(run_hermes_vision_threat(ml_result, image_url, camera, job_id=job_id))
            image_category = image_category_from_result(ml_result.get("route", "none"), hermes_result)
            archive_paths: Dict[str, str] = {}
            ts_component = Path(ml_result.get("image_path", "image.jpg")).stem.replace("det_", "") or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            for name in ml_result.get("archive_identities", []):
                archived = archive_known_sighting(name, ml_result["image_path"], image_category, ts_component)
                if archived:
                    archive_paths[name] = archived

            completion = {
                "phase": "async_job_completed",
                "job_id": job_id,
                "timestamp": datetime.now().isoformat(),
                "camera": camera,
                "image_url": image_url,
                "image_path": ml_result.get("image_path"),
                "route": ml_result.get("route"),
                "route_reason": ml_result.get("route_reason"),
                "image_category": image_category,
                "known_identities": ml_result.get("known_identities", []),
                "auto_added_identities": ml_result.get("auto_added_identities", []),
                "archive_paths": archive_paths,
                "ml_result": ml_result,
                "hermes_result": hermes_result,
            }
            append_log_entry(completion)
            update_job_status(
                job_id,
                status="completed",
                finished_at=datetime.now().isoformat(),
                image_category=image_category,
                archive_paths=archive_paths,
                hermes_result=hermes_result,
            )
        except Exception as exc:
            logger.exception(f"Background Hermes job failed: {exc}")
            update_job_status(
                job_id,
                status="error",
                finished_at=datetime.now().isoformat(),
                error=str(exc),
            )
            append_log_entry(
                {
                    "phase": "async_job_failed",
                    "job_id": job_id,
                    "timestamp": datetime.now().isoformat(),
                    "camera": camera,
                    "image_url": image_url,
                    "route": ml_result.get("route"),
                    "error": str(exc),
                }
            )
        finally:
            HERMES_QUEUE.task_done()


# ──────────────────────────────────────────────
# FULL ML PIPELINE
# ──────────────────────────────────────────────
def run_ml(image_path: str) -> Dict[str, Any]:
    global known_db, unknown_db

    frame = cv2.imread(image_path)
    if frame is None:
        return {"error": "Could not read image", "image_path": image_path}

    h, w = frame.shape[:2]
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "image_path": os.path.abspath(image_path),
        "image_size": f"{w}x{h}",
        "persons": [],
        "vehicles": [],
        "weapons": [],
        "needs_hermes": False,
        "route": "none",
        "route_reason": "No persons detected",
        "known_identities": [],
        "direct_known_identities": [],
        "auto_added_identities": [],
        "archive_identities": [],
        "cooldown_record_keys": [],
        "cooldown_skipped": [],
        "scene_flags": {
            "has_known_clean": False,
            "has_unknown": False,
            "has_suspicious": False,
            "has_no_face_suspicious": False,
            "weapon_detected": False,
        },
    }
    route_reasons: List[str] = []

    yolo_res = yolo(frame, conf=PERSON_CONF, verbose=False)[0]
    for box in yolo_res.boxes:
        cls = int(box.cls[0])
        conf = round(float(box.conf[0]), 3)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if cls == PERSON_CLASS:
            result["persons"].append({"bbox": [x1, y1, x2, y2], "confidence": conf})
        elif cls in VEHICLE_CLASSES:
            result["vehicles"].append(
                {"label": VEHICLE_CLASSES[cls], "bbox": [x1, y1, x2, y2], "confidence": conf}
            )

    if weapon_det:
        wpn_res = weapon_det(frame, conf=WEAPON_CONF, verbose=False)[0]
        for box in wpn_res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            result["weapons"].append(
                {
                    "label": wpn_res.names[int(box.cls[0])],
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(float(box.conf[0]), 3),
                }
            )

    if result["weapons"]:
        result["scene_flags"]["weapon_detected"] = True
        result["scene_flags"]["has_suspicious"] = True
        route_reasons.append("weapon detected")

    if not result["persons"]:
        if result["weapons"]:
            result["route"] = HERMES_ROUTE
            result["needs_hermes"] = True
            result["route_reason"] = "weapon detected"
        return result

    # Pass 1: weapons-near association + identity recognition + cooldown assessment
    for person_index, person in enumerate(result["persons"], start=1):
        px1, py1, px2, py2 = person["bbox"]
        person["pose"] = {
            "crouching": False,
            "running": False,
            "arms_raised": False,
            "hiding_face": False,
            "bent_over": False,
            "behavior_score": 0,
            "behavior_label": "standing",
        }
        person["weapons_near"] = []
        person["face_status"] = "no_face"
        person["name"] = None
        person["face_info"] = None
        person["identity_key"] = None
        person["on_cooldown"] = False
        person["needs_pose"] = False

        for weapon in result["weapons"]:
            wx = (weapon["bbox"][0] + weapon["bbox"][2]) / 2
            wy = (weapon["bbox"][1] + weapon["bbox"][3]) / 2
            if (px1 - 100 <= wx <= px2 + 100) and (py1 - 100 <= wy <= py2 + 100):
                person["weapons_near"].append(weapon["label"])

        pad_x = int((px2 - px1) * 0.1)
        pad_y = int((py2 - py1) * 0.05)
        cy1, cy2 = max(0, py1 - pad_y), min(h, py2 + pad_y)
        cx1, cx2 = max(0, px1 - pad_x), min(w, px2 + pad_x)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            person["needs_pose"] = True
            continue

        faces = face_app.get(crop)
        if not faces:
            person["needs_pose"] = True
            continue

        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        emb = face.embedding
        fx1, fy1, fx2, fy2 = face.bbox.astype(int)
        fabs = [cx1 + fx1, cy1 + fy1, cx1 + fx2, cy1 + fy2]
        raw_age = getattr(face, "age", None)
        raw_gender = getattr(face, "gender", None)
        gender = "Male" if raw_gender == 1 else ("Female" if raw_gender == 0 else "Unknown")
        age, size_cat = correct_age(raw_age, fabs, (px1, py1, px2, py2))
        person["face_info"] = {
            "bbox": fabs,
            "age": age,
            "raw_age": int(raw_age) if raw_age is not None else None,
            "gender": gender,
            "size_category": size_cat,
            "det_score": round(float(face.det_score), 3),
        }

        with DB_LOCK:
            known_match, known_score = match_known_face(emb, known_db)
            if known_match:
                person["face_status"] = "known"
                person["name"] = known_match
                person["similarity"] = round(known_score, 3)
                person["identity_key"] = f"known:{known_match}"
                person["on_cooldown"] = identity_on_cooldown(person["identity_key"])
                result["known_identities"].append(known_match)
                result["archive_identities"].append(known_match)
                if person["weapons_near"]:
                    result["scene_flags"]["has_suspicious"] = True
                    route_reasons.append(f"known person {known_match} near weapon")
                    result["cooldown_record_keys"].append(person["identity_key"])
                elif person["on_cooldown"]:
                    result["cooldown_skipped"].append(person["identity_key"])
                    route_reasons.append(f"known person {known_match} on cooldown")
                else:
                    result["scene_flags"]["has_known_clean"] = True
                    result["direct_known_identities"].append(known_match)
                    result["cooldown_record_keys"].append(person["identity_key"])
                # Known clean skips pose for speed; known+weapon already suspicious without pose.
            else:
                crop_path = FACE_LOG_DIR / f"unk_{ts_str}_{person_index}.jpg"
                face_crop = frame[max(0, fabs[1]):min(h, fabs[3]), max(0, fabs[0]):min(w, fabs[2])]
                if face_crop.size > 0:
                    cv2.imwrite(str(crop_path), face_crop)

                unknown_id, unknown_score = match_unknown_face(emb, unknown_db)
                if unknown_id:
                    cooldown_key = f"unknown:{unknown_id}"
                    on_cooldown = identity_on_cooldown(cooldown_key)
                    person["identity_key"] = cooldown_key
                    person["on_cooldown"] = on_cooldown
                    person["unknown_similarity"] = round(unknown_score, 3)
                    if on_cooldown:
                        person["face_status"] = "repeat_unknown"
                        person["name"] = "Unknown"
                        result["cooldown_skipped"].append(cooldown_key)
                        route_reasons.append("repeat unknown on cooldown")
                        person["needs_pose"] = False
                    else:
                        unknown_db[unknown_id]["count"] += 1
                        unknown_db[unknown_id]["last_seen"] = datetime.now().isoformat()
                        unknown_db[unknown_id]["images"].append(str(crop_path))
                        unknown_db[unknown_id]["embedding"] = (unknown_db[unknown_id]["embedding"] + emb) / 2
                        count = unknown_db[unknown_id]["count"]
                        if count >= AUTO_ADD_AFTER:
                            new_name = next_person_id(KNOWN_FACES_DIR)
                            new_person_dir = KNOWN_FACES_DIR / new_name
                            new_person_dir.mkdir(parents=True, exist_ok=True)
                            for idx, img_path in enumerate(unknown_db[unknown_id]["images"], start=1):
                                if os.path.exists(img_path):
                                    img2 = cv2.imread(img_path)
                                    if img2 is not None:
                                        cv2.imwrite(str(new_person_dir / f"auto_{idx}.jpg"), img2)
                            del unknown_db[unknown_id]
                            save_unknown(unknown_db)
                            known_db = rebuild_known()
                            person["face_status"] = "auto_added"
                            person["name"] = new_name
                            person["sighting_count"] = count
                            person["identity_key"] = f"known:{new_name}"
                            result["known_identities"].append(new_name)
                            result["auto_added_identities"].append(new_name)
                            result["archive_identities"].append(new_name)
                            result["scene_flags"]["has_unknown"] = True
                            result["cooldown_record_keys"].append(person["identity_key"])
                            route_reasons.append(f"unknown repeated {count}x and auto-added as {new_name}")
                            person["needs_pose"] = True
                        else:
                            save_unknown(unknown_db)
                            person["face_status"] = "repeat_unknown"
                            person["name"] = f"Unknown (seen {count}x)"
                            person["sighting_count"] = count
                            result["scene_flags"]["has_unknown"] = True
                            result["cooldown_record_keys"].append(cooldown_key)
                            route_reasons.append(f"repeat unknown seen {count}x")
                            person["needs_pose"] = True
                else:
                    new_uid = f"unk_{ts_str}_{person_index}"
                    unknown_db[new_uid] = {
                        "embedding": emb,
                        "count": 1,
                        "first_seen": datetime.now().isoformat(),
                        "last_seen": datetime.now().isoformat(),
                        "images": [str(crop_path)],
                    }
                    save_unknown(unknown_db)
                    person["face_status"] = "new_unknown"
                    person["name"] = "Unknown"
                    person["identity_key"] = f"unknown:{new_uid}"
                    result["scene_flags"]["has_unknown"] = True
                    result["cooldown_record_keys"].append(person["identity_key"])
                    route_reasons.append("new unknown person")
                    person["needs_pose"] = True

    # Pass 2: run pose only when needed for unknown / suspicious / no-face cases
    pose_needed = [idx for idx, person in enumerate(result["persons"]) if person.get("needs_pose")]
    if pose_needed:
        pose_res = yolo_pose(frame, conf=0.3, verbose=False)[0]
        for idx in pose_needed:
            person = result["persons"][idx]
            px1, py1, px2, py2 = person["bbox"]
            pose = person["pose"]
            if pose_res.keypoints is not None:
                best_iou, best_kp = 0.0, None
                for ki in range(len(pose_res.boxes)):
                    pb = pose_res.boxes.xyxy[ki].cpu().numpy().astype(int)
                    ix1 = max(px1, pb[0])
                    iy1 = max(py1, pb[1])
                    ix2 = min(px2, pb[2])
                    iy2 = min(py2, pb[3])
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    union = (px2 - px1) * (py2 - py1) + (pb[2] - pb[0]) * (pb[3] - pb[1]) - inter
                    iou = inter / union if union > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou
                        best_kp = pose_res.keypoints.data[ki].cpu().numpy()
                if best_kp is not None and best_iou > 0.3:
                    pose = analyze_pose(best_kp, (px1, py1, px2, py2))
            person["pose"] = pose
            if pose.get("behavior_score", 0) >= 2:
                if person.get("face_status") == "no_face":
                    result["scene_flags"]["has_no_face_suspicious"] = True
                    route_reasons.append(f"no face visible + suspicious pose: {pose.get('behavior_label')}")
                elif person.get("face_status") in {"new_unknown", "repeat_unknown", "auto_added"}:
                    result["scene_flags"]["has_suspicious"] = True
                    route_reasons.append(f"{person.get('face_status')} with suspicious pose: {pose.get('behavior_label')}")

    result["known_identities"] = unique_list(result["known_identities"])
    result["direct_known_identities"] = unique_list(result["direct_known_identities"])
    result["auto_added_identities"] = unique_list(result["auto_added_identities"])
    result["archive_identities"] = unique_list(result["archive_identities"])
    result["cooldown_record_keys"] = unique_list(result["cooldown_record_keys"])
    result["cooldown_skipped"] = unique_list(result["cooldown_skipped"])

    if (
        result["scene_flags"]["weapon_detected"]
        or result["scene_flags"]["has_unknown"]
        or result["scene_flags"]["has_suspicious"]
        or result["scene_flags"]["has_no_face_suspicious"]
    ):
        result["route"] = HERMES_ROUTE
        result["needs_hermes"] = True
    elif result["direct_known_identities"]:
        result["route"] = DIRECT_KNOWN_ROUTE
        result["needs_hermes"] = False
    else:
        result["route"] = "none"
        result["needs_hermes"] = False

    if result["route"] == "none" and result["cooldown_skipped"]:
        route_reasons.append("all matched identities currently on cooldown")

    result["route_reason"] = "; ".join(unique_list(route_reasons)) if route_reasons else result["route_reason"]
    return json.loads(json.dumps(result, default=str))


# ──────────────────────────────────────────────
# PRETTY CONSOLE LOG
# ──────────────────────────────────────────────
def log_results(result: Dict[str, Any], image_url: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"📸 Image: {image_url}")
    print(f"🕐 Time: {result.get('timestamp')}")
    print(f"📐 Size: {result.get('image_size', '?')}")
    print(f"🧭 Route: {result.get('route', 'none')} — {result.get('route_reason', '')}")
    print(f"{'─' * 60}")

    for vehicle in result.get("vehicles", []):
        print(f"  🚗 {vehicle['label']} ({vehicle['confidence']:.0%}) at {vehicle['bbox']}")
    for weapon in result.get("weapons", []):
        print(f"  🔫 WEAPON: {weapon['label']} ({weapon['confidence']:.0%}) at {weapon['bbox']}")

    persons = result.get("persons", [])
    if not persons:
        print("  👤 No persons detected")
    else:
        print(f"  👤 {len(persons)} person(s) detected:")
        for i, person in enumerate(persons, start=1):
            print(f"\n  ── Person {i} ──")
            print(f"     Bbox: {person['bbox']}")
            print(f"     Confidence: {person['confidence']}")
            print(f"     Face: {person.get('face_status')} {person.get('name') or ''}".rstrip())
            if person.get("on_cooldown"):
                print("     Cooldown: YES")
            face_info = person.get("face_info")
            if face_info:
                print(f"     Age: {face_info['age']} (raw: {face_info['raw_age']}, {face_info['size_category']})")
                print(f"     Gender: {face_info['gender']}")
            pose = person.get("pose", {})
            print(f"     Behavior: {pose.get('behavior_label', '?')} (score: {pose.get('behavior_score', 0)})")
            if person.get("weapons_near"):
                print(f"     Weapons near: {', '.join(person['weapons_near'])}")
    print(f"{'=' * 60}\n")


# ──────────────────────────────────────────────
# HERMES INTEGRATION
# ──────────────────────────────────────────────
async def run_hermes_cli(provider: str, model: str, prompt: str) -> Dict[str, Any]:
    logger.info(f"🤖 Hermes route → provider={provider} model={model}")
    result = await asyncio.to_thread(
        subprocess.run,
        ["hermes", "chat", "--provider", provider, "--model", model, "-q", prompt],
        capture_output=True,
        text=True,
        timeout=HERMES_TIMEOUT,
        cwd=str(BASE_DIR),
    )
    response = result.stdout.strip()
    stderr = result.stderr.strip()

    if result.returncode != 0:
        logger.error(f"Hermes exited with code {result.returncode}")
        if stderr:
            logger.error(f"Hermes stderr: {stderr[:1000]}")
    else:
        logger.info(f"✅ Hermes responded ({len(response)} chars)")

    if response:
        logger.info("── Hermes response ──")
        for line in response.splitlines():
            if line.strip():
                logger.info(f"  | {line}")
        logger.info("── End Hermes response ──")

    if stderr:
        logger.info("── Hermes stderr ──")
        for line in stderr.splitlines()[-20:]:
            if line.strip():
                logger.info(f"  | {line}")
        logger.info("── End Hermes stderr ──")

    return {
        "returncode": result.returncode,
        "response": response,
        "stderr": stderr,
    }


async def run_hermes_vision_threat(ml_result: Dict[str, Any], image_url: str, camera: str, job_id: Optional[str] = None) -> Dict[str, Any]:
    people = build_persons_summary(ml_result)
    vehicles_str = ", ".join(f"{v['label']} ({v['confidence']})" for v in ml_result.get("vehicles", [])) or "None"
    weapons_str = ", ".join(f"{w['label']} ({w['confidence']})" for w in ml_result.get("weapons", [])) or "None"
    prompt = f"""You are the Hermes Vision Threat Agent for a security camera pipeline.

This event MUST go through the Security Camera Analysis skill and actual image inspection.

Image file: {ml_result.get('image_path')}
Image URL: {image_url}
Camera: {camera}
Time: {ml_result.get('timestamp')}
Job ID: {job_id or 'none'}
Route: {ml_result.get('route')}
Route reason: {ml_result.get('route_reason')}
Known identities in frame: {', '.join(ml_result.get('known_identities', [])) or 'None'}
Auto-added identities this hit: {', '.join(ml_result.get('auto_added_identities', [])) or 'None'}
ML counts:
- Persons: {len(ml_result.get('persons', []))}
- Vehicles: {len(ml_result.get('vehicles', []))}
- Weapons: {len(ml_result.get('weapons', []))}
Vehicles: {vehicles_str}
Weapons: {weapons_str}
People summary:
{chr(10).join(people) if people else 'None'}

In the Notion Description field, prepend these exact source tags at the top before the descriptive analysis:
{build_source_tag_text(HERMES_ROUTE, 'pending', job_id=job_id)}

Instructions:
1. Use the Security Camera Analysis skill.
2. Actually inspect the image using vision; do not rely on metadata alone.
3. Log the result to Notion.
4. Send Telegram only if threat is MEDIUM or above.
5. Base your final threat level on the image, not just ML.
6. In your Notion Description, replace image_category=pending with image_category=normal or image_category=threat after you decide.

After you finish, the FINAL line must be exactly one JSON object in this form:
JSON_RESULT: {{"agent":"vision_threat","status":"ok","notion_logged":true,"telegram_sent":false,"threat_level":"LOW","score":2,"summary":"one-sentence summary","category":"normal"}}

Rules for category:
- category must be threat for MEDIUM/HIGH/CRITICAL
- category must be normal for NONE/LOW
- if vision fails, set threat_level to MEDIUM, category to threat, and explain in summary
"""

    cli_result = await run_hermes_cli(HERMES_PROVIDER, HERMES_VISION_MODEL, prompt)
    response = cli_result["response"]
    parsed = parse_json_result(response) or {}
    if not parsed:
        parsed = {
            "agent": HERMES_ROUTE,
            "status": "fallback",
            "notion_logged": False,
            "telegram_sent": False,
            "threat_level": infer_threat_level_from_text(response, default="UNKNOWN"),
            "score": None,
            "summary": "Hermes returned unstructured output",
            "category": "threat",
        }
    parsed["raw_response"] = response
    parsed["returncode"] = cli_result["returncode"]
    return parsed


# ──────────────────────────────────────────────
# API
# ──────────────────────────────────────────────
@app.on_event("startup")
async def on_startup() -> None:
    start_background_worker()


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...), metadata: str = Form(...)):
    start_background_worker()
    try:
        meta = json.loads(metadata)
    except Exception:
        raise HTTPException(400, "Invalid metadata JSON")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"det_{ts}.jpg"
    filepath = UPLOAD_DIR / filename

    async with aiofiles.open(filepath, "wb") as f:
        await f.write(await image.read())

    image_url = f"{VPS_HOST.rstrip('/')}/images/{filename}"
    camera = meta.get("camera", "Unknown")

    logger.info("=" * 50)
    logger.info(f"📸 Received from {camera} ({filepath.stat().st_size // 1024}KB)")
    logger.info("🔍 Running ML pipeline...")

    ml_result = await asyncio.to_thread(run_ml, str(filepath))
    ml_result["image_url"] = image_url
    log_results(ml_result, image_url)

    response_payload: Dict[str, Any] = {
        "status": "ok",
        "image_url": image_url,
        "camera": camera,
        "route": ml_result.get("route"),
        "route_reason": ml_result.get("route_reason"),
        "persons": len(ml_result.get("persons", [])),
        "known_identities": ml_result.get("known_identities", []),
        "auto_added_identities": ml_result.get("auto_added_identities", []),
        "cooldown_skipped": ml_result.get("cooldown_skipped", []),
    }

    log_entry = {
        "phase": "request_received",
        "timestamp": datetime.now().isoformat(),
        "image": filename,
        "image_url": image_url,
        "image_path": str(filepath),
        "camera": camera,
        "route": ml_result.get("route"),
        "route_reason": ml_result.get("route_reason"),
        "persons": len(ml_result.get("persons", [])),
        "vehicles": len(ml_result.get("vehicles", [])),
        "weapons": len(ml_result.get("weapons", [])),
        "known_identities": ml_result.get("known_identities", []),
        "auto_added_identities": ml_result.get("auto_added_identities", []),
        "cooldown_skipped": ml_result.get("cooldown_skipped", []),
        "ml_result": ml_result,
    }

    if ml_result.get("route") == DIRECT_KNOWN_ROUTE:
        notion_result = await asyncio.to_thread(direct_log_known_to_notion, ml_result, image_url, camera)
        for key in ml_result.get("cooldown_record_keys", []):
            record_identity_processing(key)
        archive_paths: Dict[str, str] = {}
        for name in ml_result.get("direct_known_identities", []):
            archived = archive_known_sighting(name, str(filepath), "normal", ts)
            if archived:
                archive_paths[name] = archived
        response_payload.update(
            {
                "processing_mode": "direct_python_notion",
                "queued": False,
                "notion_logged": notion_result.get("notion_logged"),
                "telegram_sent": False,
                "image_category": "normal",
                "archive_paths": archive_paths,
            }
        )
        log_entry.update(
            {
                "processing_mode": "direct_python_notion",
                "notion_result": notion_result,
                "image_category": "normal",
                "archive_paths": archive_paths,
            }
        )
    elif ml_result.get("route") == HERMES_ROUTE:
        try:
            job_id = queue_hermes_job(ml_result, image_url, camera)
            for key in ml_result.get("cooldown_record_keys", []):
                record_identity_processing(key)
            response_payload.update(
                {
                    "processing_mode": "async_hermes_queue",
                    "queued": True,
                    "job_id": job_id,
                    "job_status": "queued",
                    "notion_logged": None,
                    "telegram_sent": None,
                    "image_category": "pending",
                }
            )
            log_entry.update(
                {
                    "processing_mode": "async_hermes_queue",
                    "job_id": job_id,
                    "job_status": "queued",
                }
            )
        except queue.Full:
            logger.error("Hermes queue is full")
            response_payload.update(
                {
                    "status": "busy",
                    "processing_mode": "async_hermes_queue",
                    "queued": False,
                    "error": "Hermes queue is full",
                }
            )
            log_entry.update({"processing_mode": "async_hermes_queue", "queue_error": "full"})
    else:
        response_payload.update(
            {
                "processing_mode": "none",
                "queued": False,
                "notion_logged": None,
                "telegram_sent": None,
                "image_category": "normal",
            }
        )
        log_entry.update({"processing_mode": "none", "image_category": "normal"})

    append_log_entry(log_entry)
    logger.info("=" * 50)
    return JSONResponse(response_payload)


@app.get("/health")
async def health():
    with JOB_STATUS_LOCK:
        jobs = len(JOB_STATUS)
    with COOLDOWN_LOCK:
        prune_cooldowns()
        cooldowns = len(RECENT_PROCESSING)
    return {
        "status": "ok",
        "known_faces": len(known_db),
        "unknown_tracked": len(unknown_db),
        "auto_add_after": AUTO_ADD_AFTER,
        "cooldown_seconds": COOLDOWN_SECONDS,
        "queue_size": HERMES_QUEUE.qsize(),
        "job_count": jobs,
        "active_cooldowns": cooldowns,
        "notion_direct_enabled": bool(NOTION_API_KEY),
        "hermes": {
            "provider": HERMES_PROVIDER,
            "vision_model": HERMES_VISION_MODEL,
            "binary": shutil.which("hermes") is not None,
            "skill_installed": (Path.home() / ".hermes/skills/security-camera-analysis/SKILL.md").exists(),
            "worker_alive": bool(WORKER_THREAD and WORKER_THREAD.is_alive()),
        },
        "models": {
            "yolo": True,
            "yolo_pose": True,
            "insightface": True,
            "weapon": weapon_det is not None,
        },
    }


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    with JOB_STATUS_LOCK:
        job = JOB_STATUS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/known")
async def list_known():
    return {"faces": list(known_db.keys()), "count": len(known_db)}


@app.get("/unknowns")
async def list_unknowns():
    summary = {
        uid: {"count": data["count"], "first": data["first_seen"], "last": data["last_seen"]}
        for uid, data in unknown_db.items()
    }
    return {"unknowns": summary, "count": len(unknown_db)}


@app.get("/logs")
async def get_logs(limit: int = 20):
    log_file = LOG_DIR / "log.jsonl"
    if not log_file.exists():
        return {"logs": []}
    async with aiofiles.open(log_file, "r") as f:
        lines = await f.readlines()
    logs = []
    for line in lines[-limit:]:
        try:
            logs.append(json.loads(line))
        except Exception:
            pass
    return {"logs": logs}


if __name__ == "__main__":
    import uvicorn

    start_background_worker()
    logger.info(f"\n✓ Known faces: {list(known_db.keys()) if known_db else '(none)'}")
    logger.info(f"✓ Unknown tracked: {len(unknown_db)}")
    logger.info(f"✓ Images at: {VPS_HOST.rstrip('/')}/images/")
    logger.info(f"✓ Auto-add threshold: {AUTO_ADD_AFTER}")
    logger.info(f"✓ Cooldown seconds: {COOLDOWN_SECONDS}")
    logger.info(f"✓ Direct Notion logging: {'enabled' if NOTION_API_KEY else 'disabled'}")

    if shutil.which("hermes"):
        logger.info("✓ Hermes Agent available")
    else:
        logger.error("✗ Hermes not found in PATH")

    skill_path = Path.home() / ".hermes/skills/security-camera-analysis/SKILL.md"
    if skill_path.exists():
        logger.info("✓ Security skill installed")
    else:
        logger.warning("⚠ Security skill not found")

    logger.info("\n🚀 Pipeline: Image → ML → direct Notion or async Hermes queue → Archive")
    logger.info(f"Starting server on :{API_PORT}\n")
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
