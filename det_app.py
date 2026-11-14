"""BD532 Lab 2.2 — Real-time Object Detection API (ฉบับสมบูรณ์)

รัน:  uvicorn det_app:app --reload --port 8001
เปิด: http://localhost:8001/        (หน้าเว็บกล้อง)
      http://localhost:8001/docs    (Swagger)
"""
import io
import json
import os
import threading
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")   # ห้าม ultralytics pip install เองตอนรันจริง

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO

# ---------- 1) โหลดโมเดล ----------
DEFAULT_MODEL = "models/det_best.pt"
MODEL_PATH = os.getenv("DET_MODEL", DEFAULT_MODEL)       # เปลี่ยนเป็น yolo26n.pt เพื่อเทียบกับโมเดลก่อน fine-tune
META = json.loads(Path("models/det_metadata.json").read_text(encoding="utf-8"))
model = YOLO(MODEL_PATH, task="detect")
if MODEL_PATH != DEFAULT_MODEL:                           # ใช้โมเดลอื่น → metadata ต้องเปลี่ยนตาม
    META = {**META, "model_version": Path(MODEL_PATH).stem, "names": model.names,
            "base_model": MODEL_PATH, "dataset": "COCO (80 classes)",
            "violation_classes": [], "metrics": None}
VIOLATIONS = set(META["violation_classes"])
model_lock = threading.Lock()

MAX_BYTES = 5 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(
    title="BD532 Object Detection API",
    description="YOLO26n fine-tune บน Construction-PPE (12 คลาส) + หน้าเว็บกล้องแบบ real-time",
    version=META["model_version"],
)
# ---------- 2) สัญญาของ response ----------
class Detection(BaseModel):
    label: str
    class_id: int
    conf: float
    box: list[float]            # [x1, y1, x2, y2] หน่วย pixel ของรูปที่ส่งมา


class DetResponse(BaseModel):
    detections: list[Detection]
    counts: dict[str, int]
    violations: int             # จำนวนกล่องที่เป็นคลาส no_*
    image_size: list[int]       # [width, height]
    model_version: str
    latency_ms: float


# ---------- 3) helper ----------
def decode_image(data: bytes) -> Image.Image:
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "ไฟล์ใหญ่เกิน 5 MB")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "อ่านไฟล์รูปไม่ได้ (ไฟล์เสียหรือไม่ใช่รูป)")


def read_upload(upload: UploadFile) -> Image.Image:
    if upload.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"รองรับเฉพาะ {sorted(ALLOWED_TYPES)} แต่ได้ {upload.content_type}")
    return decode_image(upload.file.read(MAX_BYTES + 1))


def run_model(img: Image.Image, conf: float, iou: float):
    t0 = time.perf_counter()
    with model_lock:
        r = model.predict(img, imgsz=META["imgsz"], conf=conf, iou=iou, verbose=False)[0]
    return r, (time.perf_counter() - t0) * 1000


def to_response(r, latency_ms: float) -> DetResponse:
    dets = [
        Detection(label=r.names[int(c)], class_id=int(c), conf=round(float(p), 4),
                  box=[round(float(v), 1) for v in b])
        for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())
    ]
    counts = Counter(d.label for d in dets)
    h, w = r.orig_shape
    return DetResponse(
        detections=dets,
        counts=dict(counts),
        violations=sum(n for k, n in counts.items() if k in VIOLATIONS),
        image_size=[w, h],
        model_version=META["model_version"],
        latency_ms=round(latency_ms, 1),
    )


def detect_image(img: Image.Image, conf: float, iou: float) -> DetResponse:
    r, ms = run_model(img, conf, iou)
    return to_response(r, ms)


# ---------- 4) REST endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return META


@app.post("/detect", response_model=DetResponse)
def detect(file: UploadFile = File(..., description="ไฟล์รูป jpg/png/webp"),
           conf: float = Query(0.25, ge=0.0, le=1.0, description="confidence ขั้นต่ำ"),
           iou: float = Query(0.7, ge=0.0, le=1.0, description="IoU threshold ของ NMS")):
    return detect_image(read_upload(file), conf, iou)


@app.post("/detect/image", responses={200: {"content": {"image/jpeg": {}}}})
def detect_annotated(file: UploadFile = File(...),
                     conf: float = Query(0.25, ge=0.0, le=1.0),
                     iou: float = Query(0.7, ge=0.0, le=1.0)):
    r, ms = run_model(read_upload(file), conf, iou)
    plotted = r.plot()[..., ::-1]                     # BGR → RGB
    buf = io.BytesIO()
    Image.fromarray(plotted).save(buf, format="JPEG", quality=85)
    return Response(buf.getvalue(), media_type="image/jpeg",
                    headers={"X-Latency-ms": f"{ms:.1f}", "X-Detections": str(len(r.boxes))})


# ---------- 5) หน้าเว็บกล้อง (Step B3) ----------
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse("static/index.html")


# ---------- 6) WebSocket (Step B4) ----------
@app.websocket("/ws/detect")
async def ws_detect(ws: WebSocket,
                    conf: float = Query(0.25, ge=0.0, le=1.0),
                    iou: float = Query(0.7, ge=0.0, le=1.0)):
    """รับเฟรม JPEG (binary) ทีละเฟรม → ตอบผล detection เป็น JSON ทีละเฟรม"""
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            try:
                img = decode_image(data)
            except HTTPException as e:
                await ws.send_json({"error": e.detail})
                continue
            res = await run_in_threadpool(detect_image, img, conf, iou)   # ไม่ block event loop
            await ws.send_json(res.model_dump())
    except WebSocketDisconnect:
        pass
