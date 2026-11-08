"""Local panel. Bound to 127.0.0.1 by the CLI; photos are not uploaded anywhere."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from recognize import results, thumbs
from recognize.camera_flow import (
    CameraSession,
    FrameReport,
    record_sweep_frame,
    save_enrollment,
)
from recognize.clothing import hint_for
from recognize.engine import FaceEngine, GalleryRecognizer
from recognize.enroll import enroll_faces
from recognize.errors import AppError
from recognize.eta import Plan, estimate_eta, plan_units
from recognize.gallery import (
    Person,
    find_person,
    load_people,
    load_people_safe,
    present_slugs,
    purge_deleted,
    reference_room,
    restore_person,
    soft_delete,
)
from recognize.match import COSINE_THRESHOLD, DEFAULT_SAMPLE_SECONDS, ScanOptions
from recognize.media import decode_image, resize_max_edge
from recognize.messages import (
    ANGLE_GUIDANCE,
    CLOTHING_NOTE,
    ONE_PHOTO_NOTE,
    THRESHOLD_HINT,
    VIDEO_INTERVAL_HINT,
)
from recognize.models_io import SFACE_NAME, YUNET_NAME
from recognize.paths import (
    default_output_dir,
    default_pool_dir,
    event_source_dir,
    models_dir,
)
from recognize.picker import pick_folder, platform_name
from recognize.pool import REPORT_NAME, build_pool
from recognize.pose import HOLD_FRAMES
from recognize.results import REJECT_DIR, ResultsError
from recognize.reveal import failure_message, open_folder, reveal_file
from recognize.scan import ScanCancelled, ScanHooks, run_scan, validate_scan
from recognize.store import archive_report, recent_scans, remember_scan
from recognize.sweep import SWEEP_TOTAL, FaceSweep
from recognize.textutil import clean_person_name, folder_name
from recognize.thumbs import ThumbError

STATIC = Path(__file__).resolve().parent / "static"
_MAX_FILES = 12
_MAX_BYTES = 20 * 1024 * 1024
_PENDING_TTL = 60 * 60
_log = logging.getLogger("recognize")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    try:
        purge_deleted()
    except OSError:
        _log.exception("Could not purge the deleted-people folder")
    yield
    thumbs.stop_warm_up()


app = FastAPI(title="Etkinlik arşivi", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
# Strict on purpose: urlsplit would accept userinfo and other tricks.
_HOST = re.compile(r"^(?:\[(?P<v6>[0-9A-Fa-f:.]+)\]|(?P<name>[A-Za-z0-9.-]+))(?::\d{1,5})?$")
_SECURITY_HEADERS = {
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "same-origin",
}
_PICK_TITLES = {
    "source": "Fotoğraf ve videoların olduğu klasörü seçin",
    "destination": "Kişi klasörlerinin yazılacağı klasörü seçin",
}
_pick_lock = threading.Lock()
_open_lock = threading.Lock()
_last_open: tuple[str, float] = ("", 0.0)
_OPEN_COALESCE_SECONDS = 1.5


def _host_allowed(request: Request) -> bool:
    match = _HOST.fullmatch(request.headers.get("host", ""))
    host = None if match is None else (match["v6"] or match["name"]).lower()
    if host in _LOCAL_HOSTS:
        return True
    # Starlette's TestClient: Host "testserver" from the fake peer "testclient". A real socket
    # peer is always an IP address, so no browser can take this branch.
    return (
        host == "testserver" and request.client is not None and request.client.host == "testclient"
    )


@app.middleware("http")
async def local_pages_only(request: Request, call_next):
    """Only this machine's browser, only from the panel's own page, and never framed."""
    if not _host_allowed(request):
        return _secured(
            JSONResponse(
                {"detail": "Bu panel yalnızca http://127.0.0.1 adresinden açılır."},
                status_code=400,
            )
        )
    path = request.url.path
    if path.startswith("/api") and request.headers.get("sec-fetch-site") == "cross-site":
        return _secured(
            JSONResponse({"detail": "Bu panel yalnızca kendi sayfasından kullanılır."}, 403)
        )
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        origin = request.headers.get("origin")
        if origin and urlparse(origin).hostname not in _LOCAL_HOSTS:
            return _secured(
                JSONResponse(
                    {"detail": "Bu panel yalnızca kendi sayfasından kullanılır."},
                    status_code=403,
                )
            )
    response = await call_next(request)
    if not path.startswith("/api"):
        # Pages and scripts change with every release; the browser must ask each time.
        response.headers.setdefault("Cache-Control", "no-cache")
    return _secured(response)


def _secured(response: Response) -> Response:
    for name, value in _SECURITY_HEADERS.items():
        response.headers.setdefault(name, value)
    return response


@app.exception_handler(RequestValidationError)
async def _invalid_request(_request: Request, exc: RequestValidationError) -> JSONResponse:
    fields = sorted(
        {
            ".".join(str(part) for part in error.get("loc", ()) if part != "body")
            for error in exc.errors()
        }
    )
    return JSONResponse(
        {
            "detail": "İstek eksik ya da hatalı. Sayfayı yenileyip yeniden deneyin.",
            "fields": [field for field in fields if field],
        },
        status_code=422,
    )


@app.exception_handler(Exception)
async def _unexpected(_request: Request, exc: Exception) -> JSONResponse:
    _log.error("Unhandled error", exc_info=exc)
    return _secured(
        JSONResponse(
            {"detail": "Beklenmeyen bir hata oldu. Ayrıntı panelin çalıştığı terminalde."},
            status_code=500,
        )
    )


_engine_lock = threading.Lock()
_engine_init_lock = threading.Lock()
_engine: FaceEngine | None = None
_pending_lock = threading.Lock()
_pending: dict[str, _Pending] = {}
_scan_lock = threading.Lock()
_camera_lock = threading.Lock()
_camera: dict[str, _CameraPending] = {}
_CAMERA_TTL = 30 * 60
_cancel = threading.Event()
_scan: dict[str, object] = {
    "state": "idle",
    "done": 0,
    "total": 0,
    "current": "",
    "summary": None,
    "report": None,
    "error": None,
    "phase": None,
    "startedAt": None,
    "finishedAt": None,
    "elapsedSeconds": None,
    "etaSeconds": None,
    "units": None,
    "source": None,
    "destination": None,
    "move": False,
    "people": [],
    "matched": 0,
    "perPerson": {},
    "lastMatch": None,
    "cancelRequested": False,
    "progress": None,
}
_scan_started_mono = 0.0
_scan_finished_mono = 0.0
_plan: Plan | None = None
# Copies written by the running scan, so their thumbnails can be served before a checkpoint.
_live_copies: set[str] = set()
_pool_lock = threading.Lock()
_pool: dict[str, object] = {
    "state": "idle",
    "done": 0,
    "total": 0,
    "current": "",
    "summary": None,
    "destination": None,
    "error": None,
}


class _PendingFace:
    def __init__(self, index: int, det_score: float, embedding: object, jpeg: bytes) -> None:
        self.index = index
        self.det_score = det_score
        self.embedding = embedding
        self.jpeg = jpeg


class _PendingImage:
    def __init__(self, filename: str, faces: list[_PendingFace]) -> None:
        self.filename = filename
        self.faces = faces


class _Pending:
    def __init__(self, images: list[_PendingImage]) -> None:
        self.created = time.monotonic()
        self.images = images


class _CameraPending:
    def __init__(self, session: CameraSession) -> None:
        self.created = time.monotonic()
        self.session = session


class FaceChoice(BaseModel):
    image: int
    face: int


class EnrollBody(BaseModel):
    token: str
    name: str
    clothing: str | None = None
    faces: list[FaceChoice] = Field(default_factory=list)


class ScanBody(BaseModel):
    source: str
    destination: str
    threshold: float = COSINE_THRESHOLD
    interval: float = DEFAULT_SAMPLE_SECONDS
    move: bool = False
    # Slugs to sort for. None means everyone who is enrolled.
    people: list[str] | None = None


class PoolBody(BaseModel):
    source: str
    destination: str


class PickBody(BaseModel):
    purpose: Literal["source", "destination"]
    current: str = ""


class CameraStart(BaseModel):
    name: str = ""


class CameraToken(BaseModel):
    token: str


class RestoreBody(BaseModel):
    token: str


class OpenBody(BaseModel):
    output: str
    slug: str | None = None
    path: str | None = None
    target: Literal["rejected"] | None = None


class RejectBody(BaseModel):
    output: str
    path: str
    slug: str | None = None


class UndoBody(BaseModel):
    output: str
    path: str


class AsideBody(BaseModel):
    output: str
    slug: str | None = None


# ---------------------------------------------------------------- pages


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/kamera")
def camera_page() -> FileResponse:
    return FileResponse(STATIC / "kamera.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    icon = STATIC / "favicon.svg"
    if not icon.is_file():
        return Response(status_code=204)
    return FileResponse(
        icon, media_type="image/svg+xml", headers={"Cache-Control": "max-age=86400"}
    )


# ---------------------------------------------------------------- settings and people


@app.get("/api/ayarlar")
def settings() -> dict[str, object]:
    source = event_source_dir()
    original = event_source_dir(prefer_pool=False)
    last_output = results.default_output()
    return {
        "source": "" if source is None else str(source),
        "poolSource": "" if original is None else str(original),
        "pool": str(default_pool_dir()),
        "poolReady": source is not None and source == default_pool_dir().resolve(),
        "poolSummary": _saved_summary(default_pool_dir() / REPORT_NAME),
        "platform": platform_name(),
        "lastScan": _saved_summary(last_output / "rapor.json"),
        "lastScanOutput": str(last_output),
        "output": str(default_output_dir()),
        "threshold": COSINE_THRESHOLD,
        "interval": DEFAULT_SAMPLE_SECONDS,
        "angleGuidance": ANGLE_GUIDANCE,
        "onePhotoNote": ONE_PHOTO_NOTE,
        "clothingNote": CLOTHING_NOTE,
        "thresholdHint": THRESHOLD_HINT,
        "videoHint": VIDEO_INTERVAL_HINT,
        "rejectFolder": REJECT_DIR,
    }


@app.get("/api/kisiler")
def people() -> dict[str, object]:
    people_list, problems = load_people_safe()
    last = _last_scan_people()
    rows = []
    for person in people_list:
        rows.append(
            {
                "name": person.name,
                "slug": person.slug,
                "createdAt": person.created_at or None,
                "references": len(person.references),
                "clothing": None if person.clothing is None else person.clothing.name,
                "clothingLabel": None if person.clothing is None else person.clothing.label,
                "cover": _cover_url(person),
                "lastScan": last.get(person.slug),
            }
        )
    return {"people": rows, "problems": problems}


def _cover_url(person: Person) -> str | None:
    if not person.references:
        return None
    return f"/api/kisiler/{person.slug}/kapak.jpg?v={_cover_version(person)}"


def _cover_version(person: Person) -> str:
    try:
        stamp = int((person.directory / "person.json").stat().st_mtime)
    except OSError:
        stamp = 0
    return f"{len(person.references)}-{stamp}"


def _last_scan_people() -> dict[str, dict[str, object]]:
    """Per-person counts of the most recent report, for the person cards."""
    output = results.default_output()
    try:
        report = results.load_report(output)
    except (ResultsError, OSError):
        return {}
    summary = report.get("summary") if isinstance(report, dict) else None
    if not isinstance(summary, dict):
        return {}
    found: dict[str, dict[str, object]] = {}
    for row in summary.get("people") or []:
        if not isinstance(row, dict) or not isinstance(row.get("slug"), str):
            continue
        found[row["slug"]] = {
            "output": str(output),
            "scannedAt": report.get("scannedAt"),
            "complete": report.get("complete") is not False,
            "files": int(row.get("files") or 0),
            "smallFaces": int(row.get("smallFaces") or 0),
            "learned": int(row.get("learned") or 0),
            "rejected": int(row.get("rejected") or 0),
        }
    return found


@app.get("/api/kisiler/{slug}/kapak.jpg")
def cover(slug: str, request: Request) -> Response:
    _check_slug(slug)
    person = find_person(slug)
    if person is None or not person.references:
        raise HTTPException(status_code=404, detail="Kapak yok.")
    relative = person.references[0].get("file")
    if not isinstance(relative, str):
        raise HTTPException(status_code=404, detail="Kapak yok.")
    path = (person.directory / relative).resolve()
    if person.directory.resolve() not in path.parents:
        raise HTTPException(status_code=404, detail="Kapak yok.")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Kapak yok.")
    stat = path.stat()
    etag = f'"k-{stat.st_mtime_ns:x}-{stat.st_size:x}"'
    headers = {"Cache-Control": "private, no-cache", "ETag": etag}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    return FileResponse(path, media_type="image/jpeg", headers=headers)


@app.delete("/api/kisiler/{slug}")
def remove(slug: str) -> dict[str, str]:
    _check_slug(slug)
    _refuse_while_scanning()
    try:
        removal = soft_delete(slug)
    except AppError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OSError as exc:
        _log.exception("Could not move the person aside")
        raise HTTPException(status_code=500, detail="Kayıt silinemedi.") from exc
    return {"deleted": removal.slug, "name": removal.name, "restoreToken": removal.token}


@app.post("/api/kisiler/geri-al")
def restore(body: RestoreBody) -> dict[str, str]:
    token = body.token.strip()
    if not token or "/" in token or "\\" in token or token.startswith("."):
        raise HTTPException(status_code=400, detail="Geri alma belirteci geçersiz.")
    _refuse_while_scanning()
    try:
        removal = restore_person(token)
    except AppError as exc:
        status = 409 if "yeni bir kayıt" in str(exc) else 404
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except OSError as exc:
        _log.exception("Could not restore the person")
        raise HTTPException(status_code=500, detail="Kayıt geri alınamadı.") from exc
    return {"restored": removal.slug, "name": removal.name}


# ---------------------------------------------------------------- photo enrollment


@app.post("/api/onizleme")
async def preview(files: list[UploadFile] = File(...)) -> dict[str, object]:
    if _scan_running():
        raise HTTPException(status_code=409, detail="Tarama sürüyor. Bitince yeniden deneyin.")
    if not files or len(files) > _MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Bir seferde 1 ile {_MAX_FILES} arasında fotoğraf seçin.",
        )
    payloads: list[tuple[str, bytes]] = []
    for upload in files:
        data = await upload.read()
        if len(data) > _MAX_BYTES:
            raise HTTPException(status_code=400, detail=f"{upload.filename} 20 MB sınırını aşıyor.")
        payloads.append((upload.filename or "foto.jpg", data))
    try:
        token, images = await asyncio.to_thread(_build_preview, payloads)
    except AppError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"token": token, "images": images, "guidance": ANGLE_GUIDANCE}


@app.post("/api/kisiler")
def create_person(body: EnrollBody) -> dict[str, object]:
    if _scan_running():
        raise HTTPException(status_code=409, detail="Tarama sürüyor. Bitince yeniden deneyin.")
    try:
        name = clean_person_name(body.name)
    except AppError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with _pending_lock:
        _drop_expired()
        pending = _pending.get(body.token)
    if pending is None:
        raise HTTPException(
            status_code=400,
            detail="Önizleme zaman aşımına uğradı. Fotoğrafları yeniden seçin.",
        )
    samples: list[tuple[object, float, bytes, str]] = []
    seen: set[tuple[int, int]] = set()
    for choice in body.faces:
        if choice.image < 0 or choice.image >= len(pending.images):
            raise HTTPException(status_code=400, detail="Fotoğraf seçimi geçersiz.")
        image = pending.images[choice.image]
        face = next((item for item in image.faces if item.index == choice.face), None)
        if face is None:
            raise HTTPException(status_code=400, detail="Yüz seçimi geçersiz.")
        key = (choice.image, choice.face)
        if key in seen:
            continue
        seen.add(key)
        samples.append((face.embedding, face.det_score, face.jpeg, image.filename))
    if not samples:
        raise HTTPException(status_code=400, detail="En az bir yüz seçin.")
    clothing_key = (body.clothing or "").strip()
    try:
        if clothing_key:
            clothing = hint_for(clothing_key)
            update = True
        else:
            clothing = None
            update = find_person(name) is None
        prepared = [
            (np.asarray(item[0], dtype=np.float32), float(item[1]), item[2], str(item[3]))
            for item in samples
        ]
        person, warning = enroll_faces(name, prepared, clothing, update_clothing=update)
    except AppError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with _pending_lock:
        _pending.pop(body.token, None)
    return {
        "name": person.name,
        "slug": person.slug,
        "references": len(person.references),
        "clothingLabel": None if person.clothing is None else person.clothing.label,
        "warning": warning,
        "cover": _cover_url(person),
    }


# ---------------------------------------------------------------- scanning


@app.post("/api/tara")
def start_scan(body: ScanBody) -> dict[str, str]:
    if body.threshold < 0.2 or body.threshold > 0.8:
        raise HTTPException(status_code=400, detail="Eşik 0.20 ile 0.80 arasında olmalı.")
    if body.interval < 0.2 or body.interval > 5:
        raise HTTPException(
            status_code=400,
            detail="Video aralığı 0,2 ile 5 saniye arasında olmalı.",
        )
    source = _absolute(body.source, "Kaynak klasörün tam yolunu verin.")
    dest = _absolute(body.destination, "Çıktı klasörünün tam yolunu verin.")
    options = ScanOptions(threshold=body.threshold, sample_seconds=body.interval, move=body.move)
    people_list, problems = load_people_safe()
    try:
        validate_scan(source, dest, people_list)
        selected = _selected_people(people_list, body.people, problems)
    except AppError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _check_writable(dest)
    with _pool_lock:
        if _pool["state"] == "running":
            raise HTTPException(
                status_code=409, detail="Tekil havuz hazırlanıyor. Bitince taramayı başlatın."
            )
    global _scan_started_mono, _plan
    with _scan_lock:
        if _scan["state"] == "running":
            raise HTTPException(status_code=409, detail="Bir tarama zaten sürüyor.")
        _cancel.clear()
        _plan = None
        _live_copies.clear()
        _scan_started_mono = time.monotonic()
        _scan.update(
            state="running",
            done=0,
            total=0,
            current="Başlıyor",
            summary=None,
            report=None,
            error=None,
            phase="start",
            startedAt=datetime.now(UTC).isoformat(),
            finishedAt=None,
            elapsedSeconds=0.0,
            etaSeconds=None,
            units=None,
            source=str(source.resolve()),
            destination=str(dest.resolve()),
            move=body.move,
            people=[person.slug for person in selected],
            matched=0,
            perPerson={person.slug: 0 for person in selected},
            lastMatch=None,
            cancelRequested=False,
            progress=None,
            stoppedAt=None,
        )
    thread = threading.Thread(
        target=_scan_worker,
        args=(source.resolve(), dest.resolve(), options, [person.slug for person in selected]),
        name="recognize-scan",
        daemon=True,
    )
    thread.start()
    return {"state": "running"}


@app.get("/api/tara")
def scan_status() -> dict[str, object]:
    with _scan_lock:
        status = dict(_scan)
        plan = _plan
        if status["state"] == "running":
            elapsed = time.monotonic() - _scan_started_mono
        elif _scan_finished_mono and _scan_started_mono:
            elapsed = _scan_finished_mono - _scan_started_mono
        else:
            elapsed = None
    if elapsed is not None:
        status["elapsedSeconds"] = round(elapsed, 1)
    if plan is not None:
        status["units"] = {
            "done": round(plan.done_units, 2),
            "total": round(plan.total_units, 2),
        }
        if status["state"] == "running" and elapsed is not None:
            status["etaSeconds"] = estimate_eta(
                plan.done_units, plan.total_units, elapsed, plan.files_done
            )
    status["cancelRequested"] = _cancel.is_set() and status["state"] == "running"
    return status


@app.post("/api/tara/durdur")
def stop_scan() -> dict[str, str]:
    with _scan_lock:
        if _scan["state"] != "running":
            raise HTTPException(status_code=409, detail="Süren bir tarama yok.")
        _cancel.set()
        _scan["cancelRequested"] = True
        _scan["current"] = "Durduruluyor…"
    return {"state": "cancelling"}


@app.post("/api/klasor-sec")
async def choose_folder(body: PickBody) -> dict[str, str | None]:
    """Blocks until the user picks or cancels; the dialog lives on this machine's screen."""
    if not _pick_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Klasör penceresi zaten açık.")
    try:
        chosen = await asyncio.to_thread(pick_folder, _PICK_TITLES[body.purpose], body.current)
    except AppError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        _pick_lock.release()
    return {"path": chosen}


# ---------------------------------------------------------------- the unique pool


@app.post("/api/havuz")
def start_pool(body: PoolBody) -> dict[str, str]:
    source = _absolute(body.source, "Kaynak klasörün tam yolunu verin.")
    dest = _absolute(body.destination, "Havuz klasörünün tam yolunu verin.")
    if not source.is_dir():
        raise HTTPException(status_code=400, detail="Kaynak klasör bulunamadı.")
    if _scan_running():
        raise HTTPException(
            status_code=409, detail="Tarama sürüyor. Bitince havuzu hazırlayın."
        )
    with _pool_lock:
        if _pool["state"] == "running":
            raise HTTPException(status_code=409, detail="Havuz zaten hazırlanıyor.")
        _pool.update(
            state="running",
            done=0,
            total=0,
            current="Başlıyor",
            summary=None,
            destination=str(dest),
            error=None,
        )
    threading.Thread(
        target=_pool_worker,
        args=(source, dest),
        name="recognize-pool",
        daemon=True,
    ).start()
    return {"state": "running"}


@app.get("/api/havuz")
def pool_status() -> dict[str, object]:
    with _pool_lock:
        return dict(_pool)


# ---------------------------------------------------------------- results of a scan


@app.get("/api/sonuclar")
def scan_results(output: str | None = None, person: str | None = None) -> dict[str, object]:
    try:
        folder = results.resolve_output(output)
        report = results.hide_removed_people(results.load_report(folder), present_slugs())
        if person is not None:
            _check_slug(person)
        people_list, _problems = load_people_safe()
        enrolled = {p.slug: {"cover": _cover_url(p)} for p in people_list}
        return results.build_results(
            folder,
            report,
            person=person,
            scan_running=_scan_writes_to(folder),
            enrolled=enrolled,
        )
    except ResultsError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc


@app.get("/api/sonuclar/gecmis")
def scan_history() -> dict[str, object]:
    return {"items": recent_scans()}


@app.get("/api/sonuclar/onizleme")
def scan_thumbnail(
    request: Request,
    output: str,
    path: str,
    size: int = 640,
    v: str | None = None,
) -> Response:
    if size not in thumbs.SIZES:
        raise HTTPException(
            status_code=400, detail="Önizleme boyutu 320, 640 ya da 1280 olabilir."
        )
    try:
        folder = results.resolve_output(output)
        live = _live_copies_for(folder)
        try:
            report = results.load_report(folder)
        except ResultsError:
            if not live:
                raise
            report = {"results": []}
        report = results.hide_removed_people(report, present_slugs())
        target = results.listed_file(folder, report, path, extra=live)
        key, cached = thumbs.thumbnail(target, size)
    except (ResultsError, ThumbError) as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    etag = f'"t-{key[:20]}"'
    current = results.file_version(target)
    immutable = bool(v) and v == current
    cache = "private, max-age=31536000, immutable" if immutable else "private, no-cache"
    headers = {"ETag": etag, "Cache-Control": cache}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    return FileResponse(cached, media_type="image/jpeg", headers=headers)


@app.post("/api/klasoru-ac", response_model=None)
def open_in_file_manager(body: OpenBody) -> dict[str, object] | JSONResponse:
    given = [name for name in ("slug", "path", "target") if getattr(body, name) is not None]
    if len(given) > 1:
        raise HTTPException(status_code=400, detail="Açılacak yeri tek bir şekilde belirtin.")
    try:
        folder = results.resolve_output(body.output)
        report = results.load_report(folder)
        target, selected = _open_target(folder, report, body)
    except ResultsError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    key = str(target)
    with _open_lock:
        global _last_open
        now = time.monotonic()
        if _last_open[0] == key and now - _last_open[1] < _OPEN_COALESCE_SECONDS:
            return {"opened": str(target), "selected": selected, "already": True}
        _last_open = (key, now)
    try:
        if selected is not None:
            reveal_file(target)
        else:
            open_folder(target)
    except AppError:
        return JSONResponse({"detail": failure_message(), "path": str(target)}, status_code=500)
    return {"opened": str(target), "selected": selected, "already": False}


def _open_target(
    folder: Path, report: dict[str, object], body: OpenBody
) -> tuple[Path, str | None]:
    if body.path is not None:
        file = results.listed_file(folder, report, body.path)
        return file, str(file)
    if body.target == "rejected":
        target = folder / REJECT_DIR
        if not target.is_dir():
            raise ResultsError(404, "Bu klasörde ayıklanan dosya yok.")
        return target, None
    if body.slug is not None:
        _check_slug(body.slug)
        row = next(
            (
                item
                for item in report.get("people") or []
                if isinstance(item, dict) and item.get("slug") == body.slug
            ),
            None,
        )
        if row is None:
            raise ResultsError(404, "Bu kişi bu taramada yok.")
        try:
            name = folder_name(str(row.get("name") or ""))
        except AppError as exc:
            raise ResultsError(404, "Bu kişi bu taramada yok.") from exc
        counts = {
            str(item.get("slug")): int(item.get("files") or 0)
            for item in (report.get("summary") or {}).get("people") or []
            if isinstance(item, dict)
        }
        target = folder / name
        if not target.is_dir():
            if counts.get(body.slug, 0) == 0:
                raise ResultsError(404, "Bu kişi için klasör yazılmadı; eşleşme yok.")
            raise ResultsError(
                404, "Kişi klasörü bulunamadı. Taşınmış ya da silinmiş olabilir."
            )
        return target, None
    return folder, None


@app.post("/api/sonuclar/yanlis-eslesme")
def reject_match(body: RejectBody) -> dict[str, object]:
    if body.slug is not None:
        _check_slug(body.slug)
    try:
        folder = results.resolve_output(body.output)
        return results.reject_match(
            folder, body.path, body.slug, scan_running=_scan_writes_to(folder)
        )
    except ResultsError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc


@app.post("/api/sonuclar/geri-al")
def undo_reject(body: UndoBody) -> dict[str, object]:
    try:
        folder = results.resolve_output(body.output)
        return results.undo_reject(folder, body.path, scan_running=_scan_writes_to(folder))
    except ResultsError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc


@app.post("/api/sonuclar/eskileri-ayir")
def move_leftovers_aside(body: AsideBody) -> dict[str, object]:
    if body.slug is not None:
        _check_slug(body.slug)
    try:
        folder = results.resolve_output(body.output)
        return results.move_leftovers(
            folder,
            body.slug,
            scan_running=_scan_writes_to(folder),
            present=present_slugs(),
        )
    except ResultsError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc


# ---------------------------------------------------------------- camera enrollment


@app.post("/api/kamera/baslat")
def camera_start(body: CameraStart) -> dict[str, object]:
    _refuse_while_scanning()
    try:
        name = clean_person_name(body.name)
    except AppError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    room = reference_room(name)
    if room < SWEEP_TOTAL:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{name} için {SWEEP_TOTAL} yeni bakışa yer yok (kalan {room}). "
                "Kişiyi arşivden silip yeniden tanıtın."
            ),
        )
    session = CameraSession(name=name, samples=[], sweep=FaceSweep())
    token = uuid.uuid4().hex
    with _camera_lock:
        _drop_camera()
        _camera[token] = _CameraPending(session)
    return {
        "token": token,
        "name": name,
        "instruction": "Önce düz bak, yüzünü öğreniyorum",
        "coverage": session.sweep.coverage() if session.sweep is not None else [],
        "ready": False,
        "total": SWEEP_TOTAL,
    }


@app.post("/api/kamera/kare")
async def camera_frame(
    token: str = Form(...),
    frame: UploadFile = File(...),
) -> dict[str, object]:
    _refuse_while_scanning()
    _check_token(token)
    data = await frame.read()
    if not data:
        raise HTTPException(status_code=400, detail="Kare boş.")
    if len(data) > 2_000_000:
        raise HTTPException(status_code=400, detail="Kare çok büyük.")
    try:
        return await asyncio.to_thread(_assess_locked, token, data)
    except AppError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/kamera/hazir")
def camera_arm(body: CameraToken) -> dict[str, object]:
    _refuse_while_scanning()
    return _mutate_camera(body.token, "arm")


@app.post("/api/kamera/atla")
def camera_skip(body: CameraToken) -> dict[str, object]:
    _refuse_while_scanning()
    return _mutate_camera(body.token, "skip")


@app.post("/api/kamera/kaydet")
def camera_save(body: CameraToken) -> dict[str, object]:
    _refuse_while_scanning()
    _check_token(body.token)
    with _camera_lock:
        session = _live_session(body.token)
        try:
            person, warning = save_enrollment(session)
        except AppError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        ready = session.sweep is not None and session.sweep.covered
        if session.guide is not None:
            finished = session.guide.finished
            required_done = session.guide.required_done
        else:
            finished = ready
            required_done = ready
        return {
            "name": person.name,
            "slug": person.slug,
            "references": len(person.references),
            "warning": warning,
            "finished": finished,
            "requiredDone": required_done,
            "ready": ready,
        }


# ---------------------------------------------------------------- helpers


def _absolute(raw: str, message: str) -> Path:
    text = (raw or "").strip()
    path = Path(text).expanduser() if text else None
    if path is None or not path.is_absolute():
        raise HTTPException(status_code=400, detail=message)
    return path


def _check_writable(dest: Path) -> None:
    try:
        dest.mkdir(parents=True, exist_ok=True)
        probe = dest / ".recognize-yazma-denemesi"
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as exc:
        raise HTTPException(status_code=400, detail="Çıktı klasörüne yazılamıyor.") from exc


def _scan_running() -> bool:
    with _scan_lock:
        return _scan["state"] == "running"


def _scan_writes_to(output: Path) -> bool:
    with _scan_lock:
        if _scan["state"] != "running":
            return False
        destination = _scan.get("destination")
    if not isinstance(destination, str):
        return True
    try:
        return Path(destination).resolve() == output.resolve()
    except OSError:
        return True


def _live_copies_for(output: Path) -> set[str]:
    if not _scan_writes_to(output):
        return set()
    with _scan_lock:
        return set(_live_copies)


def _selected_people(
    people_list: list[Person], slugs: list[str] | None, problems: list[dict[str, str]]
) -> list[Person]:
    if slugs is None:
        return people_list
    wanted = set(slugs)
    broken = {row["folder"] for row in problems}
    if wanted & broken:
        raise AppError("Seçilen kişilerden birinin kaydı bozuk. Kaydı silip yeniden tanıtın.")
    selected = [person for person in people_list if person.slug in wanted]
    if not selected:
        raise AppError("Ayırmak için en az bir kişi seçin.")
    if len(selected) != len(wanted):
        raise AppError("Seçilen kişilerden biri artık kayıtlı değil. Sayfayı yenileyin.")
    return selected


def _models_missing() -> bool:
    folder = models_dir()
    return not ((folder / YUNET_NAME).is_file() and (folder / SFACE_NAME).is_file())


def _scan_worker(source: Path, dest: Path, options: ScanOptions, slugs: list[str]) -> None:
    global _plan, _scan_finished_mono
    thumbs.stop_warm_up()
    try:
        people_list, problems = load_people_safe()
        selected = _selected_people(people_list, slugs, problems)
        if _engine is None and _models_missing():
            with _scan_lock:
                _scan.update(
                    phase="models", current="Yüz modeli hazırlanıyor (ilk çalıştırmada indirilir)"
                )
        engine = _get_engine()
        recognizer = GalleryRecognizer(
            engine, people_list, options, targets={person.slug for person in selected}
        )

        def progress(done: int, total: int, label: str) -> None:
            plan = _plan
            if plan is not None and label:
                candidate = source / label
                if candidate in plan.frames and plan._current != candidate:
                    plan.start_file(candidate)
            with _scan_lock:
                _scan["done"] = done
                _scan["total"] = total
                _scan["current"] = label
                _scan["progress"] = {"done": done, "total": total}

        def on_plan(files: list[Path]) -> None:
            global _plan
            _plan = plan_units(files, options.sample_seconds)

        def on_phase(name: str) -> None:
            with _scan_lock:
                _scan["phase"] = name

        def on_frame() -> None:
            plan = _plan
            if plan is not None:
                plan.frame()

        def on_file(record: dict[str, object], seconds: float) -> None:
            plan = _plan
            if plan is not None:
                plan.finish_file(source / str(record.get("source") or ""), seconds)
            copied = [item for item in record.get("copiedTo") or [] if isinstance(item, str)]
            matches = [row for row in record.get("matches") or [] if isinstance(row, dict)]
            with _scan_lock:
                if copied:
                    _scan["matched"] = int(_scan["matched"]) + 1
                    _live_copies.update(copied)
                per_person = dict(_scan["perPerson"])
                for row in matches:
                    slug = str(row.get("slug") or "")
                    if slug in per_person:
                        per_person[slug] = int(per_person[slug]) + 1
                _scan["perPerson"] = per_person
                if copied and matches:
                    first = copied[0]
                    slug = str(matches[0].get("slug") or "")
                    _scan["lastMatch"] = {
                        "path": first,
                        "slug": slug,
                        "name": str(matches[0].get("name") or ""),
                        "thumb": results.thumb_url(dest, first, None, 320),
                    }

        hooks = ScanHooks(
            on_phase=on_phase,
            on_plan=on_plan,
            on_file=on_file,
            on_frame=on_frame,
            archive=archive_report,
        )
        with _engine_lock:
            report = run_scan(
                source,
                dest,
                recognizer,
                selected,
                options,
                progress,
                should_stop=_cancel.is_set,
                hooks=hooks,
            )
        summary = report.get("summary")
        _remember(dest, report)
        state = "cancelled" if report.get("complete") is False else "done"
        with _scan_lock:
            _scan_finished_mono = time.monotonic()
            _scan.update(
                state=state,
                summary=summary,
                report=str(dest / "rapor.json"),
                error=None,
                phase=None,
                finishedAt=datetime.now(UTC).isoformat(),
                stoppedAt=report.get("stoppedAt"),
                cancelRequested=False,
            )
        copies = [
            dest / str(item)
            for record in report.get("results") or []
            if isinstance(record, dict)
            for item in record.get("copiedTo") or []
            if isinstance(item, str)
        ]
        if copies:
            thumbs.warm_up(copies)
    except ScanCancelled:
        with _scan_lock:
            _scan_finished_mono = time.monotonic()
            _scan.update(
                state="cancelled",
                summary=None,
                error=None,
                phase=None,
                finishedAt=datetime.now(UTC).isoformat(),
                stoppedAt=datetime.now(UTC).isoformat(),
                cancelRequested=False,
            )
    except AppError as exc:
        with _scan_lock:
            _scan_finished_mono = time.monotonic()
            _scan.update(state="error", error=str(exc), phase=None, cancelRequested=False)
    except Exception:
        logging.exception("Scan failed")
        with _scan_lock:
            _scan_finished_mono = time.monotonic()
            _scan.update(
                state="error",
                error="Tarama beklenmedik bir hatayla durdu.",
                phase=None,
                cancelRequested=False,
            )


def _remember(dest: Path, report: dict[str, object]) -> None:
    """The recent-scan list is a convenience; a failure to write it must not fail the scan."""
    summary = report.get("summary")
    try:
        remember_scan(
            dest,
            summary if isinstance(summary, dict) else None,
            str(report.get("scannedAt") or ""),
            complete=report.get("complete") is not False,
        )
    except OSError:
        logging.exception("Could not update the recent-scan list")


def _saved_summary(path: Path) -> dict[str, object] | None:
    """Summary block of a report written earlier, for the panel's overview."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    summary = payload.get("summary") if isinstance(payload, dict) else None
    if not isinstance(summary, dict):
        return None
    when = payload.get("scannedAt") or payload.get("createdAt")
    return {
        **summary,
        "at": when if isinstance(when, str) else None,
        "complete": payload.get("complete") is not False,
        "output": str(path.parent),
    }


def _pool_worker(source: Path, dest: Path) -> None:
    def progress(done: int, total: int, label: str) -> None:
        with _pool_lock:
            _pool.update(done=done, total=total, current=label)

    try:
        report = build_pool(source, dest, progress)
        with _pool_lock:
            _pool.update(state="done", summary=report.get("summary"), error=None)
    except AppError as exc:
        with _pool_lock:
            _pool.update(state="error", error=str(exc))
    except Exception:
        logging.exception("Pool failed")
        with _pool_lock:
            _pool.update(state="error", error="Havuz beklenmedik bir hatayla durdu.")


def _get_engine() -> FaceEngine:
    global _engine
    with _engine_init_lock:
        if _engine is None:
            _engine = FaceEngine(models_dir(), log=logging.getLogger("recognize").info)
        return _engine


def _build_preview(payloads: list[tuple[str, bytes]]) -> tuple[str, list[dict[str, object]]]:
    with _engine_lock:
        engine = _get_engine()
        pending_images: list[_PendingImage] = []
        public: list[dict[str, object]] = []
        for image_index, (filename, data) in enumerate(payloads):
            faces, message = _faces_in_upload(engine, filename, data)
            pending_images.append(_PendingImage(filename, faces))
            public.append(
                {
                    "index": image_index,
                    "filename": filename,
                    "message": message,
                    "faces": [
                        {
                            "index": face.index,
                            "detScore": round(face.det_score, 3),
                            "crop": "data:image/jpeg;base64,"
                            + base64.b64encode(face.jpeg).decode("ascii"),
                        }
                        for face in faces
                    ],
                }
            )
    token = uuid.uuid4().hex
    with _pending_lock:
        _drop_expired()
        _pending[token] = _Pending(pending_images)
    return token, public


def _faces_in_upload(
    engine: FaceEngine,
    filename: str,
    data: bytes,
) -> tuple[list[_PendingFace], str | None]:
    try:
        image = resize_max_edge(decode_image(data), 2560)
    except AppError:
        return [], f"{filename} fotoğraf olarak okunamadı."
    detected = [face for face in engine.detect(image, 0.5) if min(face.width, face.height) >= 24]
    if not detected:
        return [], f"{filename} içinde yüz bulunamadı. Daha büyük ve net bir kare deneyin."
    stored: list[_PendingFace] = []
    for index, face in enumerate(detected):
        try:
            embedding = engine.embed(image, face)
            jpeg = engine.crop_jpeg(image, face)
        except (cv2.error, ValueError):
            continue
        stored.append(_PendingFace(index, face.score, embedding, jpeg))
    if not stored:
        return [], "Yüz bulundu ama vektör çıkarılamadı. Başka bir kare deneyin."
    if len(stored) == 1:
        return stored, None
    return stored, "Birden fazla yüz var. Kaydedilecek kişiyi seçin."


def _drop_expired() -> None:
    now = time.monotonic()
    stale = [key for key, value in _pending.items() if now - value.created > _PENDING_TTL]
    for key in stale:
        _pending.pop(key, None)


def _check_slug(slug: str) -> None:
    if not slug or not all(char.isalnum() or char == "-" for char in slug):
        raise HTTPException(status_code=400, detail="Geçersiz kişi.")


def _refuse_while_scanning() -> None:
    if _scan_running():
        raise HTTPException(status_code=409, detail="Tarama sürüyor. Bitince yeniden deneyin.")


def _check_token(token: str) -> None:
    if len(token) != 32 or any(char not in "0123456789abcdef" for char in token):
        raise HTTPException(status_code=400, detail="Oturum geçersiz.")


def _drop_camera() -> None:
    now = time.monotonic()
    stale = [key for key, value in _camera.items() if now - value.created > _CAMERA_TTL]
    for key in stale:
        _camera.pop(key, None)


def _live_session(token: str) -> CameraSession:
    """Caller holds ``_camera_lock``."""
    _drop_camera()
    pending = _camera.get(token)
    if pending is None:
        raise HTTPException(status_code=400, detail="Oturum kapandı. Kamerayı yeniden açın.")
    return pending.session


def _assess_locked(token: str, data: bytes) -> dict[str, object]:
    with _camera_lock:
        session = _live_session(token)
        if session.sweep is None:
            raise HTTPException(status_code=400, detail="Oturum tarama değil.")
        if not _engine_lock.acquire(blocking=False):
            if _scan_running():
                raise HTTPException(
                    status_code=409, detail="Tarama sürüyor. Bitince yeniden deneyin."
                )
            raise HTTPException(status_code=409, detail="Yüz motoru meşgul. Kare atlandı.")
        try:
            engine = _get_engine()
            image = resize_max_edge(decode_image(data), 960)
            report = record_sweep_frame(engine, session, image, time.monotonic())
        finally:
            _engine_lock.release()
        return _camera_payload(session, report)


def _mutate_camera(token: str, action: str) -> dict[str, object]:
    _check_token(token)
    with _camera_lock:
        session = _live_session(token)
        if session.guide is None:
            raise HTTPException(status_code=400, detail="Bu kayıt dönüş boyunca alınır.")
        try:
            if action == "arm":
                session.guide.arm_eyes()
                message = "Gözleriniz kapalıyken sabit durun."
            elif action == "skip":
                session.guide.skip()
                message = "Gözlerini kapat adımı atlandı."
            else:
                raise HTTPException(status_code=400, detail="Bilinmeyen adım.")
        except AppError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _camera_payload(session, _blank_report(message))


def _blank_report(message: str) -> FrameReport:
    return FrameReport(
        pose="unknown",
        yaw=None,
        pitch=None,
        face_px=0,
        score=0,
        captured=False,
        captured_label=None,
        crop=None,
        matches=False,
        message=message,
    )


def _camera_payload(session: CameraSession, report: FrameReport) -> dict[str, object]:
    guide = session.guide
    sweep = session.sweep
    step = None if guide is None else guide.current
    if sweep is not None:
        instruction = sweep.headline
    elif step is None:
        instruction = "Bakışlar tamam"
    else:
        instruction = step.instruction
    crop = None
    if report.crop:
        crop = "data:image/jpeg;base64," + base64.b64encode(report.crop).decode("ascii")
    return {
        "name": session.name,
        "pose": report.pose,
        "yaw": None if report.yaw is None else round(float(report.yaw), 4),
        "pitch": None if report.pitch is None else round(float(report.pitch), 4),
        "facePx": round(float(report.face_px), 1),
        "score": round(float(report.score), 3),
        "matches": report.matches,
        "streak": 0 if guide is None else guide.streak,
        "holdFrames": HOLD_FRAMES,
        "captured": report.captured,
        "crop": crop,
        "capturedLabel": report.captured_label,
        "stepId": None if step is None else step.id,
        "instruction": instruction,
        "requiredDone": report.ready if sweep is not None else bool(guide and guide.required_done),
        "ready": report.ready,
        "coverage": list(report.coverage),
        "finished": False if guide is None else guide.finished,
        "optional": bool(step is not None and not step.required),
        "eyesArmed": False if guide is None else guide.eyes_armed,
        "message": report.message,
        "steps": [] if guide is None else guide.step_states(),
        "hint": report.hint,
        "box": None if report.box is None else list(report.box),
        "gaze": None if report.gaze is None else [round(float(v), 3) for v in report.gaze],
        "nextRegion": None if sweep is None else sweep.next_region,
        "learned": sweep is not None and sweep.neutral_pitch is not None,
        "kept": len(session.samples),
        "total": SWEEP_TOTAL,
    }


__all__ = ["app", "load_people"]
