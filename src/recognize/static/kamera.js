const $ = (id) => document.getElementById(id);

const TICKS = 72;
// Ring angles: 0 is up, 90 is the user's right. The picture is mirrored, so
// the user's right is also the screen's right.
const REGION_ANGLE = { up: 0, right: 90, down: 180, left: 270 };
const QUALITY = new Set([
  "no_face",
  "many_faces",
  "small",
  "edge",
  "dark",
  "blur",
  "landmarks",
  "weak",
]);
// A warning shows only after it has held this long; one bad frame is not news.
const WARN_AFTER_MS = 600;
// The instruction does not change more often than this, so it can be read.
const INSTRUCTION_HOLD_MS = 700;
const SEND_GAP_MS = 70;
const MAX_EDGE = 960;
const FACE_SHARE = 0.56;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

const cam = $("cam");
const video = $("video");
const viewfinder = $("viewfinder");
const feed = $("feed");
const nudge = $("nudge");
const sender = document.createElement("canvas");

const ticks = [];
const regionTicks = {};
const state = {
  token: "",
  name: "",
  running: false,
  busy: false,
  loop: 0,
  nextSend: 0,
  finished: false,
  saveChain: Promise.resolve(),
  lit: new Set(),
  gazeTarget: null,
  gaze: [0, 0],
  box: null,
  view: { cx: 0.5, cy: 0.5, zoom: 1 },
  feedSize: 0,
  warnHint: "",
  warnSince: 0,
  instruction: "",
  instructionAt: 0,
  nextRegion: null,
  learned: false,
  lastTime: 0,
  crops: [],
};

video.muted = true;
video.playsInline = true;

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function api(url, options) {
  const response = await fetch(url, options);
  const text = await response.text();
  let body = {};
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = {};
    }
  }
  if (!response.ok) {
    const detail = body.detail;
    throw new ApiError(
      typeof detail === "string" ? detail : "İstek tamamlanamadı.",
      response.status,
    );
  }
  return body;
}

function describeCamera(error) {
  const name = error && error.name;
  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    return "Kamera izni verilmedi. Tarayıcı ayarlarından bu site için kameraya izin verin.";
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "Bu bilgisayarda kullanılabilir bir kamera bulunamadı.";
  }
  if (name === "NotReadableError" || name === "TrackStartError" || name === "AbortError") {
    return "Kamera açılamadı. Başka bir uygulama kullanıyor olabilir.";
  }
  if (name === "SecurityError") {
    return "Kamera yalnızca bu bilgisayardaki panelde açılır.";
  }
  if (error instanceof Error && error.message) return error.message;
  return "Kamera açılamadı.";
}

function localHost() {
  const host = location.hostname;
  return host === "127.0.0.1" || host === "localhost" || host === "::1" || host === "[::1]";
}

function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

function angleGap(a, b) {
  const d = (((a - b) % 360) + 360) % 360;
  return Math.min(d, 360 - d);
}

/* ---------- Ring ---------- */

function buildRing() {
  const ring = $("ring");
  const step = 360 / TICKS;
  const fragment = document.createDocumentFragment();
  for (let index = 0; index < TICKS; index += 1) {
    // Half a step off so no tick sits on the border between two directions.
    const angle = index * step + step / 2;
    const element = document.createElement("span");
    element.className = "tick";
    element.style.setProperty("--a", `${angle}deg`);
    element.addEventListener("animationend", () => element.classList.remove("pop"));
    fragment.append(element);
    ticks.push({ element, angle, scale: 1, live: false });
  }
  ring.append(fragment);
  for (const [region, center] of Object.entries(REGION_ANGLE)) {
    regionTicks[region] = ticks
      .map((tick, index) => ({ index, gap: angleGap(tick.angle, center) }))
      .filter((item) => item.gap < 45)
      .sort((a, b) => a.gap - b.gap)
      .map((item) => item.index);
  }
}

function resetRing() {
  state.lit.clear();
  for (const tick of ticks) {
    tick.element.classList.remove("lit", "pop", "live");
    tick.element.style.setProperty("--s", "1");
    tick.scale = 1;
    tick.live = false;
  }
  viewfinder.classList.remove("center-done");
}

function renderCoverage(coverage, nextRegion, animate) {
  if (!Array.isArray(coverage)) return;
  for (const region of coverage) {
    if (!region || !/^[a-z]+$/.test(region.id)) continue;
    const count = Number(region.count) || 0;
    const need = Number(region.need) || 1;
    const done = Boolean(region.done);
    const chip = document.querySelector(`#coverage [data-region="${region.id}"]`);
    if (chip) {
      const tally = chip.querySelector(".chip-count");
      if (tally) tally.textContent = `${Math.min(count, need)}/${need}`;
      chip.classList.toggle("is-done", done);
      chip.classList.toggle("is-next", !done && region.id === nextRegion);
    }
    if (region.id === "center") {
      viewfinder.classList.toggle("center-done", done);
      continue;
    }
    const order = regionTicks[region.id];
    if (!order) continue;
    const want = Math.round(order.length * Math.min(1, count / need));
    for (let k = 0; k < want; k += 1) {
      const index = order[k];
      if (state.lit.has(index)) continue;
      state.lit.add(index);
      const element = ticks[index].element;
      element.classList.add("lit");
      if (animate && !reduceMotion.matches) element.classList.add("pop");
    }
  }
}

function lightEverything() {
  for (let index = 0; index < ticks.length; index += 1) {
    state.lit.add(index);
    ticks[index].element.classList.add("lit");
  }
}

/* ---------- Per-frame motion ---------- */

function paintTicks(dt) {
  const target = state.gazeTarget || [0, 0];
  const k = 1 - Math.exp(-dt / 0.09);
  state.gaze[0] += (target[0] - state.gaze[0]) * k;
  state.gaze[1] += (target[1] - state.gaze[1]) * k;
  const [gx, gy] = state.gaze;
  const reach = Math.min(1, Math.hypot(gx, gy));
  const toward = (Math.atan2(gx, gy) * 180) / Math.PI;
  for (const tick of ticks) {
    const gap = (angleGap(tick.angle, toward) * Math.PI) / 180;
    const near = gap < Math.PI / 2 ? Math.cos(gap) ** 8 : 0;
    const pull = reach * near;
    const scale = 1 + 0.5 * pull;
    if (Math.abs(scale - tick.scale) > 0.004) {
      tick.scale = scale;
      tick.element.style.setProperty("--s", scale.toFixed(3));
    }
    const live = pull > 0.35;
    if (live !== tick.live) {
      tick.live = live;
      tick.element.classList.toggle("live", live);
    }
  }
  const resting = reach < 0.3;
  const region = state.nextRegion;
  const show =
    state.running &&
    state.learned &&
    resting &&
    !state.warnHint &&
    Object.hasOwn(REGION_ANGLE, region);
  if (show) nudge.style.setProperty("--n", `${REGION_ANGLE[region]}deg`);
  nudge.classList.toggle("on", show);
}

function measureFeed() {
  state.feedSize = feed.clientWidth;
  if (!video.videoWidth || !state.feedSize) return;
  const aspect = video.videoWidth / video.videoHeight;
  const size = state.feedSize;
  const width = aspect >= 1 ? size * aspect : size;
  const height = aspect >= 1 ? size : size / aspect;
  video.style.width = `${width}px`;
  video.style.height = `${height}px`;
}

// Keeps the face centred and roughly the same size inside the circle.
function frameVideo(dt) {
  const size = state.feedSize;
  if (!video.videoWidth || !size) return;
  const baseWidth = parseFloat(video.style.width) || size;
  const baseHeight = parseFloat(video.style.height) || size;
  let cx = 0.5;
  let cy = 0.5;
  let zoom = 1;
  if (state.box && !reduceMotion.matches) {
    const [bx, by, bw, bh] = state.box;
    // The box comes from the mirrored frame; the video element is the raw one.
    cx = 1 - (bx + bw / 2);
    cy = by + bh / 2;
    zoom = clamp((FACE_SHARE * size) / Math.max(1, bh * baseHeight), 1, 1.8);
  }
  const k = 1 - Math.exp(-dt / 0.35);
  const view = state.view;
  view.cx += (cx - view.cx) * k;
  view.cy += (cy - view.cy) * k;
  view.zoom += (zoom - view.zoom) * k;
  const width = baseWidth * view.zoom;
  const height = baseHeight * view.zoom;
  const left = clamp(size / 2 - view.cx * width, size - width, 0);
  const top = clamp(size / 2 - view.cy * height, size - height, 0);
  video.style.transform = `translate3d(${left.toFixed(1)}px, ${top.toFixed(1)}px, 0) scale(${view.zoom.toFixed(4)})`;
}

function tick(now) {
  if (!state.running) return;
  const dt = state.lastTime ? Math.min(0.1, (now - state.lastTime) / 1000) : 0.016;
  state.lastTime = now;
  paintTicks(dt);
  frameVideo(dt);
  if (!state.busy && !document.hidden && now >= state.nextSend) {
    state.nextSend = now + SEND_GAP_MS;
    sendFrame();
  }
  state.loop = requestAnimationFrame(tick);
}

/* ---------- Copy ---------- */

function setPhase(phase) {
  cam.dataset.phase = phase;
  $("live-actions").hidden = phase !== "live";
  $("done").hidden = phase !== "done";
}

function setViewState(name) {
  if (viewfinder.dataset.state !== name) viewfinder.dataset.state = name;
}

function setInstruction(text, force) {
  if (!text || text === state.instruction) return;
  const now = performance.now();
  if (!force && now - state.instructionAt < INSTRUCTION_HOLD_MS) return;
  state.instruction = text;
  state.instructionAt = now;
  $("instruction").textContent = text;
}

function setDetail(text, kind) {
  const node = $("detail");
  if (node.textContent !== text) node.textContent = text;
  if (kind) node.dataset.kind = kind;
  else delete node.dataset.kind;
}

function applyFrame(data) {
  const now = performance.now();
  state.learned = Boolean(data.learned);
  state.nextRegion = typeof data.nextRegion === "string" ? data.nextRegion : null;
  if (Array.isArray(data.box) && data.box.length === 4) state.box = data.box.map(Number);
  state.gazeTarget = Array.isArray(data.gaze) ? data.gaze.map(Number) : null;
  renderCoverage(data.coverage, state.nextRegion, true);
  if (data.captured && typeof data.crop === "string") state.crops.push(data.crop);

  if (data.ready && !state.finished) {
    state.finished = true;
    state.warnHint = "";
    lightEverything();
    setViewState("done");
    setInstruction("Yüzün kaydediliyor…", true);
    setDetail(`${data.kept} açı alındı.`);
    scheduleSave();
    return;
  }
  if (state.finished) return;

  if (QUALITY.has(data.hint)) {
    if (state.warnHint !== data.hint) {
      state.warnHint = data.hint;
      state.warnSince = now;
    }
    if (now - state.warnSince >= WARN_AFTER_MS) {
      state.gazeTarget = null;
      setViewState("warn");
      setInstruction(data.message, true);
      setDetail(state.learned ? "Açılar korunuyor; sorun geçince devam eder." : "");
    }
    return;
  }
  state.warnHint = "";
  if (!state.learned) {
    setViewState("learning");
    setInstruction("Kameraya düz bak", viewfinder.dataset.state === "warn");
    setDetail("Yüzünü öğreniyorum…");
    return;
  }
  const wasWarning = viewfinder.dataset.state === "warn";
  setViewState("tracking");
  setInstruction(data.instruction, wasWarning);
  const total = Number(data.total) || 0;
  const kept = Math.min(Number(data.kept) || 0, total || Infinity);
  if (data.captured && data.capturedLabel) {
    setDetail(`${data.capturedLabel} ${kept}/${total}`, "ok");
  } else if (!$("detail").dataset.kind || now - state.instructionAt > 1200) {
    setDetail(`${kept}/${total} açı alındı. Başını yavaş çevir.`);
  }
}

/* ---------- Camera ---------- */

function sendFrame() {
  if (!state.running || !state.token || !video.videoWidth) return;
  state.busy = true;
  let width = video.videoWidth;
  let height = video.videoHeight;
  const edge = Math.max(width, height);
  if (edge > MAX_EDGE) {
    const scale = MAX_EDGE / edge;
    width = Math.round(width * scale);
    height = Math.round(height * scale);
  }
  if (sender.width !== width || sender.height !== height) {
    sender.width = width;
    sender.height = height;
  }
  // The server reads a mirrored frame: image-left is the user's left.
  const context = sender.getContext("2d", { alpha: false });
  context.save();
  context.translate(width, 0);
  context.scale(-1, 1);
  context.drawImage(video, 0, 0, width, height);
  context.restore();
  const token = state.token;
  sender.toBlob(
    async (blob) => {
      if (!blob || !state.running || token !== state.token) {
        state.busy = false;
        return;
      }
      const body = new FormData();
      body.append("token", token);
      body.append("frame", blob, "kare.jpg");
      try {
        const data = await api("/api/kamera/kare", { method: "POST", body });
        if (token === state.token && state.running) applyFrame(data);
      } catch (error) {
        handleFrameError(error);
      } finally {
        state.busy = false;
      }
    },
    "image/jpeg",
    0.85,
  );
}

function handleFrameError(error) {
  if (error instanceof ApiError && error.status === 409 && /meşgul/.test(error.message)) {
    state.nextSend = performance.now() + 250;
    return;
  }
  if (error instanceof ApiError && error.status === 400 && /Oturum/.test(error.message)) {
    stopCamera();
    setPhase("idle");
    setViewState("idle");
    setInstruction("Oturum kapandı", true);
    setDetail(error.message, "bad");
    return;
  }
  setDetail(error.message || "Kare gönderilemedi.", "bad");
  state.nextSend = performance.now() + 800;
}

function stopCamera() {
  state.running = false;
  cancelAnimationFrame(state.loop);
  state.lastTime = 0;
  const stream = video.srcObject;
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
  }
  video.srcObject = null;
  viewfinder.classList.remove("streaming");
  nudge.classList.remove("on");
}

function scheduleSave() {
  state.saveChain = state.saveChain.then(saveOnce).catch((error) => {
    stopCamera();
    setPhase("done");
    setInstruction("Kayıt yazılamadı", true);
    setDetail(error.message, "bad");
    $("retry").hidden = false;
  });
}

async function saveOnce() {
  const data = await api("/api/kamera/kaydet", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token: state.token }),
  });
  stopCamera();
  setPhase("done");
  setInstruction("Yüzün kaydedildi", true);
  setDetail("");
  const warning = data.warning ? ` ${data.warning}` : "";
  const saved = $("saved");
  saved.textContent = `${data.name} · toplam ${data.references} referans.${warning}`;
  saved.dataset.kind = data.warning ? "warn" : "ok";
  $("retry").hidden = true;
  renderCrops();
}

function renderCrops() {
  const box = $("crops");
  box.replaceChildren();
  for (const [index, src] of state.crops.entries()) {
    if (!src.startsWith("data:image/jpeg;base64,")) continue;
    const img = document.createElement("img");
    img.src = src;
    img.alt = `Alınan açı ${index + 1}`;
    img.decoding = "async";
    box.append(img);
  }
}

function resetSession() {
  state.token = "";
  state.finished = false;
  state.crops = [];
  state.box = null;
  state.gazeTarget = null;
  state.gaze = [0, 0];
  state.view = { cx: 0.5, cy: 0.5, zoom: 1 };
  state.warnHint = "";
  state.learned = false;
  state.nextRegion = null;
  state.instruction = "";
  state.instructionAt = 0;
  resetRing();
  $("crops").replaceChildren();
  $("saved").textContent = "";
}

async function openCamera() {
  if (state.running) return;
  if (!localHost()) {
    setDetail("Kamera yalnızca bu bilgisayardaki panelde açılır.", "bad");
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setDetail("Bu tarayıcı kamerayı açamıyor. Paneli 127.0.0.1 adresinden açın.", "bad");
    return;
  }
  const name = $("camera-name").value.trim();
  if (!name) {
    setDetail("Ad soyad gerekli.", "bad");
    return;
  }
  const button = $("open-camera");
  button.disabled = true;
  resetSession();
  setInstruction("Kamera açılıyor…", true);
  setDetail("");
  try {
    const started = await api("/api/kamera/baslat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 },
      },
    });
    video.srcObject = stream;
    await video.play();
    state.token = started.token;
    state.name = started.name || name;
    $("who").textContent = state.name;
    $("camera-name").readOnly = true;
    renderCoverage(started.coverage, null, false);
    measureFeed();
    viewfinder.classList.add("streaming");
    setPhase("live");
    setViewState("learning");
    setInstruction("Kameraya düz bak", true);
    setDetail("Yüzünü öğreniyorum…");
    state.running = true;
    state.nextSend = 0;
    state.loop = requestAnimationFrame(tick);
  } catch (error) {
    stopCamera();
    setPhase("idle");
    setViewState("idle");
    setInstruction("Kamera açılamadı", true);
    setDetail(describeCamera(error), "bad");
  } finally {
    button.disabled = false;
  }
}

function cancelCamera() {
  stopCamera();
  setPhase("idle");
  setViewState("idle");
  $("camera-name").readOnly = false;
  setInstruction("Kamera kapandı", true);
  setDetail("Çember dolmadan kayıt yazılmaz.");
}

const known = [];

async function loadKnownNames() {
  try {
    const data = await api("/api/kisiler");
    for (const person of data.people || []) {
      known.push({ name: String(person.name || ""), references: Number(person.references) || 0 });
    }
  } catch {
    // The hint is a convenience; the camera works without it.
  }
}

function paintNameHint() {
  const hint = $("name-hint");
  const typed = $("camera-name").value.trim().toLocaleLowerCase("tr-TR");
  const match = typed && known.find((p) => p.name.toLocaleLowerCase("tr-TR") === typed);
  if (match) {
    hint.textContent = `Bu ad kayıtlı (${match.references} açı). Yeni açılar bu kişiye eklenir.`;
    hint.hidden = false;
  } else {
    hint.hidden = true;
  }
}

function boot() {
  buildRing();
  const params = new URLSearchParams(location.search);
  const given = params.get("ad");
  const nameField = $("camera-name");
  if (given && given.trim()) nameField.value = given.trim();
  if (params.has("yeni")) {
    // "Kişi ekle" from the archive: an empty name, so a new person never lands on someone else.
    nameField.value = "";
    nameField.focus();
  }
  nameField.addEventListener("input", paintNameHint);
  loadKnownNames().then(paintNameHint);
  $("camera-form").addEventListener("submit", (event) => {
    event.preventDefault();
    openCamera();
  });
  $("stop-camera").addEventListener("click", cancelCamera);
  $("again").addEventListener("click", () => {
    setPhase("idle");
    openCamera();
  });
  $("retry").addEventListener("click", () => {
    $("retry").hidden = true;
    setInstruction("Yüzün kaydediliyor…", true);
    scheduleSave();
  });
  video.addEventListener("loadedmetadata", measureFeed);
  if (window.ResizeObserver) new ResizeObserver(measureFeed).observe(feed);
  window.addEventListener("pagehide", stopCamera);
}

boot();
