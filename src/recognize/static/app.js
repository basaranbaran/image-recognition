const $ = (id) => document.getElementById(id);

/* ---------- Formatting (tr-TR everywhere) ---------- */

const LOCALE = "tr-TR";
const intFmt = new Intl.NumberFormat(LOCALE);
const twoFmt = new Intl.NumberFormat(LOCALE, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const oneFmt = new Intl.NumberFormat(LOCALE, { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const pctFmt = new Intl.NumberFormat(LOCALE, { style: "percent", maximumFractionDigits: 0 });
const dateFmt = new Intl.DateTimeFormat(LOCALE, { dateStyle: "long", timeStyle: "short" });
const timeFmt = new Intl.DateTimeFormat(LOCALE, { timeStyle: "short" });
const relFmt = new Intl.RelativeTimeFormat(LOCALE, { numeric: "auto" });

const fmt = {
  int: (n) => intFmt.format(Number(n) || 0),
  two: (n) => twoFmt.format(Number(n) || 0),
  one: (n) => oneFmt.format(Number(n) || 0),
  pct: (ratio) => pctFmt.format(Math.max(0, Math.min(1, Number(ratio) || 0))),
  date(iso) {
    const date = new Date(iso);
    return Number.isNaN(date.getTime()) ? "" : dateFmt.format(date);
  },
  rel(iso) {
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "";
    const seconds = (date.getTime() - Date.now()) / 1000;
    const abs = Math.abs(seconds);
    if (abs < 60) return "az önce";
    if (abs < 3600) return relFmt.format(Math.round(seconds / 60), "minute");
    if (abs < 86400 && date.getDate() === new Date().getDate()) {
      return `bugün ${timeFmt.format(date)}`;
    }
    if (abs < 86400 * 2) return `${relFmt.format(Math.round(seconds / 86400), "day")} ${timeFmt.format(date)}`;
    return dateFmt.format(date);
  },
  duration(seconds) {
    const total = Math.max(0, Math.round(Number(seconds) || 0));
    if (total < 60) return `yaklaşık ${Math.max(5, Math.ceil(total / 5) * 5)} sn`;
    const minutes = Math.round(total / 60);
    if (minutes < 60) return `yaklaşık ${minutes} dk`;
    const hours = Math.floor(minutes / 60);
    const rest = minutes % 60;
    return rest ? `yaklaşık ${hours} sa ${rest} dk` : `yaklaşık ${hours} sa`;
  },
  bytes(n) {
    const value = Number(n) || 0;
    if (value < 1024 * 1024) return `${fmt.int(Math.round(value / 1024))} KB`;
    return `${fmt.one(value / (1024 * 1024))} MB`;
  },
};

/* ---------- API helper: every failure reaches the page in Turkish ---------- */

const NETWORK_MESSAGE = "Panele ulaşılamıyor. Terminalde panel açık mı? (uv run recognize panel)";

class ApiError extends Error {
  constructor(message, status, body) {
    super(message);
    this.status = status;
    this.body = body || {};
  }
}

async function api(url, options = {}) {
  let response;
  try {
    response = await fetch(url, { cache: "no-store", ...options });
  } catch {
    throw new ApiError(NETWORK_MESSAGE, 0);
  }
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
    const message = typeof detail === "string" ? detail : "İstek tamamlanamadı.";
    throw new ApiError(message, response.status, body);
  }
  return body;
}

function postJson(url, payload) {
  return api(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
}

/* ---------- DOM helpers ---------- */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function icon(name, className) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", className ? `ic ${className}` : "ic");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#${name}`);
  svg.append(use);
  return svg;
}

function button(label, className, onClick, iconName) {
  const node = el("button", className ? `btn ${className}` : "btn");
  node.type = "button";
  if (iconName) node.append(icon(iconName));
  node.append(document.createTextNode(label));
  if (onClick) node.addEventListener("click", onClick);
  return node;
}

function monogram(name, className) {
  const parts = String(name || "").trim().split(/\s+/).filter(Boolean);
  const letters = parts.slice(0, 2).map((part) => part[0].toLocaleUpperCase(LOCALE)).join("");
  const node = el("span", className ? `mono ${className}` : "mono", letters || "?");
  node.setAttribute("aria-hidden", "true");
  return node;
}

function avatar(person, className) {
  if (person && person.cover) {
    const img = el("img", className || "avatar");
    img.src = person.cover;
    img.alt = "";
    img.decoding = "async";
    img.loading = "lazy";
    return img;
  }
  return monogram(person ? person.name : "", className || "avatar");
}

function setText(id, text) {
  const node = $(id);
  if (node && node.textContent !== text) node.textContent = text;
}

function setKind(node, kind) {
  if (kind) node.dataset.kind = kind;
  else delete node.dataset.kind;
}

function announce(text) {
  const node = $("announce");
  node.textContent = "";
  window.setTimeout(() => {
    node.textContent = text;
  }, 50);
}

function count(value) {
  return Number(value) || 0;
}

function plural(n, word) {
  return `${fmt.int(n)} ${word}`;
}

/* ---------- Toasts ---------- */

function toastHost() {
  // A modal dialog makes everything outside it inert: a toast behind the person sheet could be
  // neither read nor undone. The toast area moves into the topmost open modal instead.
  const modals = [...document.querySelectorAll("dialog[open]")].filter((d) => d.matches(":modal"));
  return modals.at(-1) || document.body;
}

document.addEventListener(
  "close",
  (event) => {
    const root = $("toasts");
    if (root && root.parentElement === event.target) document.body.append(root);
  },
  true,
);

function toast(text, { action, onAction, kind, duration = 8000 } = {}) {
  const root = $("toasts");
  const host = toastHost();
  if (root.parentElement !== host) host.append(root);
  const node = el("div", kind ? `toast ${kind}` : "toast");
  node.append(el("p", "", text));
  let timer = 0;
  const close = () => {
    window.clearTimeout(timer);
    node.remove();
  };
  if (action && onAction) {
    node.append(
      button(action, "small", async () => {
        close();
        try {
          await onAction();
        } catch (error) {
          toast(error.message, { kind: "bad" });
        }
      }),
    );
  }
  const dismiss = el("button", "btn icon small");
  dismiss.type = "button";
  dismiss.setAttribute("aria-label", "Kapat");
  dismiss.append(icon("i-x"));
  dismiss.addEventListener("click", close);
  node.append(dismiss);
  root.append(node);
  timer = window.setTimeout(close, duration);
  while (root.children.length > 3) root.firstElementChild.remove();
  return close;
}

/* ---------- State ---------- */

const state = {
  settings: null,
  people: [],
  problems: [],
  results: null,
  resultsError: null,
  scan: null,
  pool: null,
  platform: "mac",
  skipped: new Set(),
  online: true,
  booted: false,
  wasRunning: false,
  lastMilestone: -1,
  lastPhase: "",
  pollTimer: 0,
  poolTimer: 0,
  pollDelay: 500,
  retryTimer: 0,
  retryDelay: 2000,
  sheet: { slug: null, filter: "all", data: null },
  enroll: { token: "", images: [], picks: new Map() },
};

const FOLDER_LABEL = {
  mac: "Finder'da göster",
  windows: "Gezgin'de göster",
  linux: "Dosya yöneticisinde göster",
};
const PICK_OPENED = {
  mac: "Finder penceresi açıldı. Klasörü seçin.",
  windows: "Gezgin penceresi açıldı. Klasörü seçin.",
  linux: "Klasör penceresi açıldı.",
};
const REVIEW_LABEL = {
  smallFace: "küçük yüz",
  learned: "ikinci bakış",
  clothingAssisted: "kıyafet desteği",
  nearThreshold: "eşiğe yakın",
};
const PHASE_LABEL = {
  start: "Başlıyor…",
  models: "Yüz modeli hazırlanıyor (ilk çalıştırmada indirilir)…",
  hash: "Aynı dosyalar aranıyor…",
  files: "Yüzler aranıyor",
  secondLook: "İkinci bakış: etkinlikteki net yüzlerle fotoğraflara bir kez daha bakılıyor…",
  writing: "Rapor yazılıyor…",
};

function folderLabel() {
  return FOLDER_LABEL[state.platform] || FOLDER_LABEL.linux;
}

/* ---------- Connection ---------- */

function setConnection(mode) {
  const pill = $("conn");
  pill.dataset.state = mode;
  const use = pill.querySelector("use");
  use.setAttribute("href", mode === "ok" ? "#i-lock" : "#i-off");
  setText(
    "conn-text",
    mode === "ok" ? "Yerel" : mode === "retry" ? "Yeniden bağlanıyor…" : "Bağlantı yok",
  );
  pill.title =
    mode === "ok" ? "Fotoğraflar bu bilgisayardan çıkmaz" : NETWORK_MESSAGE;
}

function showBanner(text) {
  $("banner-text").textContent = text;
  $("banner").hidden = false;
}

function hideBanner() {
  $("banner").hidden = true;
}

/* ---------- Boot ---------- */

async function boot() {
  bindOnce();
  buildHeroRing();
  await loadEverything();
}

async function loadEverything() {
  window.clearTimeout(state.retryTimer);
  const [settings, people, scan, pool, results] = await Promise.allSettled([
    api("/api/ayarlar"),
    api("/api/kisiler"),
    api("/api/tara"),
    api("/api/havuz"),
    api("/api/sonuclar"),
  ]);
  const network = [settings, people, scan, pool].every(
    (item) => item.status === "rejected" && item.reason instanceof ApiError && item.reason.status === 0,
  );
  if (network) {
    setConnection("off");
    showBanner(NETWORK_MESSAGE);
    $("page-head").hidden = false;
    setText("status-line", "Panele bağlanılamadı; yeniden deneniyor…");
    state.retryDelay = Math.min(state.retryDelay * 2, 16000);
    state.retryTimer = window.setTimeout(loadEverything, state.retryDelay);
    return;
  }
  state.retryDelay = 2000;
  setConnection("ok");
  hideBanner();
  if (settings.status === "fulfilled") applySettings(settings.value);
  if (people.status === "fulfilled") applyPeople(people.value);
  else showProblems([{ folder: "", detail: people.reason.message }]);
  if (results.status === "fulfilled") {
    state.results = results.value;
    state.resultsError = null;
  } else {
    state.results = null;
    state.resultsError = results.reason;
  }
  state.booted = true;
  renderResults();
  renderPeople();
  renderWho();
  renderOverview();
  if (pool.status === "fulfilled") {
    renderPool(pool.value);
    if (pool.value.state === "running") pollPool();
  }
  if (scan.status === "fulfilled") {
    state.scan = scan.value;
    if (scan.value.state === "running") {
      state.wasRunning = true;
      renderScan(scan.value);
      poll();
    } else if (scan.value.state !== "idle") {
      // The panel remembers the last job of this session, so a reload keeps its summary.
      renderScan(scan.value);
    }
  }
  updatePreflight();
}

function applySettings(settings) {
  state.settings = settings;
  state.platform = settings.platform || "linux";
  document.documentElement.dataset.platform = state.platform;
  if (settings.source && !$("source").value) $("source").value = settings.source;
  if (!$("destination").value) $("destination").value = settings.output || "";
  $("threshold").value = String(settings.threshold ?? 0.5);
  $("interval").value = String(settings.interval ?? 0.5);
  paintThreshold();
  setText("guidance", settings.angleGuidance || $("guidance").textContent);
  setText("one-photo", settings.onePhotoNote || $("one-photo").textContent);
  setText("clothing-note", settings.clothingNote || $("clothing-note").textContent);
  setText("threshold-hint", settings.thresholdHint || $("threshold-hint").textContent);
  setText("video-hint", settings.videoHint || $("video-hint").textContent);
  state.pool = settings.poolReady ? settings.poolSummary : null;
  poolTargets.source = settings.poolSource || "";
  poolTargets.target = settings.pool || "";
  poolTargets.defaultSource = poolTargets.source;
  poolTargets.defaultTarget = poolTargets.target;
  showFolder("source");
  showFolder("destination");
  if (settings.poolReady && settings.poolSummary) {
    $("pool-start").textContent = "Yenile";
    setPoolStatus(poolLine(settings.poolSummary), "ok");
  }
  for (const node of document.querySelectorAll("[data-folder-label]")) {
    node.textContent = folderLabel();
  }
  $("sheet-open").textContent = folderLabel();
}

function applyPeople(data) {
  state.people = Array.isArray(data.people) ? data.people : [];
  state.problems = Array.isArray(data.problems) ? data.problems : [];
  showProblems(state.problems);
}

function showProblems(problems) {
  const box = $("people-problems");
  box.replaceChildren();
  if (!problems.length) {
    box.hidden = true;
    return;
  }
  box.append(icon("i-warn"));
  const list = el("div");
  for (const problem of problems) list.append(el("p", "", problem.detail));
  box.append(list);
  box.hidden = false;
}

async function reloadPeople() {
  try {
    applyPeople(await api("/api/kisiler"));
  } catch (error) {
    toast(error.message, { kind: "bad" });
    return;
  }
  renderPeople();
  renderWho();
  renderOverview();
  updatePreflight();
}

function refreshResults() {
  return reloadResults(state.results ? state.results.output : undefined);
}

async function reloadResults(output) {
  try {
    state.results = await api(
      output ? `/api/sonuclar?${new URLSearchParams({ output })}` : "/api/sonuclar",
    );
    state.resultsError = null;
  } catch (error) {
    state.results = null;
    state.resultsError = error;
  }
  renderResults();
  renderOverview();
}

/* ---------- Overview: hero or page head ---------- */

function buildHeroRing() {
  const ring = $("hero-ring");
  const total = 48;
  const fragment = document.createDocumentFragment();
  for (let index = 0; index < total; index += 1) {
    const tick = el("span", index % 3 === 0 ? "tick lit" : "tick");
    tick.style.setProperty("--a", `${(index * 360) / total + 360 / total / 2}deg`);
    fragment.append(tick);
  }
  ring.prepend(fragment);
}

function paintStageRing(ratio) {
  const ring = $("stage-ring");
  const total = 48;
  if (!ring.querySelector(".tick")) {
    const fragment = document.createDocumentFragment();
    for (let index = 0; index < total; index += 1) {
      const tick = el("span", "tick");
      tick.style.setProperty("--a", `${(index * 360) / total + 360 / total / 2}deg`);
      fragment.append(tick);
    }
    ring.prepend(fragment);
  }
  const lit = Math.round(total * Math.max(0, Math.min(1, ratio)));
  ring.querySelectorAll(".tick").forEach((tick, index) => tick.classList.toggle("lit", index < lit));
}

function renderOverview() {
  const firstRun = state.booted && state.people.length === 0 && !state.results;
  $("hero").hidden = !firstRun;
  $("page-head").hidden = firstRun;
  const parts = [];
  if (state.people.length) parts.push(plural(state.people.length, "kişi"));
  if (state.pool) parts.push(`${fmt.int(count(state.pool.unique))} tekil dosya`);
  if (state.results && state.results.scannedAt) {
    parts.push(`son ayırma ${fmt.rel(state.results.scannedAt)}`);
  } else if (state.people.length) {
    parts.push("henüz ayırma yapılmadı");
  }
  setText("status-line", parts.join(" · "));
  $("page").dataset.ready = "1";
}

/* ---------- Results ---------- */

function renderResults() {
  const section = $("sonuclar");
  const data = state.results;
  const list = $("results-people");
  list.replaceChildren();
  $("results-actions").replaceChildren();
  $("results-notice").hidden = true;
  $("unmatched").hidden = true;
  if (!data) {
    const shouldShow = state.people.length > 0;
    section.hidden = !shouldShow;
    if (!shouldShow) return;
    const empty = el("div", "empty-results");
    const error = state.resultsError;
    if (error && error.status && error.status !== 404) {
      empty.append(el("p", "t-callout", error.message));
    } else {
      empty.append(el("p", "t-headline", "Henüz ayırma yapılmadı"));
      empty.append(
        el("p", "t-footnote", "Aşağıdan kaynak ve çıktı klasörünü seçip ayırmayı başlatın."),
      );
    }
    list.append(empty);
    setText("results-meta", "");
    return;
  }
  section.hidden = false;
  const summary = data.summary || {};
  const peopleWithFiles = data.people.filter((row) => row.counts.files > 0);
  const matched = count(summary.matchedFiles);
  const files = count(summary.files);
  paintStageRing(files ? matched / files : 0);
  setText("stage-number", fmt.int(matched));
  setText("stage-label", matched === 1 ? "dosya ayrıldı" : "dosya ayrıldı");
  setText("results-eyebrow", `Son ayırma · ${fmt.date(data.scannedAt)}`);
  let title;
  if (!matched) title = `${plural(files, "dosya")} tarandı, kayıtlı kimse bulunamadı`;
  else if (peopleWithFiles.length === 1) {
    title = `${plural(files, "dosya")} tarandı, ${peopleWithFiles[0].name} ${plural(matched, "dosyada")} bulundu`;
  } else {
    title = `${plural(files, "dosya")} tarandı, ${fmt.int(peopleWithFiles.length)} kişinin klasörüne kopyalandı`;
  }
  setText("results-title", title);
  const meta = [];
  if (summary.videos) meta.push(`${fmt.int(summary.images)} fotoğraf · ${fmt.int(summary.videos)} video`);
  if (summary.rejected) meta.push(`${fmt.int(summary.rejected)} ayıklandı`);
  const review = data.people.reduce((sum, row) => sum + count(row.counts.review), 0);
  if (review) meta.push(`${fmt.int(review)} kontrol edilmeli`);
  if (data.move) meta.push("kaynaktan taşındı");
  meta.push(`eşik ${fmt.two(data.threshold)}`);
  setText("results-meta", meta.join(" · "));
  const stagePeople = $("stage-people");
  stagePeople.replaceChildren();
  for (const person of peopleWithFiles) {
    const chip = el("span", "count-chip");
    chip.append(avatar(person, "mono"), el("span", "", person.name), el("b", "", fmt.int(person.counts.files)));
    stagePeople.append(chip);
  }
  $("results-actions").append(
    button(folderLabel(), "small", () => openInFileManager({ output: data.output }), "i-open"),
  );

  const notice = $("results-notice");
  notice.className = "notice";
  if (data.scanRunning) {
    showNotice(notice, "i-info", "Bu klasöre şu an ayırma yapılıyor; liste bir önceki durumu gösteriyor.");
  } else if (!data.complete) {
    const progress = data.progress || {};
    notice.classList.add("warn");
    showNotice(
      notice,
      "i-warn",
      `Ayırma yarıda durduruldu (${fmt.int(progress.done)}/${fmt.int(progress.total)} dosya). Kopyalananlar kişi klasörlerinde duruyor.`,
    );
  } else if (data.relocated) {
    showNotice(notice, "i-info", "Çıktı klasörü taşınmış; dosyalar yeni yerinden gösteriliyor.");
  } else if (summary.missing) {
    notice.classList.add("warn");
    showNotice(
      notice,
      "i-warn",
      `${plural(summary.missing, "dosya")} artık klasöründe değil. Elle taşınmış ya da silinmiş olabilir.`,
    );
  }

  for (const person of data.people) list.append(personBlock(person, data));
  if (!data.people.length) {
    list.append(el("div", "empty-results", "Bu raporda kimse yok."));
  }
  renderUnmatched(data.unmatched || {});
}

function showNotice(node, iconName, text) {
  node.replaceChildren(icon(iconName), el("p", "", text));
  node.hidden = false;
}

function personBlock(person, data) {
  const block = el("article", "person-block");
  const head = el("div", "person-head");
  head.append(avatar(person));
  const main = el("div", "row-main");
  main.append(el("p", "t-headline", person.name));
  const counts = person.counts;
  const metaParts = [];
  if (counts.files) {
    metaParts.push(plural(counts.files, "dosya"));
    if (counts.review) metaParts.push(`${fmt.int(counts.review)} kontrol edilmeli`);
    if (counts.rejected) metaParts.push(`${fmt.int(counts.rejected)} ayıklandı`);
  } else {
    metaParts.push("Bu ayırmada bulunamadı");
  }
  main.append(el("p", "t-footnote muted", metaParts.join(" · ")));
  head.append(main);
  const actions = el("div", "actions");
  if (counts.files) {
    actions.append(button("Tümünü gör", "small", () => openSheet(person.slug)));
    actions.append(
      button(folderLabel(), "small quiet", () => openInFileManager({ output: data.output, slug: person.slug }), "i-open"),
    );
  } else if (person.enrolled) {
    const add = el("a", "btn small", "Açı ekle");
    add.href = cameraLink(person.name);
    actions.append(add);
  }
  head.append(actions);
  block.append(head);
  const note = leftoverNotice(person);
  if (note) block.append(note);
  if (!counts.files) {
    block.classList.add("empty");
    return block;
  }
  const strip = el("div", "strip");
  // One row only: as many tiles as fit beside the "+N" tile.
  const width = Math.max(0, $("results-people").clientWidth - 32);
  const perRow = width >= 700 - 32 ? Math.floor((width + 8) / 128) : Math.floor((width + 8) / 104);
  const room = Math.max(2, perRow - 1);
  const shown = person.files.length > room + 1 ? person.files.slice(0, room) : person.files;
  for (const item of shown) strip.append(tile(item, () => openSheet(person.slug, item.path)));
  if (person.files.length > shown.length) {
    const more = el("button", "tile more", `+${fmt.int(person.files.length - shown.length)}`);
    more.type = "button";
    more.setAttribute("aria-label", `${person.name}: tüm dosyaları gör`);
    more.addEventListener("click", () => openSheet(person.slug));
    strip.append(more);
  }
  block.append(strip);
  return block;
}

function thumbAt(url, size) {
  if (!url) return null;
  return size ? `${url}&size=${size}` : url;
}

function tile(item, onClick, size = 640) {
  const node = el("button", "tile");
  node.type = "button";
  node.setAttribute("aria-label", item.name || item.path);
  node.title = item.source ? `Kaynak: ${item.source}` : item.name || "";
  if (item.exists === false) node.classList.add("missing");
  const url = thumbAt(item.thumb, size);
  if (url) {
    const img = el("img");
    img.alt = "";
    img.loading = "lazy";
    img.decoding = "async";
    img.addEventListener("load", () => img.classList.add("in"));
    img.addEventListener("error", () => {
      img.remove();
      node.append(placeholder(item));
    });
    img.src = url;
    node.append(img);
  } else {
    node.append(placeholder(item));
  }
  if (item.kind === "video") {
    const kind = el("span", "kind");
    kind.append(icon("i-film"));
    kind.title = "Video";
    node.append(kind);
  }
  const reasons = Array.isArray(item.review) ? item.review : [];
  if (reasons.length) {
    const badge = el("span", "badge warn", REVIEW_LABEL[reasons[0]] || "kontrol et");
    node.append(badge);
  }
  if (onClick) node.addEventListener("click", onClick);
  return node;
}

function placeholder(item) {
  const box = el("span", "placeholder");
  box.append(icon(item.kind === "video" ? "i-film" : "i-image"));
  return box;
}

function renderUnmatched(unmatched) {
  const errors = unmatched.errors || [];
  const skipped = unmatched.skipped || [];
  const uncertain = unmatched.uncertain || [];
  const total = errors.length + skipped.length + uncertain.length;
  const box = $("unmatched");
  const list = $("unmatched-list");
  list.replaceChildren();
  if (!total) {
    box.hidden = true;
    return;
  }
  const parts = [];
  if (errors.length) parts.push(`${plural(errors.length, "dosya")} okunamadı`);
  if (skipped.length) parts.push(`${plural(skipped.length, "dosya")} atlandı`);
  if (uncertain.length) parts.push(`${plural(uncertain.length, "dosya")} belirsiz`);
  setText("unmatched-title", parts.join(" · "));
  for (const row of errors) list.append(unmatchedRow(row.source, row.error));
  for (const row of skipped) list.append(unmatchedRow(row.source, row.note));
  for (const row of uncertain) {
    const names = row.candidates.map((c) => `${c.name} (${fmt.two(c.score)})`).join(", ");
    list.append(unmatchedRow(row.source, `Belirsiz: ${names}`));
  }
  box.hidden = false;
}

function unmatchedRow(path, text) {
  const row = el("div", "unmatched-row");
  const name = el("span", "path", path);
  name.title = path;
  row.append(name, el("span", "muted", text));
  return row;
}

/* ---------- Person sheet ---------- */

async function openSheet(slug, focusPath) {
  const dialog = $("person-sheet");
  state.sheet.slug = slug;
  state.sheet.filter = "all";
  state.sheet.data = null;
  for (const chip of document.querySelectorAll("#sheet-filters .filter")) {
    chip.setAttribute("aria-pressed", chip.dataset.filter === "all" ? "true" : "false");
  }
  const known = state.results && state.results.people.find((row) => row.slug === slug);
  if (known) state.sheet.data = known;
  renderSheet();
  if (!dialog.open) dialog.showModal();
  dialog.scrollTop = 0;
  await refreshSheet();
  if (focusPath) {
    const target = dialog.querySelector(`[data-path="${CSS.escape(focusPath)}"] .tile`);
    if (target) target.focus();
  }
}

async function refreshSheet() {
  if (!state.results || !state.sheet.slug) return;
  try {
    const data = await api(
      `/api/sonuclar?${new URLSearchParams({ output: state.results.output, person: state.sheet.slug })}`,
    );
    const person = data.people[0];
    if (!person) return;
    state.sheet.data = person;
    const index = state.results.people.findIndex((row) => row.slug === person.slug);
    if (index >= 0) state.results.people[index] = person;
    state.results.summary = data.summary;
    renderSheet();
    renderResults();
  } catch (error) {
    toast(error.message, { kind: "bad" });
  }
}

function renderSheet() {
  const person = state.sheet.data;
  const grid = $("sheet-grid");
  const rejectedGrid = $("sheet-rejected-grid");
  grid.replaceChildren();
  rejectedGrid.replaceChildren();
  if (!person) return;
  setText("sheet-title", person.name);
  const cover = $("sheet-cover");
  if (person.cover) {
    cover.src = person.cover;
    cover.hidden = false;
  } else {
    cover.hidden = true;
  }
  const counts = person.counts;
  const meta = [plural(counts.files, "dosya")];
  if (counts.videos) meta.push(`${fmt.int(counts.videos)} video`);
  if (counts.review) meta.push(`${fmt.int(counts.review)} kontrol edilmeli`);
  if (counts.rejected) meta.push(`${fmt.int(counts.rejected)} ayıklandı`);
  setText("sheet-meta", meta.join(" · "));
  const leftovers = $("sheet-leftovers");
  const note = leftoverNotice(person);
  leftovers.replaceChildren(...(note ? [note] : []));
  leftovers.hidden = !note;
  const filter = state.sheet.filter;
  const files = person.files.filter((item) => filter === "all" || item.review.length > 0);
  if (!files.length) {
    grid.append(
      el(
        "p",
        "t-callout muted",
        filter === "review" ? "Kontrol edilecek bir dosya yok." : "Bu kişi için dosya yok.",
      ),
    );
  }
  for (const item of files) grid.append(sheetTile(person, item));
  const rejectedBox = $("sheet-rejected");
  rejectedBox.hidden = !person.rejected.length;
  for (const item of person.rejected) rejectedGrid.append(rejectedTile(person, item));
}

function sheetTile(person, item) {
  const wrap = el("div", "tile-wrap");
  wrap.dataset.path = item.path;
  wrap.append(tile(item, () => openLightbox(person, item)));
  const reject = button("Bu kişi değil", "small reject", () => rejectMatch(person, item, wrap), "i-x");
  reject.setAttribute("aria-label", `${item.name}: bu kişi değil`);
  wrap.append(reject);
  return wrap;
}

function rejectedTile(person, item) {
  const wrap = el("div", "tile-wrap");
  wrap.dataset.path = item.path;
  wrap.append(tile({ ...item, review: [] }, null));
  const undo = button("Geri al", "small reject", () => undoReject(person, item.path), "i-undo");
  wrap.append(undo);
  return wrap;
}

async function rejectMatch(person, item, wrap) {
  const output = state.results.output;
  wrap.classList.add("leaving");
  try {
    const data = await postJson("/api/sonuclar/yanlis-eslesme", {
      output,
      path: item.path,
      slug: person.slug,
    });
    announce(`${item.name} ${person.name} klasöründen çıkarıldı.`);
    toast(`${person.name} klasöründen çıkarıldı · Fotoğraf silinmedi`, {
      action: "Geri al",
      onAction: () => undoReject(person, data.movedTo),
    });
    await refreshSheet();
    reloadPeople();
  } catch (error) {
    wrap.classList.remove("leaving");
    toast(error.message, { kind: "bad" });
  }
}

async function undoReject(person, movedTo) {
  const output = state.results.output;
  try {
    await postJson("/api/sonuclar/geri-al", { output, path: movedTo });
    toast(`${person.name} klasörüne geri kondu.`, { duration: 4000 });
    await refreshSheet();
    reloadPeople();
  } catch (error) {
    toast(error.message, { kind: "bad" });
  }
}

function leftoverNotice(person) {
  const names = person.leftovers || [];
  if (!names.length) return null;
  const note = el("div", "notice warn");
  note.append(
    icon("i-warn"),
    el("p", "", `Önceki taramalardan kalan ${fmt.int(names.length)} dosya bu klasörde duruyor.`),
    button("Önceki taramaya taşı", "small", () => moveLeftovers(person), "i-folder"),
  );
  return note;
}

async function moveLeftovers(person) {
  const output = state.results.output;
  try {
    const data = await postJson("/api/sonuclar/eskileri-ayir", { output, slug: person.slug });
    toast(`${plural(data.moved, "dosya")} önceki taramalar klasörüne taşındı · Hiçbiri silinmedi`, {
      duration: 6000,
    });
    await reloadResults(output);
    if ($("person-sheet").open) await refreshSheet();
  } catch (error) {
    toast(error.message, { kind: "bad" });
  }
}

/* ---------- Lightbox ---------- */

function openLightbox(person, item) {
  const dialog = $("lightbox");
  const img = $("lightbox-img");
  const box = $("lightbox-box");
  box.hidden = true;
  img.removeAttribute("src");
  img.alt = item.name;
  const caption = [item.name];
  if (item.score !== null && item.score !== undefined) caption.push(`benzerlik ${fmt.two(item.score)}`);
  if (item.facePx) caption.push(`yüz ${fmt.int(Math.round(item.facePx))} px`);
  if (item.review && item.review.length) {
    caption.push(item.review.map((reason) => REVIEW_LABEL[reason] || reason).join(", "));
  }
  if (item.with && item.with.length) {
    const names = item.with
      .map((slug) => (state.people.find((p) => p.slug === slug) || {}).name || slug)
      .join(", ");
    caption.push(`karede ayrıca: ${names}`);
  }
  setText("lightbox-caption", caption.join(" · "));
  const place = () => {
    if (!Array.isArray(item.box) || item.box.length !== 4 || !img.clientWidth) {
      box.hidden = true;
      return;
    }
    const [x, y, w, h] = item.box;
    box.style.left = `${(x * img.clientWidth).toFixed(1)}px`;
    box.style.top = `${(y * img.clientHeight).toFixed(1)}px`;
    box.style.width = `${(w * img.clientWidth).toFixed(1)}px`;
    box.style.height = `${(h * img.clientHeight).toFixed(1)}px`;
    box.hidden = false;
  };
  img.onload = place;
  img.src = thumbAt(item.thumb, 1280);
  if (!dialog.open) dialog.showModal();
  window.addEventListener("resize", place, { once: true });
}

/* ---------- People ---------- */

function cameraLink(name) {
  return name ? `/kamera?ad=${encodeURIComponent(name)}` : "/kamera?yeni=1";
}

function renderPeople() {
  const grid = $("people");
  grid.replaceChildren();
  grid.removeAttribute("aria-busy");
  for (const person of state.people) grid.append(personCard(person));
  const add = el("a", "add-card");
  add.href = "/kamera?yeni=1";
  const plus = el("span", "add-icon");
  plus.append(icon("i-plus"));
  add.append(
    plus,
    el("span", "t-headline", state.people.length ? "Kişi ekle" : "Kamerayla tanıt"),
    el("span", "t-footnote", "Kamerada başınızı yavaşça çevirin; 16 açı yeterli"),
  );
  grid.append(add);
}

function personCard(person) {
  const card = el("article", "person");
  const top = el("div", "top");
  const ring = el("span", "coverage");
  const refs = count(person.references);
  ring.style.setProperty("--p", String(Math.min(100, Math.round((refs / 16) * 100))));
  if (refs < 5) ring.classList.add("low");
  ring.title = refs < 5 ? "Az açı: yan bakışlar daha sık kaçar" : `${fmt.int(refs)} açı`;
  ring.append(avatar(person));
  top.append(ring);
  const name = el("div", "name");
  name.append(el("p", "t-headline", person.name));
  const meta = el("p", "t-footnote muted");
  const parts = [plural(refs, "açı")];
  if (person.clothingLabel) parts.push(`kıyafet: ${person.clothingLabel}`);
  meta.textContent = parts.join(" · ");
  name.append(meta);
  top.append(name);
  card.append(top);
  const last = person.lastScan;
  const line = el("p", "t-footnote");
  if (last && last.files) {
    line.textContent = `Son ayırmada ${plural(last.files, "dosya")}`;
    line.classList.add("muted");
  } else if (last) {
    line.textContent = "Son ayırmada bulunamadı";
    line.classList.add("muted");
  } else {
    line.textContent = "Henüz ayırmaya katılmadı";
    line.classList.add("muted");
  }
  card.append(line);
  const actions = el("div", "actions");
  const more = el("a", "btn small", "Açı ekle");
  more.href = cameraLink(person.name);
  const remove = button("Sil", "small quiet", () => confirmDeletePerson(person, card, remove), "i-trash");
  remove.setAttribute("aria-label", `${person.name} kaydını sil`);
  actions.append(more, remove);
  card.append(actions);
  return card;
}

let deleteDecision = null;

function confirmDeletePerson(person, card, buttonNode) {
  setText("delete-title", `${person.name} silinsin mi?`);
  setText(
    "delete-text",
    "Kişinin kaydı ve açıları kaldırılır. Etkinlik arşivinde bu kişinin fotoğrafları artık görünmez. Fotoğraflar çıktı klasöründe kalır. Hemen çıkacak «Geri al» ile geri getirebilirsiniz.",
  );
  return new Promise((resolve) => {
    deleteDecision = resolve;
    $("delete-dialog").showModal();
  }).then((confirmed) => {
    if (confirmed) deletePerson(person, card, buttonNode);
  });
}

async function deletePerson(person, card, buttonNode) {
  buttonNode.disabled = true;
  try {
    const data = await api(`/api/kisiler/${encodeURIComponent(person.slug)}`, { method: "DELETE" });
    card.remove();
    state.people = state.people.filter((row) => row.slug !== person.slug);
    state.skipped.delete(person.slug);
    renderWho();
    renderOverview();
    updatePreflight();
    await refreshResults();
    announce(`${person.name} silindi.`);
    toast(`${person.name} silindi`, {
      action: "Geri al",
      onAction: async () => {
        await postJson("/api/kisiler/geri-al", { token: data.restoreToken });
        toast(`${person.name} geri alındı.`, { duration: 4000 });
        await reloadPeople();
        await refreshResults();
      },
    });
    if (!state.people.length) renderPeople();
  } catch (error) {
    buttonNode.disabled = false;
    let line = card.querySelector(".error");
    if (!line) {
      line = el("p", "t-footnote error");
      card.append(line);
    }
    line.textContent = error.message;
  }
}

/* ---------- Who to sort for ---------- */

function renderWho() {
  const list = $("who-list");
  list.replaceChildren();
  $("who-group").hidden = state.people.length <= 1;
  for (const person of state.people) {
    const chip = el("label", "who-chip");
    const box = el("input");
    box.type = "checkbox";
    box.value = person.slug;
    box.checked = !state.skipped.has(person.slug);
    box.addEventListener("change", () => {
      if (box.checked) state.skipped.delete(person.slug);
      else state.skipped.add(person.slug);
      paintWhoNote();
      updatePreflight();
    });
    chip.append(box);
    chip.append(avatar(person, "mono"));
    chip.append(el("span", "", person.name));
    chip.append(icon("i-check"));
    list.append(chip);
  }
  paintWhoNote();
}

function chosenPeople() {
  if (state.people.length === 1) return [state.people[0].slug];
  return state.people.filter((person) => !state.skipped.has(person.slug)).map((p) => p.slug);
}

function paintWhoNote() {
  const total = state.people.length;
  const chosen = chosenPeople().length;
  let text = "";
  if (total > 1) {
    if (!chosen) text = "Kimse seçili değil. Ayırmak için en az bir kişiyi işaretleyin.";
    else if (chosen === total) text = `Kayıtlı ${fmt.int(total)} kişinin hepsi için ayrılır.`;
    else
      text =
        `${fmt.int(total)} kişiden ${fmt.int(chosen)} kişi için ayrılır. Yüzler yine herkesle karşılaştırılır; ` +
        "yalnızca seçilenlerin klasörü yazılır.";
  }
  setText("who-note", text);
  $("who-note").hidden = !text;
}

function setAllWho(checked) {
  for (const box of document.querySelectorAll("#who-list input")) {
    box.checked = checked;
    if (checked) state.skipped.delete(box.value);
    else state.skipped.add(box.value);
  }
  paintWhoNote();
  updatePreflight();
}

/* ---------- Photo enrollment (dialog) ---------- */

function openEnroll() {
  const dialog = $("enroll-dialog");
  if (!dialog.open) dialog.showModal();
  $("person-name").focus();
}

function closeEnroll() {
  $("enroll-dialog").close();
}

function resetEnroll() {
  state.enroll = { token: "", images: [], picks: new Map() };
  $("preview").replaceChildren();
  $("files").value = "";
  setText("drop-status", "");
  setKind($("drop-status"), null);
}

function paintNameHint() {
  const typed = $("person-name").value.trim().toLocaleLowerCase(LOCALE);
  const hint = $("name-hint");
  if (!typed) {
    hint.hidden = true;
    return;
  }
  const known = state.people.find((p) => p.name.toLocaleLowerCase(LOCALE) === typed);
  if (known) {
    hint.textContent = `Bu ad kayıtlı (${plural(known.references, "açı")}). Yeni açılar bu kişiye eklenir.`;
    hint.hidden = false;
  } else {
    hint.hidden = true;
  }
}

function renderPreview(images) {
  const root = $("preview");
  root.replaceChildren();
  const picks = state.enroll.picks;
  picks.clear();
  state.enroll.images = images;
  for (const image of images) {
    const block = el("div", "shot");
    block.dataset.image = String(image.index);
    const head = el("p", "t-footnote");
    head.append(el("b", "", image.filename));
    if (image.message) head.append(document.createTextNode(` · ${image.message}`));
    block.append(head);
    if (image.faces.length) {
      const strip = el("div", "shot-faces");
      if (image.faces.length === 1) picks.set(image.index, image.faces[0].index);
      for (const face of image.faces) {
        const face_button = el("button", "face");
        face_button.type = "button";
        face_button.dataset.image = String(image.index);
        face_button.dataset.face = String(face.index);
        face_button.setAttribute("aria-pressed", picks.get(image.index) === face.index ? "true" : "false");
        face_button.setAttribute("aria-label", `${image.filename} yüz ${face.index + 1}`);
        const img = el("img");
        img.src = face.crop;
        img.alt = "";
        face_button.append(img);
        face_button.addEventListener("click", () => {
          if (picks.get(image.index) === face.index) picks.delete(image.index);
          else picks.set(image.index, face.index);
          block.classList.remove("unresolved");
          paintPressed();
        });
        strip.append(face_button);
      }
      block.append(strip);
    }
    root.append(block);
  }
}

function paintPressed() {
  const picks = state.enroll.picks;
  for (const node of document.querySelectorAll(".face")) {
    const image = Number(node.dataset.image);
    const face = Number(node.dataset.face);
    node.setAttribute("aria-pressed", picks.get(image) === face ? "true" : "false");
  }
}

const UPLOAD_LIMIT = 20 * 1024 * 1024;
const SHRINK_ABOVE = 6 * 1024 * 1024;
const SERVER_EDGE = 2560; // /api/onizleme resizes to this edge before looking for faces

async function shrinkForUpload(file) {
  // Event cameras write 25 MB frames; the server would scale them to 2560 px anyway.
  if (file.size <= SHRINK_ABOVE || typeof createImageBitmap !== "function") return file;
  let bitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    return file; // HEIC outside Safari: sent as is, the size check still applies
  }
  try {
    const scale = Math.min(1, SERVER_EDGE / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
    return blob ? new File([blob], file.name, { type: "image/jpeg" }) : file;
  } finally {
    bitmap.close();
  }
}

async function sendPreview(fileList) {
  const picked = [...fileList];
  if (!picked.length) return;
  const status = $("drop-status");
  if (picked.length > 12) {
    setText("drop-status", "Bir seferde en fazla 12 fotoğraf seçin.");
    setKind(status, "bad");
    return;
  }
  const stamp = Symbol("preview");
  state.enroll.stamp = stamp;
  $("save").disabled = true;
  setKind(status, null);
  const files = [];
  for (const file of picked) {
    if (file.size > SHRINK_ABOVE) setText("drop-status", `${file.name} küçültülüyor…`);
    files.push(await shrinkForUpload(file)); // one at a time: a 6000 px frame is ~100 MB decoded
    if (state.enroll.stamp !== stamp) return;
  }
  const big = files.find((file) => file.size > UPLOAD_LIMIT);
  if (big) {
    setText("drop-status", `${big.name} açılamadı ve 20 MB sınırını aşıyor (${fmt.bytes(big.size)}).`);
    setKind(status, "bad");
    $("save").disabled = false;
    return;
  }
  const body = new FormData();
  for (const file of files) body.append("files", file, file.name);
  setText("drop-status", `${plural(files.length, "fotoğraf")} inceleniyor…`);
  try {
    const data = await api("/api/onizleme", { method: "POST", body });
    if (state.enroll.stamp !== stamp) return;
    state.enroll.token = data.token;
    renderPreview(data.images);
    const found = data.images.filter((image) => image.faces.length).length;
    setText(
      "drop-status",
      found ? `${plural(found, "karede")} yüz bulundu. Doğru yüzleri seçip kaydedin.` : "Hiçbir karede yüz bulunamadı.",
    );
    setKind(status, found ? "ok" : "bad");
  } catch (error) {
    if (state.enroll.stamp !== stamp) return;
    state.enroll.token = "";
    setText("drop-status", error.message);
    setKind(status, "bad");
  } finally {
    $("save").disabled = false;
  }
}

async function savePerson(event) {
  event.preventDefault();
  const statusNode = $("enroll-status");
  const name = $("person-name").value.trim();
  if (!name) {
    setText("enroll-status", "Ad soyad gerekli.");
    setKind(statusNode, "bad");
    $("person-name").focus();
    return;
  }
  if (!state.enroll.token) {
    setText("enroll-status", "Önce tanıtım fotoğrafı seçin.");
    setKind(statusNode, "bad");
    return;
  }
  for (const image of state.enroll.images) {
    if (image.faces.length > 1 && !state.enroll.picks.has(image.index)) {
      setText("enroll-status", "Birden fazla yüz olan karede kişiyi seçin.");
      setKind(statusNode, "bad");
      const block = document.querySelector(`.shot[data-image="${image.index}"]`);
      if (block) {
        block.classList.add("unresolved");
        block.scrollIntoView({ block: "nearest" });
      }
      return;
    }
  }
  const faces = [...state.enroll.picks.entries()].map(([image, face]) => ({ image, face }));
  if (!faces.length) {
    setText("enroll-status", "Seçilen fotoğraflarda kullanılacak yüz yok.");
    setKind(statusNode, "bad");
    return;
  }
  $("save").disabled = true;
  setText("enroll-status", "Kaydediliyor…");
  setKind(statusNode, null);
  try {
    const data = await postJson("/api/kisiler", {
      token: state.enroll.token,
      name,
      clothing: $("clothing").value,
      faces,
    });
    const parts = [`${data.name} kaydedildi`, plural(data.references, "açı")];
    if (data.clothingLabel) parts.push(`kıyafet: ${data.clothingLabel}`);
    setText("enroll-status", "");
    resetEnroll();
    $("person-name").value = "";
    $("clothing").value = "";
    paintNameHint();
    closeEnroll();
    toast(parts.join(" · ") + (data.warning ? `. ${data.warning}` : ""), {
      duration: data.warning ? 12000 : 6000,
    });
    await reloadPeople();
  } catch (error) {
    setText("enroll-status", error.message);
    setKind(statusNode, "bad");
  } finally {
    $("save").disabled = false;
  }
}

function bindDrop() {
  const drop = $("drop");
  const input = $("files");
  input.addEventListener("change", () => sendPreview(input.files));
  drop.addEventListener("dragover", (event) => {
    event.preventDefault();
    drop.classList.add("hot");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("hot"));
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    drop.classList.remove("hot");
    if (event.dataTransfer) sendPreview(event.dataTransfer.files);
  });
}

/* ---------- Folder rows ---------- */

const poolTargets = { source: "", target: "", defaultSource: "", defaultTarget: "" };

function splitPath(path) {
  const trimmed = path.replace(/[\\/]+$/, "");
  const cut = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  if (cut < 0) return { parent: "", name: trimmed, sep: "/" };
  return { parent: trimmed.slice(0, cut), name: trimmed.slice(cut + 1), sep: trimmed[cut] };
}

function homePath(path) {
  return path.replace(/^\/Users\/[^/]+(?=\/|$)/, "~").replace(/^C:\\Users\\[^\\]+(?=\\|$)/, "~");
}

function showFolder(target) {
  const value = $(target).value.trim();
  const node = $(`${target}-path`);
  node.title = value;
  node.classList.toggle("split", Boolean(value));
  if (!value) {
    node.textContent = "Seçilmedi";
  } else {
    // The parent shrinks with an ellipsis; the folder's own name always stays readable.
    const { parent, name, sep } = splitPath(homePath(value));
    node.replaceChildren(el("span", "path-head", parent ? parent + sep : ""), el("span", "path-tail", name));
  }
  $(`${target}-pick`).textContent = value ? "Değiştir…" : "Seç…";
  updatePreflight();
}

// A source other than the default event folder gets its own pool beside it.
function followSource(path) {
  if (!path || path === poolTargets.target) return;
  if (path === poolTargets.defaultSource) {
    poolTargets.source = poolTargets.defaultSource;
    poolTargets.target = poolTargets.defaultTarget;
    return;
  }
  const { parent, name, sep } = splitPath(path);
  poolTargets.source = path;
  poolTargets.target = `${parent}${sep}${name} (tekil)`;
}

async function chooseFolder(target) {
  const pick = $(`${target}-pick`);
  if (pick.getAttribute("aria-busy") === "true") return;
  pick.setAttribute("aria-busy", "true");
  const status = $("pick-status");
  status.hidden = false;
  status.textContent = PICK_OPENED[state.platform] || PICK_OPENED.linux;
  setKind(status, null);
  try {
    const data = await postJson("/api/klasor-sec", { purpose: target, current: $(target).value.trim() });
    status.hidden = true;
    if (typeof data.path === "string" && data.path) {
      $(target).value = data.path;
      showFolder(target);
      if (target === "source") followSource(data.path);
    }
  } catch (error) {
    status.textContent = error.message;
    setKind(status, "bad");
    // Only a missing dialog (not a busy one, not a network error) opens the typed-path field.
    if (error.status === 400 && /elle/.test(error.message)) {
      $(`${target}-manual`).hidden = false;
      $(target).focus();
    }
  } finally {
    pick.removeAttribute("aria-busy");
  }
}

/* ---------- Unique pool ---------- */

function poolLine(summary) {
  const parts = [`${fmt.int(count(summary.photos))} fotoğraf, ${fmt.int(count(summary.videos))} video`];
  const left = [];
  if (summary.duplicates) left.push(`${fmt.int(summary.duplicates)} kopya`);
  if (summary.broken) left.push(`${fmt.int(summary.broken)} açılmayan dosya`);
  if (left.length) parts.push(`${left.join(" ve ")} alınmadı`);
  if (summary.emptyFiles) parts.push(`${fmt.int(summary.emptyFiles)} dosya 0 bayt; kopyalama yarım kalmış olabilir`);
  return `Havuz hazır: ${parts.join(". ")}.`;
}

function setPoolStatus(text, kind) {
  const node = $("pool-status");
  node.textContent = text;
  node.hidden = !text;
  setKind(node, kind);
  node.style.color = kind === "ok" ? "var(--ok-text)" : kind === "bad" ? "var(--bad)" : "";
}

function renderPool(data) {
  if (data.state === "running") {
    lockForm(true);
    const total = count(data.total);
    const done = count(data.done);
    showJob({
      title: "Tekil havuz hazırlanıyor",
      current: data.current || "",
      ratio: total ? done / total : null,
      line: total ? `${fmt.int(done)}/${fmt.int(total)} dosya · ${fmt.pct(done / total)}` : "Aynı dosyalar aranıyor…",
      actions: [],
      people: [],
      indeterminate: !total,
    });
    setActivity(`Havuz · ${total ? fmt.pct(done / total) : "…"}`, total ? done / total : 0);
    $("pool-start").disabled = true;
    return;
  }
  $("pool-start").disabled = false;
  if (data.state === "error") {
    setPoolStatus(data.error || "Havuz hazırlanamadı. Yeniden deneyin; sorun sürerse terminaldeki kayda bakın.", "bad");
    finishJob();
    return;
  }
  if (data.state === "done" && data.summary) {
    setPoolStatus(`${poolLine(data.summary)} Kaynak havuza çevrildi.`, "ok");
    if (data.destination) {
      $("source").value = data.destination;
      showFolder("source");
    }
    $("pool-start").textContent = "Yenile";
    state.pool = data.summary;
    renderOverview();
    finishJob();
  }
}

async function pollPool() {
  window.clearTimeout(state.poolTimer);
  try {
    const data = await api("/api/havuz");
    setConnection("ok");
    renderPool(data);
    if (data.state === "running") state.poolTimer = window.setTimeout(pollPool, 600);
    else {
      lockForm(false);
      updatePreflight();
    }
  } catch (error) {
    if (error.status === 0) {
      setConnection("retry");
      state.poolTimer = window.setTimeout(pollPool, 2000);
      return;
    }
    setPoolStatus(error.message, "bad");
    $("pool-start").disabled = false;
  }
}

async function startPool() {
  window.clearTimeout(state.poolTimer);
  if (!poolTargets.source) {
    setPoolStatus("Önce kaynak klasörü seçin; havuz o klasörden hazırlanır.", "bad");
    return;
  }
  $("pool-start").disabled = true;
  setPoolStatus("Başlıyor…");
  try {
    await postJson("/api/havuz", { source: poolTargets.source, destination: poolTargets.target });
    pollPool();
  } catch (error) {
    if (error.status === 409 && /zaten/.test(error.message)) {
      pollPool();
      return;
    }
    setPoolStatus(error.message, "bad");
    $("pool-start").disabled = false;
  }
}

/* ---------- Scan form: preflight, move mode, start ---------- */

function paintThreshold() {
  const value = Number($("threshold").value);
  setText("threshold-out", fmt.two(value));
  const defaults = state.settings || { threshold: 0.5, interval: 0.5 };
  const parts = [];
  if (Math.abs(value - defaults.threshold) > 0.0001) parts.push(`eşik ${fmt.two(value)}`);
  const interval = Number($("interval").value);
  if (Math.abs(interval - defaults.interval) > 0.0001) parts.push(`video aralığı ${fmt.one(interval)} sn`);
  setText("advanced-summary", parts.length ? parts.join(" · ") : "Varsayılan değerler");
}

function resetAdvanced() {
  const defaults = state.settings || { threshold: 0.5, interval: 0.5 };
  $("threshold").value = String(defaults.threshold);
  $("interval").value = String(defaults.interval);
  paintThreshold();
}

function paintMove() {
  const on = $("move").checked;
  $("move-warning").hidden = !on;
  $("start").textContent = on ? "Taşımayı başlat" : "Ayırmayı başlat";
}

function lockForm(locked) {
  state.locked = locked;
  $("scan-fields").disabled = locked || state.people.length === 0;
  $("start").disabled = locked || !preflightOk();
  $("open-enroll").disabled = locked;
}

function preflightItems() {
  const items = [];
  if (!state.people.length) {
    items.push({ text: "Ayırmak için önce en az bir kişiyi tanıtın.", href: "/kamera" });
  }
  if (!$("source").value.trim()) items.push({ text: "Kaynak klasörü seçin.", id: "source-pick" });
  if (!$("destination").value.trim()) items.push({ text: "Çıktı klasörünü seçin.", id: "destination-pick" });
  if (state.people.length > 1 && !chosenPeople().length) items.push({ text: "En az bir kişiyi işaretleyin.", id: "who-list" });
  return items;
}

function preflightOk() {
  return preflightItems().length === 0;
}

function updatePreflight() {
  const list = $("preflight");
  list.replaceChildren();
  const items = preflightItems();
  for (const item of items) {
    const li = el("li");
    li.append(icon("i-info"));
    if (item.href) {
      const link = el("a", "", item.text);
      link.href = item.href;
      li.append(link);
    } else if (item.id) {
      const link = el("a", "", item.text);
      link.href = "#ayir";
      link.addEventListener("click", (event) => {
        event.preventDefault();
        const target = $(item.id);
        if (target) {
          target.scrollIntoView({ block: "center" });
          const focusable = target.matches("button, input") ? target : target.querySelector("input, button");
          if (focusable) focusable.focus();
        }
      });
      li.append(link);
    } else li.append(el("span", "", item.text));
    list.append(li);
  }
  list.hidden = !items.length;
  const running = state.scan && state.scan.state === "running";
  $("scan-fields").disabled = Boolean(state.locked) || state.people.length === 0;
  $("start").disabled = Boolean(running) || items.length > 0 || Boolean(state.locked);
  const chosen = chosenPeople().length;
  const who = chosen === 1 ? "1 kişi" : `${fmt.int(chosen)} kişi`;
  setText("start-title", items.length ? "Başlamadan önce" : `${who} için ${$("move").checked ? "taşıma" : "kopyalama"}`);
  setText(
    "start-note",
    items.length
      ? ""
      : $("move").checked
        ? "Eşleşen dosyalar kişi klasörlerine taşınır ve kaynaktan silinir."
        : "Eşleşen dosyalar kişi klasörlerine kopyalanır; kaynak olduğu gibi kalır.",
  );
}

function submitScan(event) {
  event.preventDefault();
  if (!preflightOk()) {
    updatePreflight();
    return;
  }
  if ($("move").checked) {
    $("move-dialog").showModal();
    return;
  }
  startScan();
}

async function startScan() {
  window.clearTimeout(state.pollTimer);
  const people = chosenPeople();
  lockForm(true);
  state.lastMilestone = -1;
  state.lastPhase = "";
  showJob({
    title: "Ayırma başlıyor",
    current: "",
    ratio: null,
    line: "",
    actions: [],
    people: [],
    indeterminate: true,
  });
  setJobStatus("", null);
  try {
    await postJson("/api/tara", {
      source: $("source").value.trim(),
      destination: $("destination").value.trim(),
      threshold: Number($("threshold").value),
      interval: Number($("interval").value),
      move: $("move").checked,
      people,
    });
    state.wasRunning = true;
    announce("Ayırma başladı.");
    poll();
  } catch (error) {
    if (error.status === 409 && /zaten/.test(error.message)) {
      // Another tab started it: attach to the running job instead of failing.
      state.wasRunning = true;
      setJobStatus("Bir ayırma zaten sürüyor; ilerlemesi burada.", null);
      poll();
      return;
    }
    finishJob();
    lockForm(false);
    updatePreflight();
    setJobStatus(error.message, "bad");
    $("job").hidden = false;
  }
}

async function stopScan(buttonNode) {
  buttonNode.disabled = true;
  buttonNode.textContent = "Durduruluyor…";
  try {
    await postJson("/api/tara/durdur");
  } catch (error) {
    buttonNode.disabled = false;
    buttonNode.textContent = "Durdur";
    if (error.status !== 409) toast(error.message, { kind: "bad" });
  }
}

/* ---------- Progress component (one for scan and pool) ---------- */

function showJob({ title, current, ratio, line, actions, people, indeterminate, thumb, kind }) {
  const job = $("job");
  job.hidden = false;
  setText("job-title", title);
  const currentNode = $("job-current");
  currentNode.textContent = current || "";
  currentNode.title = current || "";
  currentNode.hidden = !current;
  const meter = $("meter");
  meter.className = `meter${indeterminate ? " indeterminate" : ""}${kind ? ` ${kind}` : ""}`;
  const value = ratio === null || ratio === undefined ? 0 : Math.max(0, Math.min(1, ratio));
  $("meter-fill").style.transform = `scaleX(${indeterminate ? 0.3 : value})`;
  if (indeterminate) meter.removeAttribute("aria-valuenow");
  else meter.setAttribute("aria-valuenow", String(Math.round(value * 100)));
  setText("job-eta", line || "");
  const peopleNode = $("job-people");
  peopleNode.replaceChildren();
  for (const chip of people || []) peopleNode.append(chip);
  const actionsNode = $("job-actions");
  actionsNode.replaceChildren();
  for (const node of actions || []) actionsNode.append(node);
  const thumbNode = $("job-thumb");
  if (thumb) {
    if (thumbNode.dataset.src !== thumb) {
      thumbNode.dataset.src = thumb;
      thumbNode.src = thumb;
    }
    thumbNode.hidden = false;
  } else {
    thumbNode.hidden = true;
    thumbNode.removeAttribute("src");
    delete thumbNode.dataset.src;
  }
}

function setJobStatus(text, kind) {
  const node = $("job-status");
  node.textContent = text || "";
  setKind(node, kind);
}

function finishJob() {
  setActivity(null);
  document.title = "Etkinlik arşivi";
}

function setActivity(text, ratio) {
  const pill = $("activity");
  if (!text) {
    pill.hidden = true;
    return;
  }
  pill.hidden = false;
  setText("activity-text", text);
  pill.style.setProperty("--p", String(Math.round((ratio || 0) * 100)));
}

function personChips(perPerson) {
  const chips = [];
  for (const [slug, n] of Object.entries(perPerson || {})) {
    const person = state.people.find((row) => row.slug === slug) || { name: slug, slug };
    const chip = el("span", "count-chip");
    chip.append(avatar(person, "mono"));
    chip.append(el("span", "", person.name));
    chip.append(el("b", "", fmt.int(n)));
    chips.push(chip);
  }
  return chips;
}

function scanRatio(data) {
  const units = data.units;
  if (units && units.total > 0) return Math.min(1, units.done / units.total);
  const total = count(data.total);
  return total > 0 ? Math.min(1, count(data.done) / total) : null;
}

function renderScan(data) {
  state.scan = data;
  const total = count(data.total);
  const done = count(data.done);
  const ratio = scanRatio(data);
  if (data.state === "running") {
    lockForm(true);
    const phase = data.cancelRequested ? "cancelling" : data.phase || "start";
    let title;
    if (phase === "cancelling") title = "Durduruluyor…";
    else if (phase === "files") {
      title = total
        ? `Yüzler aranıyor · ${fmt.int(done)}/${fmt.int(total)} dosya · ${fmt.pct(ratio || 0)}`
        : "Yüzler aranıyor…";
    } else title = PHASE_LABEL[phase] || PHASE_LABEL.start;
    const stop = button("Durdur", "small", (event) => stopScan(event.currentTarget), "i-stop");
    if (data.cancelRequested) {
      stop.disabled = true;
      stop.textContent = "Durduruluyor…";
    }
    const lineParts = [];
    if (data.etaSeconds !== null && data.etaSeconds !== undefined && phase === "files") {
      lineParts.push(`${fmt.duration(data.etaSeconds)} kaldı`);
    } else if (phase === "files") lineParts.push("Süre hesaplanıyor…");
    if (data.matched) lineParts.push(`${fmt.int(data.matched)} dosyada eşleşme`);
    showJob({
      title,
      current: phase === "files" || phase === "hash" ? data.current : "",
      ratio,
      line: lineParts.join(" · "),
      actions: [stop],
      people: personChips(data.perPerson),
      indeterminate: ratio === null || phase === "hash" || phase === "secondLook" || phase === "models",
      thumb: data.lastMatch ? data.lastMatch.thumb : null,
    });
    setJobStatus("", null);
    const pct = ratio === null ? "…" : fmt.pct(ratio);
    setActivity(`Ayırma · ${pct}`, ratio || 0);
    document.title = ratio === null ? "Etkinlik arşivi" : `(${pct}) Etkinlik arşivi`;
    if (phase !== state.lastPhase) {
      state.lastPhase = phase;
      announce(title);
    } else if (ratio !== null) {
      const milestone = Math.floor(ratio * 4);
      if (milestone > state.lastMilestone && milestone > 0) {
        state.lastMilestone = milestone;
        announce(`Ayırma ${pct} tamamlandı.`);
      }
    }
    return;
  }

  lockForm(false);
  updatePreflight();
  finishJob();
  if (data.state === "idle") {
    if (state.wasRunning) {
      state.wasRunning = false;
      showJob({ title: "Ayırma bilgisi bulunamadı", current: "", ratio: null, line: "", actions: [], people: [], indeterminate: false });
      setJobStatus("Panel yeniden başlamış olabilir. Çıktı klasöründe kopyalananlar duruyor; sonuçları kontrol edin.", "warn");
    }
    return;
  }
  if (data.state === "error") {
    showJob({
      title: "Ayırma durdu",
      current: "",
      ratio,
      line: total ? `${fmt.int(done)}/${fmt.int(total)} dosya` : "",
      actions: [],
      people: personChips(data.perPerson),
      indeterminate: false,
      kind: "error",
    });
    const where = total ? ` ${fmt.int(done)}/${fmt.int(total)} dosyada` : "";
    setJobStatus(
      `${data.error || "Ayırma beklenmedik bir hatayla durdu."}${where ? ` Ayırma${where} durdu; o ana kadar kopyalananlar çıktı klasöründe.` : ""}`,
      "bad",
    );
    return;
  }
  const summary = data.summary;
  if (data.state === "cancelled") {
    const seen = summary ? summary.files : done;
    showJob({
      title: "Ayırma durduruldu",
      current: "",
      ratio: ratio === null ? 0 : ratio,
      line: total ? `${fmt.int(seen)}/${fmt.int(total)} dosyaya bakıldı` : "",
      actions: completionActions(data, summary),
      people: personChips(data.perPerson),
      indeterminate: false,
      kind: "cancelled",
    });
    setJobStatus(
      summary
        ? `${fmt.int(summary.matchedFiles)} dosyada eşleşme bulundu; kopyalananlar kişi klasörlerinde duruyor.`
        : "Hiçbir dosyaya bakılmadan durduruldu; önceki sonuçlar olduğu gibi kaldı.",
      "warn",
    );
    if (state.wasRunning) {
      state.wasRunning = false;
      announce("Ayırma durduruldu.");
      afterScan(data);
    }
    return;
  }
  if (data.state === "done" && summary) {
    const matched = count(summary.matchedFiles);
    const zero = matched === 0;
    showJob({
      title: zero ? "Ayırma bitti, kimse bulunamadı" : "Ayırma bitti",
      current: "",
      ratio: 1,
      line: `${plural(summary.files, "dosya")} tarandı · ${fmt.int(matched)} dosyada kayıtlı biri bulundu`,
      actions: completionActions(data, summary),
      people: personChips(data.perPerson),
      indeterminate: false,
      kind: "done",
    });
    const problems = [];
    if (summary.errors) problems.push(`${plural(summary.errors, "dosya")} okunamadı`);
    if (summary.unreadable) problems.push(`${plural(summary.unreadable, "videoda")} okunabilir kare yok`);
    if (summary.emptyFiles) problems.push(`${plural(summary.emptyFiles, "dosya")} 0 bayt`);
    if (zero) {
      setJobStatus(
        "Açı ekleyin ya da gelişmiş ayarlardan eşiği biraz düşürüp yeniden deneyin.",
        "warn",
      );
    } else if (problems.length) setJobStatus(`${problems.join(" · ")}. Ayrıntı sonuçların altında.`, "warn");
    else setJobStatus("", null);
    if (state.wasRunning) {
      state.wasRunning = false;
      announce(zero ? "Ayırma bitti, kimse bulunamadı." : `Ayırma bitti. ${fmt.int(matched)} dosyada kayıtlı biri bulundu.`);
      afterScan(data);
    }
  }
}

function completionActions(data, summary) {
  const actions = [];
  const output = data.destination || $("destination").value.trim();
  if (summary && summary.matchedFiles) {
    actions.push(
      button("Sonuçları gör", "small primary", () => {
        $("sonuclar").scrollIntoView({ block: "start" });
      }),
    );
  }
  if (output) {
    actions.push(button(folderLabel(), "small", () => openInFileManager({ output }), "i-open"));
  }
  return actions;
}

async function afterScan(data) {
  await reloadResults(data.destination || undefined);
  await reloadPeople();
}

async function poll() {
  window.clearTimeout(state.pollTimer);
  try {
    const data = await api("/api/tara");
    setConnection("ok");
    hideBanner();
    state.pollDelay = 500;
    renderScan(data);
    if (data.state === "running") state.pollTimer = window.setTimeout(poll, state.pollDelay);
  } catch (error) {
    if (error.status === 0) {
      setConnection("retry");
      setJobStatus("Bağlantı koptu. Ayırma arka planda sürüyor olabilir; yeniden bağlanılıyor…", "warn");
      state.pollDelay = Math.min(state.pollDelay * 2, 8000);
      state.pollTimer = window.setTimeout(poll, state.pollDelay);
      return;
    }
    setJobStatus(error.message, "bad");
    lockForm(false);
    updatePreflight();
  }
}

/* ---------- Open in Finder / Explorer ---------- */

async function openInFileManager(body) {
  try {
    await postJson("/api/klasoru-ac", body);
  } catch (error) {
    const path = error.body && error.body.path;
    if (path && navigator.clipboard) {
      toast(error.message, {
        kind: "bad",
        action: "Yolu kopyala",
        onAction: () => navigator.clipboard.writeText(path),
        duration: 12000,
      });
    } else toast(error.message, { kind: "bad" });
  }
}

/* ---------- Scroll spy ---------- */

function bindScrollSpy() {
  const links = [...document.querySelectorAll(".segnav a")];
  const sections = links.map((link) => $(link.getAttribute("href").slice(1))).filter(Boolean);
  if (!("IntersectionObserver" in window)) return;
  const visible = new Map();
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) visible.set(entry.target.id, entry.intersectionRatio);
      let best = null;
      for (const section of sections) {
        if (section.hidden) continue;
        const ratio = visible.get(section.id) || 0;
        if (!best || ratio > best.ratio) best = { id: section.id, ratio };
      }
      for (const link of links) {
        const current = best && best.ratio > 0 && link.getAttribute("href") === `#${best.id}`;
        if (current) link.setAttribute("aria-current", "true");
        else link.removeAttribute("aria-current");
      }
    },
    { rootMargin: "-80px 0px -40% 0px", threshold: [0, 0.2, 0.5, 1] },
  );
  for (const section of sections) observer.observe(section);
}

/* ---------- Wiring ---------- */

function bindOnce() {
  bindDrop();
  bindScrollSpy();
  $("banner-retry").addEventListener("click", () => {
    setConnection("retry");
    loadEverything();
  });
  $("enroll").addEventListener("submit", savePerson);
  $("open-enroll").addEventListener("click", openEnroll);
  $("hero-photo").addEventListener("click", openEnroll);
  $("enroll-close").addEventListener("click", closeEnroll);
  $("person-name").addEventListener("input", paintNameHint);
  $("scan").addEventListener("submit", submitScan);
  $("pool-start").addEventListener("click", startPool);
  for (const target of ["source", "destination"]) {
    $(`${target}-pick`).addEventListener("click", () => chooseFolder(target));
    $(target).addEventListener("input", () => {
      showFolder(target);
      if (target === "source") followSource($(target).value.trim());
    });
  }
  $("threshold").addEventListener("input", paintThreshold);
  $("interval").addEventListener("input", paintThreshold);
  $("advanced-reset").addEventListener("click", resetAdvanced);
  $("move").addEventListener("change", () => {
    paintMove();
    updatePreflight();
  });
  $("move-cancel").addEventListener("click", () => $("move-dialog").close());
  $("delete-cancel").addEventListener("click", () => $("delete-dialog").close());
  $("delete-ok").addEventListener("click", () => {
    if (deleteDecision) deleteDecision(true);
    deleteDecision = null;
    $("delete-dialog").close();
  });
  $("delete-dialog").addEventListener("close", () => {
    if (deleteDecision) deleteDecision(false);
    deleteDecision = null;
  });
  $("move-ok").addEventListener("click", () => {
    $("move-dialog").close();
    startScan();
  });
  $("who-all").addEventListener("click", () => setAllWho(true));
  $("who-none").addEventListener("click", () => setAllWho(false));
  $("sheet-close").addEventListener("click", () => $("person-sheet").close());
  $("sheet-open").addEventListener("click", () => {
    if (state.results && state.sheet.slug) {
      openInFileManager({ output: state.results.output, slug: state.sheet.slug });
    }
  });
  for (const chip of document.querySelectorAll("#sheet-filters .filter")) {
    chip.addEventListener("click", () => {
      state.sheet.filter = chip.dataset.filter;
      for (const other of document.querySelectorAll("#sheet-filters .filter")) {
        other.setAttribute("aria-pressed", other === chip ? "true" : "false");
      }
      renderSheet();
    });
  }
  $("lightbox-close").addEventListener("click", () => $("lightbox").close());
  $("lightbox").addEventListener("click", (event) => {
    if (event.target === event.currentTarget) $("lightbox").close();
  });
  for (const dialog of document.querySelectorAll("dialog.sheet, dialog.confirm")) {
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close();
    });
  }
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && state.scan && state.scan.state === "running") poll();
  });
  window.addEventListener("online", () => {
    if (!$("banner").hidden) loadEverything();
  });
  paintMove();
}

boot();
