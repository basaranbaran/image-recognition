"""Command line for enrollment and scanning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated
from urllib.parse import urlencode

import typer
from rich.console import Console
from rich.table import Table

from recognize.clothing import hint_for
from recognize.engine import FaceEngine, GalleryRecognizer
from recognize.enroll import enroll_faces
from recognize.errors import AppError
from recognize.gallery import delete_person, find_person, load_people
from recognize.match import COSINE_THRESHOLD, DEFAULT_SAMPLE_SECONDS, ScanOptions
from recognize.media import (
    ENROLL_EXTENSIONS,
    FORMATS_LINE,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    decode_image,
    resize_max_edge,
)
from recognize.messages import (
    ANGLE_GUIDANCE,
    CLOTHING_NOTE,
    ONE_PHOTO_NOTE,
    THRESHOLD_HINT,
    VIDEO_INTERVAL_HINT,
)
from recognize.paths import (
    default_output_dir,
    default_pool_dir,
    event_source_dir,
    models_dir,
    people_dir,
)
from recognize.pool import build_pool
from recognize.scan import ScanHooks, run_scan
from recognize.store import archive_report, remember_scan
from recognize.textutil import clean_person_name

app = typer.Typer(
    help="Etkinlik fotoğraf ve videolarında tanıdığınız kişileri klasörlere ayırır.",
    no_args_is_help=True,
)
console = Console()

TANIT_HELP = (
    f"{ANGLE_GUIDANCE} Tek önden kare de olur; yan açılar daha sık kaçar. "
    f'{CLOTHING_NOTE} Örnek: recognize tanit "Ayşe Yılmaz" kareler/*.jpg --kiyafet sari'
)


@app.command("tanit", help=TANIT_HELP, short_help="Fotoğraflardan bir kişiyi tanıtır.")
def enroll_command(
    ad: Annotated[str, typer.Argument(help="Kişinin adı")],
    fotolar: Annotated[list[Path], typer.Argument(help="Tanıtım fotoğrafları", exists=True)],
    kiyafet: Annotated[
        str | None,
        typer.Option(
            "--kiyafet",
            help="Belirgin renk: sari, kirmizi, mavi, yesil, siyah, beyaz, yok",
        ),
    ] = None,
    en_buyuk: Annotated[
        bool,
        typer.Option("--en-buyuk", help="Birden fazla yüz varsa en büyük olanı al"),
    ] = False,
) -> None:
    """Enroll reference faces. Several angles are kept separately."""
    console.print(ANGLE_GUIDANCE)
    console.print(ONE_PHOTO_NOTE)
    try:
        engine = _engine()
        samples, skipped = _samples_from_files(engine, fotolar, en_buyuk)
        clothing = None
        update = False
        if kiyafet is not None:
            clothing = hint_for(kiyafet)
            update = True
            console.print(CLOTHING_NOTE)
        if not samples:
            raise AppError("Hiçbir fotoğraftan yüz kaydedilemedi.")
        person, warning = enroll_faces(
            ad,
            samples,
            clothing,
            update_clothing=update,
            root=people_dir(),
        )
    except AppError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    hint = "yok" if person.clothing is None else person.clothing.label
    console.print(
        f"Kaydedildi: {person.name} · {len(person.references)} referans · kıyafet: {hint}"
    )
    if warning:
        console.print(f"[yellow]{warning}[/yellow]")
    if skipped:
        console.print("[yellow]Bazı fotoğraflar atlandı. Mesajlara bakın.[/yellow]")
        raise typer.Exit(code=1)


@app.command("kamera")
def camera_command(
    ad: Annotated[str, typer.Argument(help="Kaydedilecek kişinin adı")],
    port: Annotated[int, typer.Option("--port")] = 8765,
) -> None:
    """Tarayıcıda kamerayla tanıtma sayfasını açar. Kayıt, dönüş bitince yazılır."""
    try:
        cleaned = clean_person_name(ad)
    except AppError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    url = f"http://127.0.0.1:{port}/kamera?{urlencode({'ad': cleaned})}"
    console.print(url)
    console.print("Yavaşça sola, sağa, yukarı ve aşağı dönün. Tek kare yetmez.")
    console.print("Bu kamerada derinlik sensörü yok. Kayıt, dönüş boyunca alınan yüz yapısıdır.")
    console.print("Fotoğraflar bu bilgisayardan çıkmaz. Kıyafet rengi eklenmez.")
    console.print("Panel kapalıysa önce: uv run recognize panel --tarayicisiz")
    import webbrowser

    try:
        webbrowser.open(url)
    except OSError:
        console.print("Tarayıcı açılmadı. Adresi kendiniz açın.")


@app.command("liste")
def list_command() -> None:
    """Kayıtlı kişileri göster."""
    people = load_people()
    if not people:
        console.print("Henüz kimse tanıtılmadı.")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Ad")
    table.add_column("Referans", justify="right")
    table.add_column("Kıyafet")
    for person in people:
        table.add_row(
            person.name,
            str(len(person.references)),
            "—" if person.clothing is None else person.clothing.label,
        )
    console.print(table)


@app.command("sil")
def remove_command(ad: Annotated[str, typer.Argument(help="Ad veya kısa ad")]) -> None:
    """Kişiyi ve referanslarını sil."""
    try:
        person = delete_person(ad)
    except AppError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    console.print(f"Silindi: {person.name}")


@app.command(
    "tara",
    help=(
        "Etkinlik klasörünü tarar, eşleşen dosyaları kişi klasörlerine kopyalar. "
        f"{THRESHOLD_HINT} {VIDEO_INTERVAL_HINT}"
    ),
    short_help="Etkinlik klasörünü tarar, eşleşenleri kişi klasörlerine kopyalar.",
)
def scan_command(
    kaynak: Annotated[
        Path,
        typer.Argument(help="Fotoğraf ve videoların klasörü", exists=True, file_okay=False),
    ],
    cikti: Annotated[
        Path | None,
        typer.Option("--cikti", help="Kişi klasörlerinin yazılacağı yer"),
    ] = None,
    esik: Annotated[
        float,
        typer.Option("--esik", help="Kosinüs eşiği. 0.36 kalabalıkta başka yüzleri de seçer"),
    ] = COSINE_THRESHOLD,
    aralik: Annotated[
        float,
        typer.Option("--aralik", help="Videoda kaç saniyede bir kareye bakılacağı"),
    ] = DEFAULT_SAMPLE_SECONDS,
    tasi: Annotated[
        bool,
        typer.Option("--tasi", help="Kopyalamak yerine kaynak dosyayı sil"),
    ] = False,
    min_kare: Annotated[
        int,
        typer.Option("--min-kare", help="Videoda kaç karede görünmeli"),
    ] = 2,
    kisi: Annotated[
        list[str] | None,
        typer.Option("--kisi", help="Yalnızca bu kişi için ayır. Birden çok kez verilebilir"),
    ] = None,
) -> None:
    """Fotoğraf ve videoları kişi klasörlerine ayır."""
    destination = (cikti or default_output_dir()).expanduser()
    options = ScanOptions(threshold=esik, sample_seconds=aralik, min_video_hits=min_kare, move=tasi)
    console.print(THRESHOLD_HINT)
    console.print(VIDEO_INTERVAL_HINT)
    console.print("Dosyalar kopyalanır. Aynı karede birden fazla kişi varsa her klasöre yazılır.")
    try:
        people = load_people()
        selected = people
        if kisi:
            selected = []
            for name in kisi:
                person = find_person(name)
                if person is None:
                    raise AppError(f"Böyle bir kişi yok: {name}")
                selected.append(person)
        engine = _engine()
        recognizer = GalleryRecognizer(
            engine, people, options, targets={person.slug for person in selected}
        )

        def progress(done: int, total: int, label: str) -> None:
            if total:
                console.print(f"[dim]{done}/{total}[/dim] {label}", overflow="ellipsis")

        report = run_scan(
            kaynak,
            destination,
            recognizer,
            selected,
            options,
            progress,
            hooks=ScanHooks(archive=archive_report),
        )
    except AppError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    summary = report["summary"]
    if not isinstance(summary, dict):
        return
    try:
        # The panel opens on the most recent scan, including ones started here.
        remember_scan(destination, summary, str(report.get("scannedAt") or ""))
    except OSError:
        pass
    console.print(
        f"Bitti. {summary['files']} dosya, {summary['matchedFiles']} eşleşme, "
        f"{summary['errors']} hata. Rapor: {destination / 'rapor.json'}"
    )
    if summary.get("duplicates"):
        console.print(
            f"{summary['duplicates']} dosya başka bir dosyanın birebir kopyası; bir kez yazıldı."
        )
    if summary.get("emptyFiles"):
        console.print(
            f"[yellow]{summary['emptyFiles']} dosya boş (0 bayt) ve atlandı. "
            "Kopyalama yarım kalmış olabilir.[/yellow]"
        )
    people_summary = summary.get("people")
    if not isinstance(people_summary, list):
        return
    for row in people_summary:
        if not isinstance(row, dict):
            continue
        assisted = int(row["clothingAssisted"])
        extra = f", {assisted} tanesi kıyafet desteğiyle" if assisted else ""
        learned = int(row.get("learned") or 0)
        if learned:
            extra += f", {learned} tanesi etkinlikteki net karelerinden öğrenilerek"
        small = int(row.get("smallFaces") or 0)
        if small:
            extra += f", {small} tanesinde yüz çok küçük (raporda smallFace), elle bakın"
        label = row.get("clothingLabel")
        worn = f" ({label})" if isinstance(label, str) and label else ""
        console.print(f"{row['name']}{worn}: {row['files']} dosya{extra}")


@app.command("havuz")
def pool_command(
    kaynak: Annotated[
        Path | None,
        typer.Argument(help="Etkinlik klasörü. Boşsa en çok fotoğraf olan klasör", file_okay=False),
    ] = None,
    cikti: Annotated[
        Path | None,
        typer.Option("--cikti", help="Tekil havuzun yazılacağı yer"),
    ] = None,
) -> None:
    """Her farklı fotoğraf ve videodan birer tane içeren tekil havuz kurar. Kaynağa dokunmaz."""
    source = kaynak or event_source_dir(prefer_pool=False)
    if source is None:
        console.print("[red]Etkinlik klasörü bulunamadı. Kaynak klasörü verin.[/red]")
        raise typer.Exit(1)
    destination = (cikti or default_pool_dir()).expanduser()
    try:
        report = build_pool(source, destination)
    except AppError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    summary = report["summary"]
    if not isinstance(summary, dict):
        return
    console.print(
        f"Havuz hazır: {summary['photos']} fotoğraf, {summary['videos']} video. {destination}"
    )
    console.print(
        f"Alınmayanlar: {summary['duplicates']} kopya, {summary['emptyFiles']} boş, "
        f"{summary['broken']} açılmayan dosya. Ayrıntı: {destination / 'havuz.json'}"
    )


@app.command("bak")
def inspect_command(
    dosya: Annotated[Path, typer.Argument(exists=True, help="Tek fotoğraf veya video")],
) -> None:
    """Tek dosyada kim çıkıyor, ona bak."""
    options = ScanOptions()
    try:
        people = load_people()
        if not people:
            raise AppError("Henüz kimse tanıtılmadı.")
        engine = _engine()
        recognizer = GalleryRecognizer(engine, people, options)
        suffix = dosya.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            inspection = recognizer.inspect_video(dosya)
        elif suffix in IMAGE_EXTENSIONS:
            inspection = recognizer.inspect_image(dosya)
        else:
            raise AppError(f"Bu dosya türüne bakılamaz. Desteklenen: {FORMATS_LINE}.")
    except AppError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    if not inspection.hits:
        console.print("Kayıtlı kimse eşleşmedi.")
    for hit in inspection.hits:
        assisted = " · kıyafet desteği" if hit.clothing_assisted else ""
        console.print(f"{hit.name}: {hit.score:.3f} ({hit.samples} kare){assisted}")
    for hit in inspection.uncertain:
        console.print(f"Belirsiz: {hit.name} {hit.score:.3f}")


@app.command("panel")
def panel_command(
    port: Annotated[int, typer.Option("--port")] = 8765,
    tarayici: Annotated[bool, typer.Option("--tarayici/--tarayicisiz")] = True,
) -> None:
    """Yerel paneli aç. Yalnızca bu bilgisayardan erişilir."""
    import threading
    import webbrowser

    import uvicorn

    url = f"http://127.0.0.1:{port}"
    console.print(f"Panel: {url}")
    console.print("Fotoğraflar bu bilgisayardan çıkmaz.")
    if tarayici:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    uvicorn.run(
        "recognize.web:app",
        host="127.0.0.1",
        port=port,
        log_level="info",
        proxy_headers=False,
    )


def _engine() -> FaceEngine:
    return FaceEngine(models_dir(), log=lambda text: console.print(text))


def _samples_from_files(
    engine: FaceEngine,
    paths: list[Path],
    largest: bool,
) -> tuple[list[tuple[object, float, bytes, str]], bool]:
    samples: list[tuple[object, float, bytes, str]] = []
    skipped = False
    for path in paths:
        if path.suffix.lower() not in ENROLL_EXTENSIONS:
            console.print(f"[yellow]Atlandı (jpg, png, webp veya heic değil): {path.name}[/yellow]")
            skipped = True
            continue
        image = resize_max_edge(decode_image(path.read_bytes()), 2560)
        faces = [face for face in engine.detect(image, 0.5) if min(face.width, face.height) >= 24]
        if not faces:
            console.print(f"[yellow]Yüz bulunamadı: {path.name}[/yellow]")
            skipped = True
            continue
        if len(faces) > 1 and not largest:
            console.print(
                f"[yellow]{path.name} içinde {len(faces)} yüz var. "
                "Doğru yüzü seçmek için paneli kullanın ya da --en-buyuk verin.[/yellow]"
            )
            skipped = True
            continue
        face = max(faces, key=lambda item: item.width * item.height)
        samples.append(
            (engine.embed(image, face), face.score, engine.crop_jpeg(image, face), path.name)
        )
    return samples, skipped


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    app()
