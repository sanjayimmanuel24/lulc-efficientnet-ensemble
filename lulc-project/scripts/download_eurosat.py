"""
Robust EuroSAT-MS downloader with retry and resume support.

Run from inside lulc-project/:
    pip install requests
    python scripts/download_eurosat.py
"""
import sys
import zipfile
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Run:  pip install requests")
    sys.exit(1)

URL = "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSATallBands.zip"
DEST_DIR  = Path(__file__).parent.parent / "data" / "eurosat"
DEST_ZIP  = DEST_DIR / "EuroSATallBands.zip"
DEST_DIR.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 10
CHUNK = 1024 * 1024  # 1 MB chunks


def sizeof_fmt(num):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def download_with_resume(url, dest):
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=30)
            if r.status_code == 416:          # server says "range not satisfiable"
                print("File already fully downloaded.")
                return
            r.raise_for_status()

            total = int(r.headers.get("content-length", 0)) + existing
            mode  = "ab" if existing else "wb"

            print(f"Downloading {sizeof_fmt(total)} → {dest}")
            if existing:
                print(f"  Resuming from {sizeof_fmt(existing)} (attempt {attempt})")

            with open(dest, mode) as f:
                downloaded = existing
                last_print = time.time()
                for chunk in r.iter_content(chunk_size=CHUNK):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if time.time() - last_print > 5:       # print progress every 5 s
                            pct = downloaded / total * 100 if total else 0
                            print(f"  {sizeof_fmt(downloaded)} / {sizeof_fmt(total)}  ({pct:.1f}%)")
                            last_print = time.time()
            print(f"  Download complete: {sizeof_fmt(dest.stat().st_size)}")
            return

        except (requests.ConnectionError, requests.Timeout, Exception) as e:
            existing = dest.stat().st_size if dest.exists() else 0
            headers  = {"Range": f"bytes={existing}-"}
            wait = 5 * attempt
            print(f"  Error on attempt {attempt}: {e}")
            if attempt < MAX_RETRIES:
                print(f"  Retrying in {wait}s  (resumed from {sizeof_fmt(existing)})")
                time.sleep(wait)
            else:
                print("  Max retries reached. Run the script again — it will resume automatically.")
                sys.exit(1)


def extract(zip_path, dest_dir):
    print(f"\nExtracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.namelist()
        for i, member in enumerate(members, 1):
            z.extract(member, dest_dir)
            if i % 1000 == 0:
                print(f"  Extracted {i}/{len(members)} files")
    print(f"  Done — {len(members)} files extracted.")


def verify(dest_dir):
    tif_root = dest_dir / "ds/images/remote_sensing/otherDatasets/sentinel_2/tif"
    if not tif_root.exists():
        print(f"\nWarning: expected folder not found at {tif_root}")
        print("The zip may have a different internal structure — check data/eurosat/ manually.")
        return

    classes = sorted(p.name for p in tif_root.iterdir() if p.is_dir())
    total   = sum(len(list(p.glob("*.tif"))) for p in tif_root.iterdir() if p.is_dir())
    print(f"\n--- Verification ---")
    print(f"Root:    {tif_root}")
    print(f"Classes: {classes}")
    print(f"Files:   {total} .tif GeoTIFFs")
    if total >= 27000:
        print("All good — ready to train.")
    else:
        print(f"Expected ~27000 files, got {total} — extraction may be incomplete.")


# ── main ─────────────────────────────────────────────────────────────────────
if DEST_ZIP.exists() and DEST_ZIP.stat().st_size > 100_000_000:
    # Zip already partially or fully downloaded
    print(f"Found existing zip ({sizeof_fmt(DEST_ZIP.stat().st_size)}) — checking if complete.")

download_with_resume(URL, DEST_ZIP)

tif_root = DEST_DIR / "ds/images/remote_sensing/otherDatasets/sentinel_2/tif"
if not tif_root.exists():
    extract(DEST_ZIP, DEST_DIR)
else:
    print("\nAlready extracted — skipping.")

verify(DEST_DIR)
