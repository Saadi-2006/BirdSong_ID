"""Download bird-call recordings from Wikimedia Commons (robust version)."""
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

BASE = Path(__file__).parent / "audio_raw"
UA = {"User-Agent": "BirdSongID/1.0 (university coursework project)"}
API = "https://commons.wikimedia.org/w/api.php"

SPECIES = [
    ("Turdus merula", "blackbird"),
    ("Erithacus rubecula", "robin"),
    ("Troglodytes troglodytes", "wren"),
    ("Fringilla coelebs", "chaffinch"),
    ("Parus major", "great_tit"),
    ("Cyanistes caeruleus", "blue_tit"),
    ("Turdus philomelos", "song_thrush"),
    ("Chloris chloris", "greenfinch"),
    ("Luscinia megarhynchos", "nightingale"),
    ("Sylvia atricapilla", "blackcap"),
]


def api(params):
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(API + "?" + q, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def get_urls(scientific, limit=12):
    # broad search: just the scientific name, many results
    d = api({"action": "query", "list": "search",
             "srsearch": scientific, "srnamespace": 6,
             "srlimit": 50, "format": "json"})
    titles = [r["title"] for r in d["query"]["search"]
              if r["title"].lower().endswith((".ogg", ".mp3", ".wav", ".oga", ".flac", ".opus"))]
    out = []
    for i in range(0, len(titles), 10):
        chunk = titles[i:i + 10]
        d2 = api({"action": "query", "titles": "|".join(chunk),
                  "prop": "imageinfo", "iiprop": "url|size", "format": "json"})
        for p in d2["query"]["pages"].values():
            if "imageinfo" not in p:
                continue
            ii = p["imageinfo"][0]
            url, sz = ii.get("url", ""), ii.get("size", 0)
            title = p.get("title", "")
            if not url or not (30000 < sz < 20000000):
                continue
            # extension from the File: title (reliable)
            ext = title.rsplit(".", 1)[-1].lower()[:4]
            out.append((url, ext))
            if len(out) >= limit:
                break
        if len(out) >= limit:
            break
    return out


for scientific, slug in SPECIES:
    outdir = BASE / slug
    outdir.mkdir(exist_ok=True)
    have = len(list(outdir.glob("*")))
    if have >= 8:
        print(f"{slug}: already have {have}, skipping")
        continue
    try:
        urls = get_urls(scientific, limit=12)
    except Exception as e:
        print(f"{slug}: search failed: {e}")
        continue
    print(f"{slug}: found {len(urls)}")
    n = 0
    for j, (u, ext) in enumerate(urls):
        fp = outdir / f"{slug}_{j}.{ext}"
        if fp.exists():
            n += 1
            continue
        try:
            req = urllib.request.Request(u, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r, open(fp, "wb") as f:
                f.write(r.read())
            n += 1
        except Exception as e:
            print(f"  failed: {e}")
    print(f"  -> {slug}: {n} files")
print("done")
