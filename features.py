"""BirdSong ID - feature extraction: audio -> fixed MFCC clips -> feature vectors.

Each recording is cut into CLIP_SEC-second clips. For every clip we compute
N_MFCC mel-frequency cepstral coefficients per frame, then summarise each
coefficient by its mean and std across time -> a fixed 2*N_MFCC vector.
This is what the NumPy MLP is trained on.
"""
import numpy as np
import librosa
from pathlib import Path

SR = 22050          # resample everything to this
CLIP_SEC = 5.0      # fixed clip length
N_MFCC = 40         # MFCC coefficients per frame
N_FFT = 2048
HOP = 512


def load_clips(path, clip_sec=CLIP_SEC, sr=SR):
    """Load audio file, return list of fixed-length mono clips (numpy)."""
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
    except Exception as e:
        print(f"  skip {Path(path).name}: {e}")
        return []
    if y.size < sr * 1.0:  # too short to be useful
        return []
    clip_len = int(sr * clip_sec)
    clips = []
    # non-overlapping windows; pad the last one if >= 50% full
    for s in range(0, len(y), clip_len):
        c = y[s:s + clip_len]
        if len(c) < clip_len * 0.5:
            break
        if len(c) < clip_len:
            c = np.pad(c, (0, clip_len - len(c)))
        # skip near-silent clips
        if np.abs(c).mean() < 1e-4:
            continue
        clips.append(c)
    return clips


def mfcc_features(clip, sr=SR, n_mfcc=N_MFCC):
    """40 MFCCs -> mean + std over time = 80-dim vector."""
    m = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=n_mfcc,
                             n_fft=N_FFT, hop_length=HOP)
    # delta features add temporal dynamics (optional but cheap)
    d1 = librosa.feature.delta(m)
    feats = np.concatenate([m.mean(axis=1), m.std(axis=1),
                            d1.mean(axis=1), d1.std(axis=1)])
    return feats.astype(np.float32)  # 160-dim


def build_dataset(audio_root, clip_sec=CLIP_SEC):
    """Walk audio_root/<species>/*.audio -> X (N,160), y (N,), species list."""
    audio_root = Path(audio_root)
    species = sorted([d.name for d in audio_root.iterdir() if d.is_dir()])
    Xs, ys = [], []
    for si, sp in enumerate(species):
        n_clips = 0
        for fp in sorted((audio_root / sp).glob("*")):
            if fp.suffix.lower() not in (".mp3", ".ogg", ".oga", ".wav", ".flac", ".opus", ".m4a"):
                continue
            for clip in load_clips(fp, clip_sec):
                Xs.append(mfcc_features(clip))
                ys.append(si)
                n_clips += 1
        print(f"{sp}: {n_clips} clips", flush=True)
    X = np.stack(Xs)
    y = np.array(ys)
    return X, y, species
