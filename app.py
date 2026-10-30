"""BirdSong ID - Gradio demo: upload or record a bird call, get species prediction."""
from pathlib import Path

import numpy as np
import gradio as gr
import librosa

from features import mfcc_features, SR, CLIP_SEC
from numpy_nn import softmax

BASE = Path(__file__).parent


def load_model():
    z = np.load(BASE / "birdsong_mlp.npz", allow_pickle=True)
    dims = z["dims"]
    layers = []
    for i in range(len(dims) - 1):
        layers.append((z[f"W{i}"], z[f"b{i}"]))
    species = [str(s) for s in z["species"]]
    mu = np.load(BASE / "feat_mu.npy")
    sd = np.load(BASE / "feat_sd.npy")
    return layers, species, mu, sd


LAYERS, SPECIES, MU, SD = load_model()
PRETTY = [s.replace("_", " ").title() for s in SPECIES]


def forward(x):
    a = x
    for i, (W, b) in enumerate(LAYERS):
        z = a @ W + b
        a = np.maximum(0, z) if i < len(LAYERS) - 1 else z
    return softmax(a)


def predict(audio):
    """audio: (sr, numpy) tuple from Gradio. Returns {species: prob}."""
    if audio is None:
        return {}
    sr, y = audio
    if y.ndim > 1:
        y = y.mean(axis=1)
    # resample to model SR if needed
    if sr != SR:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR)
    y = y.astype(np.float32)
    # take up to CLIP_SEC from the middle (or pad)
    need = int(SR * CLIP_SEC)
    if len(y) > need:
        s = (len(y) - need) // 2
        y = y[s:s + need]
    elif len(y) < need:
        y = np.pad(y, (0, need - len(y)))
    if np.abs(y).mean() < 1e-5:
        return {"(silence detected — try again)": 1.0}
    feats = (mfcc_features(y) - MU) / SD
    probs = forward(feats[None])[0]
    order = np.argsort(probs)[::-1]
    return {PRETTY[i]: float(probs[i]) for i in order}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(sources=["upload", "microphone"], type="numpy",
                    label="Upload or record a bird call (a few seconds)"),
    outputs=gr.Label(num_top_classes=5, label="Predicted species"),
    title="🐦 BirdSong ID",
    description=("A neural network written from scratch in NumPy (dense layers, "
                 "ReLU, softmax, cross-entropy, mini-batch GD) classifies the "
                 "bird species from MFCC features. Top-5 predictions with "
                 "confidence scores."),
    examples=None,
)

if __name__ == "__main__":
    demo.launch()
