# 🐦 BirdSong ID

Upload or record a bird call → the app predicts which of **10 British bird
species** it is, with confidence scores. The neural network is **written from
scratch in NumPy** (dense layers, ReLU, softmax, cross-entropy, mini-batch
gradient descent) — no Keras/TensorFlow.

**Test accuracy: 84.1%** (target 79–85%+ ✅)

## Quick start

```bash
pip install -r requirements.txt

# train (uses X.npy/y.npy; or rebuild features from audio_raw/)
python train.py

# run the demo (unset proxies if Gradio fails to start)
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u no_proxy -u NO_PROXY \
  python app.py
# -> http://127.0.0.1:7860
```

## Deploy a free demo link

1. Push this folder to GitHub (or directly create a Hugging Face Space).
2. Hugging Face → Spaces → New Space → SDK **Gradio** → upload files.
3. You get a public link like `https://huggingface.co/spaces/<you>/birdsong-id`.

## How it works

See [REPORT.md](REPORT.md) for the full baby-steps explanation.

- **Audio:** 10 species' recordings → 5-second clips → 1,449 clips
- **Features:** 40 MFCCs + deltas, mean/std over time = 160 numbers per clip
- **Model:** NumPy MLP 160 → 64 (ReLU) → 10 (softmax), mini-batch GD, lr=0.05
- **Evaluated:** accuracy 84.1%, macro F1 0.819, confusion matrix, training curves
