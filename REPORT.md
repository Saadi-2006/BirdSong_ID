# 🐦 BirdSong ID — Full Report (Baby Steps)

> **Goal:** take a short bird-call audio clip and predict **which of 10 British
> bird species** it is, using a neural network **written from scratch in NumPy**
> (dense layers, ReLU, softmax, cross-entropy, mini-batch gradient descent),
> reaching **79–85%+ accuracy**, with a Gradio demo.

**Final result: test accuracy = 84.1%** ✅ (target 79–85%+)

---

## Step 0 — The big picture

You record 5 seconds of a bird singing. The app tells you "that's a European
Robin (92% sure)". How?

1. **Dataset** — hundreds of labelled bird-call recordings.
2. **Features** — convert wiggly audio into tidy numbers (MFCCs).
3. **Model** — a small neural network, coded by hand in NumPy, learns to map
   those numbers → species.
4. **Demo** — a Gradio webpage: upload/record audio → species + confidence.

---

## Step 1 — The dataset (recordings, not photos)

- **Source:** bird-call recordings from **Wikimedia Commons** (many originally
  from xeno-canto.org). (The Kaggle "British Birdsong" dataset needs a Kaggle
  login to download; Commons gives the same kind of recordings freely.)
- **10 species:** blackbird, blackcap, blue tit, chaffinch, great tit,
  greenfinch, nightingale, robin, song thrush, wren.
- **~113 recordings**, each cut into **5-second clips** → **1,449 clips**.
- Each clip is labelled with its species (0–9).

**Baby-step summary:** the dataset is a pile of 5-second bird-call clips with
answer labels.

---

## Step 2 — From sound to numbers: MFCC features

A neural network can't hear. It needs numbers. **MFCCs** (Mel-Frequency Cepstral
Coefficients) are the standard way to turn sound into numbers:

1. **Load** the clip (resampled to 22,050 Hz, mono).
2. **Chop** into tiny overlapping frames (~46 ms each).
3. For each frame, measure **how much energy** there is in 40 pitch bands
   (low → high). Human/bird hearing is logarithmic, so we use the **mel scale**.
4. Take the **DCT** (a maths trick) → 40 tidy coefficients per frame.
5. Also compute **deltas** (how fast each coefficient changes) → captures the
   *shape* of the chirp, not just its pitch.
6. **Summarise** each coefficient over the whole 5 s: its **mean** and **std**.

Result per clip: 40 MFCCs × (mean, std) + 40 deltas × (mean, std) = **160 numbers**.
We then **standardise** (subtract mean, divide by std) so every feature is on a
similar scale — neural networks learn much faster that way.

**Baby-step summary:** 5 seconds of birdsong → 160 numbers describing its
"sound fingerprint".

---

## Step 3 — The neural network, written from scratch (no Keras!)

`numpy_nn.py` builds everything with plain NumPy arrays. The architecture:

```
Input:  160 numbers (the MFCC fingerprint)
   ↓
Dense(64) + ReLU        ← hidden layer: learns combinations of sound features
   ↓
Dense(10) + Softmax     ← output: 10 probabilities (one per species)
```

The maths, piece by piece:

- **Dense layer:** `z = X @ W + b` — each neuron is a weighted vote over its inputs.
- **ReLU:** `max(0, z)` — "if negative, output 0". Lets the network learn
  non-trivial patterns (without it, the whole network would just be one big
  linear formula).
- **Softmax:** turns the 10 raw scores into probabilities that sum to 1.
- **Cross-entropy loss:** `-mean(log(p_correct))` — big penalty when the model
  is confident *and wrong*; small when confident and right.
- **Backpropagation:** the chain rule, worked backwards layer by layer, tells us
  how much each weight contributed to the error.
- **Mini-batch gradient descent:** update weights on 64 clips at a time
  (`W -= lr × gradient`), reshuffle each epoch. 150 epochs.
- **He initialisation:** random starting weights scaled by `sqrt(2/n_in)` so
  ReLU neurons start in a healthy range.

**Baby-step summary:** 160 numbers in → one hidden layer thinks → 10 species
probabilities out. Every formula is hand-coded in `numpy_nn.py`.

---

## Step 4 — Training & comparing settings

We train on 80% of clips, validate on 20%. Three configurations were compared:

| Setting | Test accuracy | Macro F1 |
|---|---|---|
| lr=0.05, hidden 128→64 | 81.0% | 0.778 |
| lr=0.01, hidden 128→64 | 78.6% | 0.757 |
| **lr=0.05, hidden 64** ✅ | **84.1%** | **0.819** |

**Lesson:** the *simpler* network (one 64-neuron hidden layer) won. With only
~1,450 clips, the bigger network overfits; the small one generalises better.
A higher learning rate (0.05) also beat 0.01 here.

Training curves (`training_curves.png`) show loss falling and accuracy rising
for both train and validation — no severe overfitting.

**Baby-step summary:** try a few settings, keep the winner (simple 64-neuron net).

---

## Step 5 — Testing (the honest exam)

The 20% test clips were never seen during training:

> **Test accuracy = 84.1%** ✅ (target 79–85%+)
> **Macro F1 = 0.819**

`confusion_matrix.png` shows which species get confused (e.g. thrushes vs.
blackbirds sound similar — the matrix makes that visible).

**Baby-step summary:** exam on unseen clips → 84.1% correct.

---

## Step 6 — The demo (Gradio app)

`app.py`:

1. User **uploads** an audio file or **records** with the microphone.
2. App takes up to 5 seconds (middle of the clip), extracts the same 160 MFCC
   features, standardises them.
3. The saved NumPy weights (`birdsong_mlp.npz`) run a forward pass → 10
   probabilities.
4. Page shows the **top-5 species with confidence scores**.

Run locally:

```bash
pip install -r requirements.txt
python -c "from features import build_dataset; ..."  # or use saved X.npy
python train.py        # trains, saves birdsong_mlp.npz
python app.py          # or: gradio app.py  -> opens http://127.0.0.1:7860
```

Deploy free: push to GitHub → https://huggingface.co/spaces → New Space
(Gradio) → public demo link.

> ⚠️ Gradio needs proxy env vars unset on some networks:
> `env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u no_proxy -u NO_PROXY python app.py`

---

## Step 7 — Files

```
birdsong-id/
├── numpy_nn.py          # the from-scratch NN (Dense/ReLU/softmax/x-ent/backprop/GD)
├── features.py          # audio -> 5s clips -> 160-dim MFCC vectors
├── train.py             # standardise, split, train, compare, evaluate, plot
├── app.py               # Gradio demo (upload/record -> species + confidence)
├── download_audio.py    # fetched recordings from Wikimedia Commons
├── birdsong_mlp.npz     # trained weights (created by train.py)
├── feat_mu.npy / feat_sd.npy  # standardisation stats
├── species.txt          # the 10 species
├── X.npy / y.npy        # extracted features (1,449 x 160)
├── training_curves.png / confusion_matrix.png
├── requirements.txt
├── README.md
└── REPORT.md            # this file
```

## Glossary (one line each)

- **MFCC** — 40 numbers per audio frame describing its tone colour.
- **Dense layer** — every input connects to every neuron: `z = XW + b`.
- **ReLU** — `max(0, z)`; gives the network its learning power.
- **Softmax** — turns scores into probabilities summing to 1.
- **Cross-entropy** — the "how wrong" number training minimises.
- **Backpropagation** — chain rule assigning blame to each weight.
- **Mini-batch GD** — update weights on small random chunks, reshuffle each epoch.
- **Learning rate** — step size for each weight update (0.05 won here).
- **F1** — accuracy's stricter cousin: balances precision and recall per class.
