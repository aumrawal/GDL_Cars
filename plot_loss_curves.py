import pdfplumber
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import re

PDF_PATH = "training_log.pdf"

def parse_training_log(pdf_path):
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for line in text.split("\n"):
                # Match lines starting with an epoch number
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    epoch = int(parts[0])
                except ValueError:
                    continue
                # Expect at least 13 numeric columns
                if len(parts) < 13:
                    continue
                try:
                    nums = [float(p) for p in parts[:13]]
                except ValueError:
                    continue
                rows.append(nums)
    return rows

rows = parse_training_log(PDF_PATH)

if not rows:
    raise RuntimeError("No data parsed from PDF. Check the PDF format.")

data = np.array(rows)
# columns: epoch, lr, train_total, train_cp, train_wss, train_cd, train_cl,
#          val_total, val_cp, val_wss, val_cd, val_cl, time_s
epochs      = data[:, 0]
train_total = data[:, 2]
train_cp    = data[:, 3]
train_wss   = data[:, 4]
train_cd    = data[:, 5]
train_cl    = data[:, 6]
val_total   = data[:, 7]
val_cp      = data[:, 8]
val_wss     = data[:, 9]
val_cd      = data[:, 10]
val_cl      = data[:, 11]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Training & Validation Loss Curves", fontsize=16, fontweight="bold")

loss_pairs = [
    ("Total Loss", train_total, val_total),
    ("Cp Loss",    train_cp,    val_cp),
    ("WSS Loss",   train_wss,   val_wss),
    ("Cd Loss",    train_cd,    val_cd),
]

for ax, (title, tr, vl) in zip(axes.flat, loss_pairs):
    ax.plot(epochs, tr, label="Train", linewidth=1.5, color="#1f77b4")
    ax.plot(epochs, vl, label="Val",   linewidth=1.5, color="#ff7f0e", linestyle="--")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.tight_layout()
out_path = "outputs/viz/loss_curves.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.show()
