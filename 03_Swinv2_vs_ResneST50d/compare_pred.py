# compare_preds_better.py
import pandas as pd, numpy as np

GT   = "test.csv"
SWIN = "test_preds_epoch20_swin.csv"
RES  = "test_preds_epoch20_resneST50d.csv"

# --- Tunables ---
THRESH = 0.5        # binarization for both models
TOPN = 20           # for the Top-ΔF1 list
# "Clear win" criteria (more realistic than 0.80/0.20/0.40)
F1_SW_MIN  = 0.35
F1_RN_MAX  = 0.10
DELTA_MIN  = 0.20
# Label-wise improvement: at least K true positives that Swin got and resneST50d missed
K_MIN_TP_SW_ONLY = 1   # set 2 for stricter

# --- Load & align ---
gt = pd.read_csv(GT)
labels = list(gt.columns[4:])
df = gt[["file"]+labels] \
      .merge(pd.read_csv(SWIN), on="file", suffixes=("", "_swin")) \
      .merge(pd.read_csv(RES),  on="file", suffixes=("", "_res"))

y_true = df[labels].to_numpy(dtype=np.uint8, copy=False)
sw_val = df[[c+"_swin" for c in labels]].to_numpy(dtype=np.float32, copy=False)
rs_val = df[[c+"_res"  for c in labels]].to_numpy(dtype=np.float32, copy=False)

y_s = (sw_val >= THRESH).astype(np.uint8, copy=False)
y_r = (rs_val >= THRESH).astype(np.uint8, copy=False)

def f1_row(t, p, eps=1e-9):
    tp = (t & p).sum(1); fp = ((1 - t) & p).sum(1); fn = (t & (1 - p)).sum(1)
    return (2 * tp) / (2 * tp + fp + fn + eps)

f1_s, f1_r = f1_row(y_true, y_s), f1_row(y_true, y_r)
delta = f1_s - f1_r

# 1) Top ΔF1 list (always non-empty if rows exist)
top = (pd.DataFrame({"file": df["file"], "f1_swin": f1_s, "f1_resneST50d": f1_r, "delta": delta})
       .sort_values("delta", ascending=False)
       .head(TOPN))
top.to_csv("swin_better_than_resneST50d.csv", index=False)
print("[✓] Saved swin_better_than_resneST50d.csv")

# 2) “Clear wins” with more realistic cutoffs
mask_clear = (f1_s >= F1_SW_MIN) & (f1_r <= F1_RN_MAX) & (delta >= DELTA_MIN)
clear = (pd.DataFrame({"file": df["file"], "f1_swin": f1_s, "f1_resneST50d": f1_r, "delta": delta})
         .loc[mask_clear]
         .sort_values("delta", ascending=False))
clear.to_csv("swin_good_resneST50d_bad.csv", index=False)
print(f"[✓] Saved swin_good_resneST50d_bad.csv (n={len(clear)})")

# 3) Label-wise: Swin true positives that resneST50d missed (TP_sw & FN_res) ≥ K
tp_sw_only = ((y_true == 1) & (y_s == 1) & (y_r == 0)).sum(axis=1)
mask_tp = tp_sw_only >= K_MIN_TP_SW_ONLY

# Optional: also show *which* labels these are (short string)
def labels_from_mask(row_mask):
    return [labels[i] for i, v in enumerate(row_mask) if v]

rows = []
for i in np.where(mask_tp)[0]:
    sw_only_mask = (y_true[i] == 1) & (y_s[i] == 1) & (y_r[i] == 0)
    rows.append({
        "file": df.iloc[i]["file"],
        "tp_swin_missed_by_resneST50d": int(tp_sw_only[i]),
        "f1_swin": float(f1_s[i]),
        "f1_resneST50d": float(f1_r[i]),
        "delta": float(delta[i]),
        "labels_swin_tp_resneST50d_fn": ", ".join(labels_from_mask(sw_only_mask))[:120]
    })

tp_report = pd.DataFrame(rows).sort_values(["tp_swin_missed_by_resneST50d","delta"], ascending=[False, False])
tp_report.to_csv("swin_tp_missed_by_resneST50d.csv", index=False)
print(f"[✓] Saved swin_tp_missed_by_resneST50d.csv (n={len(tp_report)}) with K_MIN_TP_SW_ONLY={K_MIN_TP_SW_ONLY}")
