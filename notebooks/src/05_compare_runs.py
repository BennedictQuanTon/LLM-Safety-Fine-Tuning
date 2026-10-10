# %% [markdown]
# # 05 — Benchmark nhiều lần fine-tune (Kaggle CPU)
#
# **Cài đặt Kaggle:** Accelerator = **None** (CPU) · Internet = Off cũng được ·
# Add Input → output của **mọi** bản notebook 02 cần so (và notebook 03 tương ứng nếu đã đánh giá).
#
# Notebook tự tìm mọi `run_record.json` (từ 02) và `eval_record.json` (từ 03), ghép theo `run_id`, rồi trả lời:
# **specs nào đã thay đổi so với baseline, và thay đổi đó ảnh hưởng thế nào** đến loss, độ ổn định, tốc độ, bộ nhớ và hành vi.
#
# **Output** (`/kaggle/working/compare/`):
# - `runs_comparison.md` — bảng xếp hạng, bảng ảnh hưởng so với baseline, lần chạy được khuyến nghị
# - `runs_comparison.csv` — mỗi lần chạy một hàng: specs + sức khỏe + đánh giá
# - `compare_curves.png` — chồng đường `eval_loss` / `train_loss` của mọi lần chạy
# - `compare_summary.png` — so loss tốt nhất, overfit, cải thiện, tốc độ, VRAM, thời gian
# - `compare_hparam_effect.png` — từng siêu tham số thay đổi vs `eval_loss` tốt nhất
# - `compare_eval_metrics.png` — bản đồ nhiệt chỉ số hành vi (nếu có `eval_record.json`)

# %%
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONFIG = dict(
    baseline_run_id=None,
    out_dir="/kaggle/working/compare",
)
OUT = Path(CONFIG["out_dir"])
OUT.mkdir(parents=True, exist_ok=True)

SPEC_KEYS = [
    "model_name", "lora_r", "lora_alpha", "lora_dropout", "target_modules", "learning_rate", "lr_scheduler_type",
    "warmup_ratio", "weight_decay", "max_grad_norm", "per_device_train_batch_size", "gradient_accumulation_steps",
    "num_train_epochs", "max_steps", "max_seq_length", "max_eval_samples", "seed",
]
HEALTH_KEYS = [
    "eval_loss_first", "eval_loss_best", "eval_loss_final", "improvement_pct", "overfit_rise_pct", "generalization_gap",
    "steps_to_90pct", "loss_spikes", "nan_or_inf_steps", "grad_norm_p95", "grad_norm_max", "tokens_per_sec_mean", "vram_peak_gb",
]
SUMMARY_PANELS = [
    ("eval_loss_best", "Best eval_loss (↓)"),
    ("improvement_pct", "Improvement % (↑)"),
    ("overfit_rise_pct", "Overfit rise % (↓)"),
    ("tokens_per_sec_mean", "Throughput tokens/s (↑)"),
    ("vram_peak_gb", "Peak VRAM GB (↓)"),
    ("runtime_min", "Runtime min (↓)"),
]

# %% [markdown]
# ## Gom các lần chạy

# %%
def load_runs():
    runs = {}
    for path in sorted(glob.glob("/kaggle/input/**/run_record.json", recursive=True)):
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
        if rec.get("schema_version") != 1:
            print(f"Bỏ qua {path}: schema_version = {rec.get('schema_version')} không hỗ trợ")
            continue
        if rec["run_id"] in runs:
            continue
        log_path = Path(path).with_name("training_log.csv")
        rec["_log"] = pd.read_csv(log_path) if log_path.exists() else None
        runs[rec["run_id"]] = rec
    return runs


def load_evals():
    evals = {}
    for path in sorted(glob.glob("/kaggle/input/**/eval_record.json", recursive=True)):
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
        evals.setdefault(rec["run_id"], rec)
    return evals


RUNS, EVALS = load_runs(), load_evals()
if not RUNS:
    raise FileNotFoundError("Không thấy run_record.json nào trong /kaggle/input. Hãy Add Input → output của các bản notebook 02.")
print(f"{len(RUNS)} lần chạy, {len(EVALS)} bản đánh giá")
for rid in RUNS:
    print(f"  {'✓ eval' if rid in EVALS else '  ----'}  {rid}")
if len(RUNS) < 2:
    print("⚠️ Chỉ có 1 lần chạy: vẫn xuất báo cáo, nhưng cần ≥ 2 lần chạy để so sánh specs.")

# %% [markdown]
# ## Bảng tổng hợp: mỗi lần chạy một hàng

# %%
LOWER_IS_BETTER_EVAL = set()


def flatten(rid, rec):
    cfg = rec["config"]
    row = dict(run_id=rid, run_name=rec["run_name"], created_at=rec["created_at"], diagnosis=rec["diagnosis"]["code"])
    for k in SPEC_KEYS:
        v = cfg.get(k)
        row[k] = ",".join(v) if isinstance(v, list) else v
    row.update(
        effective_batch_size=rec["derived"]["effective_batch_size"],
        trainable_pct=rec["derived"]["trainable_pct"],
        n_train=rec["data"]["n_train"],
        runtime_min=rec["results"]["train_runtime_min"],
        stopped_early=rec["results"]["stopped_early"],
        gpu=rec["environment"]["gpu"],
    )
    row.update({k: rec["health"].get(k) for k in HEALTH_KEYS})
    for m in EVALS.get(rid, {}).get("metrics", []):
        col = f"{m['category']}/{m['metric']}"
        row[col] = m["finetuned"]
        if m["lower_is_better"]:
            LOWER_IS_BETTER_EVAL.add(col)
    return row


df = pd.DataFrame([flatten(rid, rec) for rid, rec in RUNS.items()]).sort_values("eval_loss_best").reset_index(drop=True)
EVAL_COLS = sorted(c for c in df.columns if "/" in c)

BASELINE_ID = CONFIG["baseline_run_id"] or min(RUNS, key=lambda r: RUNS[r]["created_at"])
base_cfg = RUNS[BASELINE_ID]["config"]


def config_diff(rid):
    if rid == BASELINE_ID:
        return "(baseline)"
    cfg = RUNS[rid]["config"]
    changes = [f"{k}: {base_cfg.get(k)} → {cfg.get(k)}" for k in SPEC_KEYS if cfg.get(k) != base_cfg.get(k)]
    return "; ".join(changes) or "(specs giống baseline, khác seed/dữ liệu?)"


df["changes_vs_baseline"] = df["run_id"].map(config_diff)
df.to_csv(OUT / "runs_comparison.csv", index=False)
df[["run_name", "diagnosis", "eval_loss_best", "improvement_pct", "overfit_rise_pct", "runtime_min", "changes_vs_baseline"]].round(4)

# %% [markdown]
# ## Ảnh hưởng của từng thay đổi so với baseline
# Mỗi hàng: thay đổi gì, và các kết quả tăng/giảm bao nhiêu so với baseline (Δ = run − baseline).

# %%
OUTCOMES = [k for k, _ in SUMMARY_PANELS] + EVAL_COLS
base_row = df[df["run_id"] == BASELINE_ID].iloc[0]
effect = df[["run_name", "changes_vs_baseline"]].copy()
for col in OUTCOMES:
    if pd.api.types.is_numeric_dtype(df[col]) and pd.notna(base_row[col]):
        effect[f"Δ {col}"] = df[col] - base_row[col]
effect.round(4)

# %% [markdown]
# ## Biểu đồ so sánh

# %%
colors = {rid: plt.cm.tab10(i % 10) for i, rid in enumerate(df["run_id"])}
fig, (ax_eval, ax_train) = plt.subplots(1, 2, figsize=(18, 5.5))
for rid in df["run_id"]:
    log = RUNS[rid]["_log"]
    if log is None:
        continue
    name, lw = RUNS[rid]["run_name"], 2.5 if rid == BASELINE_ID else 1.5
    ev = log[log["eval_loss"].notna()] if "eval_loss" in log else log.iloc[0:0]
    tr = log[log["loss"].notna()] if "loss" in log else log.iloc[0:0]
    ax_eval.plot(ev["step"], ev["eval_loss"], "o-", ms=3, lw=lw, color=colors[rid], label=name)
    ax_train.plot(tr["step"], tr["loss"].rolling(5, min_periods=1).mean(), lw=lw, color=colors[rid], label=name)
ax_eval.set_title("eval_loss theo step (đậm = baseline)"); ax_train.set_title("train_loss (làm mượt)")
for a in (ax_eval, ax_train):
    a.set_xlabel("step"); a.grid(alpha=0.3); a.legend(fontsize=7)
plt.tight_layout()
plt.savefig(OUT / "compare_curves.png", dpi=120)
plt.show()

# %%
fig, axes = plt.subplots(2, 3, figsize=(20, max(6, 1.2 * len(df) + 4)))
for a, (col, title) in zip(axes.flatten(), SUMMARY_PANELS):
    values = df[col].astype(float)
    bar_colors = ["gold" if rid == BASELINE_ID else colors[rid] for rid in df["run_id"]]
    a.barh(df["run_name"], values, color=bar_colors, edgecolor="black")
    for y, v in enumerate(values):
        if pd.notna(v):
            a.text(v, y, f" {v:.4g}", va="center", fontsize=8)
    a.set_title(title); a.invert_yaxis(); a.tick_params(axis="y", labelsize=7); a.grid(axis="x", alpha=0.3)
fig.suptitle("So sánh các lần fine-tune (vàng = baseline)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "compare_summary.png", dpi=120)
plt.show()

# %% [markdown]
# ### Siêu tham số nào ảnh hưởng đến `eval_loss`?
# Chỉ vẽ các siêu tham số dạng số có ≥ 2 giá trị khác nhau giữa các lần chạy.

# %%
varying = [k for k in SPEC_KEYS if pd.api.types.is_numeric_dtype(df[k]) and df[k].nunique(dropna=True) >= 2]
if varying:
    fig, axes = plt.subplots(1, len(varying), figsize=(5.5 * len(varying), 4.5), squeeze=False)
    for a, k in zip(axes[0], varying):
        a.scatter(df[k], df["eval_loss_best"], s=80, c=[colors[r] for r in df["run_id"]], edgecolor="black")
        for x, y, n in zip(df[k], df["eval_loss_best"], df["run_name"]):
            a.annotate(n.split("_")[-1], (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
        if df[k].min() > 0 and df[k].max() / df[k].min() >= 10:
            a.set_xscale("log")
        a.set_xlabel(k); a.set_ylabel("best eval_loss"); a.grid(alpha=0.3)
    fig.suptitle("Siêu tham số vs best eval_loss", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "compare_hparam_effect.png", dpi=120)
    plt.show()
else:
    print("Chưa có siêu tham số số học nào thay đổi giữa các lần chạy.")

# %% [markdown]
# ### Hành vi sau fine-tune (từ notebook 03)
# Màu theo từng cột: xanh = tốt nhất trong các lần chạy, đỏ = kém nhất (đã tính chiều ↑/↓ của từng chỉ số).

# %%
if EVAL_COLS:
    values = df.set_index("run_name")[EVAL_COLS].astype(float)
    goodness = values.copy()
    for col in EVAL_COLS:
        span = values[col].max() - values[col].min()
        norm = (values[col] - values[col].min()) / span if span > 0 else 0.5
        goodness[col] = 1 - norm if col in LOWER_IS_BETTER_EVAL else norm
    fig, a = plt.subplots(figsize=(1.3 * len(EVAL_COLS) + 4, 0.6 * len(values) + 2.5))
    a.imshow(goodness.to_numpy(dtype=float), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    a.set_xticks(range(len(EVAL_COLS)), [c + (" ↓" if c in LOWER_IS_BETTER_EVAL else " ↑") for c in EVAL_COLS], rotation=45, ha="right", fontsize=8)
    a.set_yticks(range(len(values)), values.index, fontsize=8)
    for i in range(len(values)):
        for j, col in enumerate(EVAL_COLS):
            v = values.iloc[i, j]
            if pd.notna(v):
                a.text(j, i, f"{v:.3g}", ha="center", va="center", fontsize=8)
    a.set_title("Chỉ số hành vi của bản fine-tune theo từng lần chạy")
    plt.tight_layout()
    plt.savefig(OUT / "compare_eval_metrics.png", dpi=120)
    plt.show()
else:
    print("Chưa có eval_record.json: chạy notebook 03 cho các lần chạy rồi Add Input để so cả hành vi.")

# %% [markdown]
# ## Báo cáo `runs_comparison.md`

# %%
def md_table(frame):
    def fmt(v):
        if isinstance(v, float):
            return "—" if np.isnan(v) else f"{v:.4g}"
        return str(v)
    header = "| " + " | ".join(frame.columns) + " |\n|" + "---|" * len(frame.columns) + "\n"
    return header + "".join("| " + " | ".join(fmt(v) for v in row) + " |\n" for row in frame.itertuples(index=False))


healthy = df[df["diagnosis"] == "HEALTHY"]
recommended = (healthy if not healthy.empty else df).iloc[0]
ranking_cols = ["run_name", "diagnosis", "eval_loss_best", "improvement_pct", "overfit_rise_pct", "loss_spikes",
                "tokens_per_sec_mean", "vram_peak_gb", "runtime_min", "changes_vs_baseline"]
report = [
    "# Fine-tuning runs comparison\n",
    f"{len(df)} lần chạy · baseline: `{RUNS[BASELINE_ID]['run_name']}`\n",
    f"**Khuyến nghị:** `{recommended.run_name}` — best eval_loss {recommended.eval_loss_best:.4f}, "
    f"chẩn đoán {recommended.diagnosis}, thay đổi: {recommended.changes_vs_baseline}\n",
    "## Xếp hạng theo best eval_loss\n", md_table(df[ranking_cols]),
    "\n## Ảnh hưởng so với baseline (Δ = run − baseline)\n", md_table(effect),
]
if EVAL_COLS:
    report += ["\n## Chỉ số hành vi (bản fine-tune)\n", md_table(df[["run_name"] + EVAL_COLS])]
    winners = [f"- `{c}` ({'↓' if c in LOWER_IS_BETTER_EVAL else '↑'}): **{df.loc[(df[c].idxmin() if c in LOWER_IS_BETTER_EVAL else df[c].idxmax()), 'run_name']}**"
               for c in EVAL_COLS if df[c].notna().any()]
    report += ["\n### Lần chạy tốt nhất theo từng chỉ số\n"] + winners
report += [
    "\n## Cách đọc\n",
    "- So các lần chạy chỉ khác **một** spec với baseline thì mới kết luận được spec đó ảnh hưởng thế nào.",
    "- Chênh lệch `eval_loss` rất nhỏ (< ~0.01) có thể chỉ là nhiễu do seed; chạy lại cùng cấu hình với seed khác để kiểm tra.",
    "- Loss thấp hơn chưa chắc hành vi tốt hơn: luôn đối chiếu bảng chỉ số hành vi từ notebook 03.",
]
(OUT / "runs_comparison.md").write_text("\n".join(report), encoding="utf-8")
print("\n".join(report[:3]))
