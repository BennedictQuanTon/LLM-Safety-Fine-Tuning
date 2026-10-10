# %% [markdown]
# # 04 — Phân loại jailbreak (RMIT Hackathon 2025, Challenge 1+2) và tự chấm điểm
#
# **Cài đặt Kaggle:** Accelerator = **GPU T4 x1** · Internet = **On** ·
# Add Input → **Competitions → RMIT HACKATHON 2025** (phải bấm *Join / Accept rules* trên trang cuộc thi trước).
#
# Không nộp bài, mà **tự chấm** theo cách mô phỏng private leaderboard:
#
# 1. Tách cố định **20% train làm holdout** ("test giả"), không bao giờ dùng để chọn mô hình hay trọng số.
# 2. 80% còn lại (dev) chạy **5-fold CV** để huấn luyện và chọn cách gộp mô hình.
# 3. Điểm cuối = **ROC-AUC trên holdout**, kèm khoảng tin cậy 95% (bootstrap), rồi tra hạng trên leaderboard 2025.
#
# Mô hình: **A** TF-IDF + Logistic Regression (CPU), **B** fine-tune DeBERTa-v3 (GPU), gộp bằng rank-average với trọng số chọn trên CV.
#
# **Output** (`/kaggle/working/clf/`): `leaderboard_compare.md`, `scores.json`, `holdout_errors.csv`, `oof_predictions.csv`, `submission.csv`.

# %%
import glob
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset
from scipy.sparse import hstack
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

CONFIG = dict(
    seed=42,
    n_folds=5,
    n_holdout_splits=5,
    tfidf_C_grid=[4.0, 16.0, 64.0],
    transformers=[
        dict(name="microsoft/deberta-v3-base", max_len=512, head_len=384, lr=2e-5, epochs=3, batch_size=8, grad_accum=2),
        # Thêm mô hình thứ 2 để gộp nếu chưa đạt top 3, ví dụ:
        # dict(name="FacebookAI/xlm-roberta-base", max_len=512, head_len=384, lr=2e-5, epochs=3, batch_size=8, grad_accum=2),
    ],
    ensemble_iters=20,
    n_bootstrap=2000,
    out_dir="/kaggle/working/clf",
)
OUT = Path(CONFIG["out_dir"])
OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

# Private leaderboard cuối cùng của RMIT Hackathon 2025 (top 49 / 110 đội), lấy từ
# https://www.kaggle.com/competitions/rmit-hackathon-2025/leaderboard
LEADERBOARD_2025 = [
    0.99824, 0.99658, 0.99658, 0.99614, 0.99603, 0.99603, 0.99593, 0.99570, 0.99568, 0.99461,
    0.99425, 0.99425, 0.99359, 0.99359, 0.99359, 0.99359, 0.99359, 0.99324, 0.99316, 0.99261,
    0.99236, 0.99236, 0.99236, 0.99232, 0.99187, 0.99187, 0.99185, 0.99088, 0.99077, 0.99049,
    0.99039, 0.99039, 0.98989, 0.98910, 0.98805, 0.98788, 0.98664, 0.98570, 0.98545, 0.98509,
    0.98448, 0.98373, 0.98321, 0.98257, 0.98200, 0.98165, 0.98157, 0.98157, 0.98145,
]
N_TEAMS_2025 = 110

# %% [markdown]
# ## Nạp dữ liệu cuộc thi

# %%
def find_input(filename: str) -> str:
    matches = sorted(glob.glob(f"/kaggle/input/**/{filename}", recursive=True), key=lambda p: ("rmit" not in p.lower(), p))
    if not matches:
        raise FileNotFoundError(
            f"Không thấy {filename} trong /kaggle/input. Hãy Join cuộc thi RMIT HACKATHON 2025 rồi Add Input → Competitions."
        )
    return matches[0]


def pick_column(df: pd.DataFrame, *candidates: str) -> str:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in lower:
            return lower[name]
    raise KeyError(f"Không tìm thấy cột nào trong {candidates}; các cột hiện có: {list(df.columns)}")


train_path, test_path = find_input("train.csv"), find_input("test.csv")
train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)
print("train:", train_path, train_df.shape, "| test:", test_path, test_df.shape)

TEXT, LABEL = pick_column(train_df, "text", "prompt"), pick_column(train_df, "label", "target")
TEST_TEXT, TEST_ID = pick_column(test_df, "text", "prompt"), pick_column(test_df, "id")

labels = train_df[LABEL]
y_all = (labels.astype(str).str.lower() == "jailbreak").astype(int).values if labels.dtype == object else labels.astype(int).values
texts_all = train_df[TEXT].fillna("").astype(str).values
test_texts = test_df[TEST_TEXT].fillna("").astype(str).values
print(pd.Series(y_all).value_counts().rename({0: "benign", 1: "jailbreak"}))

# %% [markdown]
# ## Khám phá nhanh
# Độ dài prompt quyết định cách cắt cho transformer; trùng lặp giữa các fold làm điểm CV cao ảo nên phải gom nhóm.

# %%
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


groups_all = np.array([hashlib.md5(normalize(t).encode("utf-8")).hexdigest() for t in texts_all])
eda = pd.DataFrame({"label": y_all, "n_words": [len(t.split()) for t in texts_all],
                    "ascii_ratio": [sum(c.isascii() for c in t) / max(len(t), 1) for t in texts_all]})
print(eda.groupby("label")[["n_words", "ascii_ratio"]].describe().T.round(2))
dup = pd.DataFrame({"g": groups_all, "y": y_all}).groupby("g")["y"].agg(["count", "nunique"])
print(f"Nhóm trùng lặp: {(dup['count'] > 1).sum()} | trùng nhưng nhãn mâu thuẫn: {(dup['nunique'] > 1).sum()}")
print(f"Prompt trong test trùng y hệt train: {np.isin([hashlib.md5(normalize(t).encode()).hexdigest() for t in test_texts], groups_all).sum()}")

# %% [markdown]
# ## Tách holdout 20% và 5 fold trên phần dev
# `StratifiedGroupKFold`: giữ tỷ lệ nhãn, và các prompt trùng nhau luôn nằm cùng một phía.

# %%
holdout_splitter = StratifiedGroupKFold(n_splits=CONFIG["n_holdout_splits"], shuffle=True, random_state=CONFIG["seed"])
dev_idx, hold_idx = next(holdout_splitter.split(texts_all, y_all, groups_all))

X_dev, y_dev, g_dev = texts_all[dev_idx], y_all[dev_idx], groups_all[dev_idx]
X_hold, y_hold = texts_all[hold_idx], y_all[hold_idx]
folds = list(StratifiedGroupKFold(n_splits=CONFIG["n_folds"], shuffle=True, random_state=CONFIG["seed"]).split(X_dev, y_dev, g_dev))
print(f"dev: {len(dev_idx)} (jailbreak {y_dev.mean():.1%}) | holdout: {len(hold_idx)} (jailbreak {y_hold.mean():.1%})")

PREDS = {}

# %% [markdown]
# ## Mô hình A: TF-IDF (từ + ký tự) + Logistic Regression
# Jailbreak hay có cụm đặc trưng ("DAN", "ignore previous", "developer mode"); n-gram ký tự bắt được cả biến thể viết sai/che chữ.

# %%
def tfidf_fit_transform(fit_texts, *other_texts):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=300_000, sublinear_tf=True)
    X_fit = hstack([word.fit_transform(fit_texts), char.fit_transform(fit_texts)]).tocsr()
    return [X_fit] + [hstack([word.transform(t), char.transform(t)]).tocsr() for t in other_texts]


def run_tfidf(C):
    oof, hold, test = np.zeros(len(X_dev)), np.zeros(len(X_hold)), np.zeros(len(test_texts))
    for tr, va in folds:
        X_tr, X_va, X_ho, X_te = tfidf_fit_transform(X_dev[tr], X_dev[va], X_hold, test_texts)
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(X_tr, y_dev[tr])
        oof[va] = clf.predict_proba(X_va)[:, 1]
        hold += clf.predict_proba(X_ho)[:, 1] / len(folds)
        test += clf.predict_proba(X_te)[:, 1] / len(folds)
    return dict(oof=oof, holdout=hold, test=test, cv_auc=roc_auc_score(y_dev, oof))


t0 = time.time()
tfidf_runs = {C: run_tfidf(C) for C in CONFIG["tfidf_C_grid"]}
for C, r in tfidf_runs.items():
    print(f"TF-IDF C={C:<5} CV AUC = {r['cv_auc']:.5f}")
best_C = max(tfidf_runs, key=lambda C: tfidf_runs[C]["cv_auc"])
PREDS[f"tfidf_lr_C{best_C:g}"] = tfidf_runs[best_C]
print(f"Chọn C={best_C} theo CV ({time.time() - t0:.0f}s)")

# %% [markdown]
# ## Mô hình B: fine-tune transformer (DeBERTa-v3-base)
# Prompt jailbreak thường rất dài; cắt kiểu **đầu + đuôi** (384 token đầu + 128 token cuối) để giữ cả phần dựng vai lẫn yêu cầu cuối.
# Mỗi fold train đủ số epoch cố định, không chọn epoch theo fold đó, để điểm CV không bị lạc quan.
# 5 mô hình fold cùng dự đoán holdout và test, lấy trung bình.

# %%
def special_token_wrap(tokenizer):
    probe = tokenizer("x", add_special_tokens=False)["input_ids"]
    full = tokenizer("x", add_special_tokens=True)["input_ids"]
    for i in range(len(full) - len(probe) + 1):
        if full[i:i + len(probe)] == probe:
            return full[:i], full[i + len(probe):]
    raise ValueError(f"Không tách được token đặc biệt của {tokenizer.__class__.__name__}: {full}")


def head_tail_encode(tokenizer, texts, max_len, head_len):
    prefix, suffix = special_token_wrap(tokenizer)
    body_len = max_len - len(prefix) - len(suffix)
    tail_len = body_len - head_len
    ids_list = tokenizer(list(texts), add_special_tokens=False, truncation=False)["input_ids"]
    input_ids = []
    for ids in ids_list:
        if len(ids) > body_len:
            ids = ids[:head_len] + ids[-tail_len:]
        input_ids.append(prefix + ids + suffix)
    return input_ids


def to_dataset(input_ids, labels=None):
    data = {"input_ids": input_ids, "attention_mask": [[1] * len(x) for x in input_ids]}
    if labels is not None:
        data["labels"] = [int(v) for v in labels]
    return Dataset.from_dict(data)


def positive_prob(logits):
    return torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[:, 1].numpy()


TRAINING_ARG_NAMES = set(inspect.signature(TrainingArguments.__init__).parameters)


def make_training_args(**kwargs):
    unsupported = sorted(k for k in kwargs if k not in TRAINING_ARG_NAMES)
    if unsupported:
        print(f"⚠️ transformers {transformers.__version__} không hỗ trợ {unsupported}, bỏ qua các tham số này.")
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in TRAINING_ARG_NAMES})


def run_transformer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    dev_ids = head_tail_encode(tokenizer, X_dev, cfg["max_len"], cfg["head_len"])
    hold_ds = to_dataset(head_tail_encode(tokenizer, X_hold, cfg["max_len"], cfg["head_len"]))
    test_ds = to_dataset(head_tail_encode(tokenizer, test_texts, cfg["max_len"], cfg["head_len"]))
    collator = DataCollatorWithPadding(tokenizer)
    print(f"Độ dài token lớn nhất: {max(map(len, dev_ids))} | mẫu đầu: {tokenizer.decode(dev_ids[0][:40])!r}")

    oof, hold, test = np.zeros(len(X_dev)), np.zeros(len(X_hold)), np.zeros(len(test_texts))
    for k, (tr, va) in enumerate(folds):
        t0 = time.time()
        fold_dir = f"/kaggle/working/tmp_{cfg['name'].split('/')[-1]}_fold{k}"
        model = AutoModelForSequenceClassification.from_pretrained(cfg["name"], num_labels=2)
        steps_per_epoch = math.ceil(len(tr) / (cfg["batch_size"] * cfg["grad_accum"]))
        args = make_training_args(
            output_dir=fold_dir,
            per_device_train_batch_size=cfg["batch_size"],
            per_device_eval_batch_size=cfg["batch_size"] * 4,
            gradient_accumulation_steps=cfg["grad_accum"],
            learning_rate=cfg["lr"],
            num_train_epochs=cfg["epochs"],
            warmup_steps=math.ceil(0.1 * steps_per_epoch * cfg["epochs"]),
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            fp16=torch.cuda.is_available(),
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=20,
            group_by_length=True,
            report_to="none",
            seed=CONFIG["seed"] + k,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=to_dataset([dev_ids[i] for i in tr], y_dev[tr]),
            eval_dataset=to_dataset([dev_ids[i] for i in va], y_dev[va]),
            data_collator=collator,
            compute_metrics=lambda p: {"roc_auc": roc_auc_score(p.label_ids, positive_prob(p.predictions))},
        )
        trainer.train()
        oof[va] = positive_prob(trainer.predict(trainer.eval_dataset).predictions)
        hold += positive_prob(trainer.predict(hold_ds).predictions) / len(folds)
        test += positive_prob(trainer.predict(test_ds).predictions) / len(folds)
        print(f"{cfg['name']} fold {k}: AUC = {roc_auc_score(y_dev[va], oof[va]):.5f} ({(time.time() - t0) / 60:.1f} phút)")

        del trainer, model
        shutil.rmtree(fold_dir, ignore_errors=True)
        torch.cuda.empty_cache()
    return dict(oof=oof, holdout=hold, test=test, cv_auc=roc_auc_score(y_dev, oof))


for cfg in CONFIG["transformers"]:
    PREDS[cfg["name"].split("/")[-1]] = run_transformer(cfg)
    print(f"{cfg['name']} CV AUC = {PREDS[cfg['name'].split('/')[-1]]['cv_auc']:.5f}")

# %% [markdown]
# ## Gộp mô hình (rank-average, trọng số chọn trên CV)
# ROC-AUC chỉ phụ thuộc thứ tự, nên gộp **thứ hạng** thay vì xác suất thô.
# Trọng số chọn bằng hill-climbing trên dự đoán OOF của dev; holdout không tham gia.

# %%
def rank01(x):
    return (rankdata(x) - 1) / max(len(x) - 1, 1)


def blend(preds, weights, key):
    return sum(w * rank01(preds[name][key]) for name, w in weights.items() if w > 0)


def hill_climb(preds, y, iters):
    counts = {name: 0 for name in preds}
    best_auc, current = -1.0, None
    for _ in range(iters):
        total = sum(counts.values())
        trials = []
        for name in preds:
            r = rank01(preds[name]["oof"])
            trial = r if current is None else (current * total + r) / (total + 1)
            trials.append((roc_auc_score(y, trial), name, trial))
        auc, name, trial = max(trials, key=lambda t: t[0])
        if current is not None and auc <= best_auc:
            break
        counts[name] += 1
        best_auc, current = auc, trial
    total = sum(counts.values())
    return {name: c / total for name, c in counts.items()}, best_auc


weights, ens_cv_auc = hill_climb(PREDS, y_dev, CONFIG["ensemble_iters"])
PREDS["ensemble"] = dict(
    oof=blend(PREDS, weights, "oof"), holdout=blend(PREDS, weights, "holdout"),
    test=blend(PREDS, weights, "test"), cv_auc=ens_cv_auc,
)
print("Trọng số:", {k: round(v, 3) for k, v in weights.items()}, f"| CV AUC ensemble = {ens_cv_auc:.5f}")

# %% [markdown]
# ## Chấm điểm trên holdout và so với leaderboard 2025
# Bootstrap 2000 lần trên holdout để thấy điểm dao động bao nhiêu: private test 2025 chỉ ~650 mẫu nên chênh 0.002 có thể chỉ là may rủi.

# %%
def bootstrap_ci(y, p, n, seed):
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        scores.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def lb_rank(score):
    rank = 1 + sum(s > score for s in LEADERBOARD_2025)
    return f"#{rank}" if rank <= len(LEADERBOARD_2025) else f"ngoài top {len(LEADERBOARD_2025)} (/{N_TEAMS_2025})"


rows = []
for name, r in PREDS.items():
    auc = roc_auc_score(y_hold, r["holdout"])
    lo, hi = bootstrap_ci(y_hold, r["holdout"], CONFIG["n_bootstrap"], CONFIG["seed"])
    rows.append(dict(model=name, cv_auc=r["cv_auc"], holdout_auc=auc, ci95_low=lo, ci95_high=hi,
                     rank_2025=lb_rank(auc), rank_range=f"{lb_rank(hi)} … {lb_rank(lo)}"))
scores = pd.DataFrame(rows).sort_values("holdout_auc", ascending=False)
scores.round(5)

# %%
final = scores.set_index("model").loc["ensemble"]
top3_line = LEADERBOARD_2025[2]
verdict = (
    f"ĐẠT top 3 (holdout AUC {final.holdout_auc:.5f} ≥ {top3_line})" if final.holdout_auc >= top3_line
    else f"CHƯA đạt top 3: thiếu {top3_line - final.holdout_auc:.5f} so với mốc {top3_line}"
)
print(verdict)

md = [
    "# So sánh với RMIT Hackathon 2025 (Challenge 1+2, ROC-AUC)\n",
    f"**Kết luận:** {verdict}. Hạng tương đương: **{final.rank_2025}** "
    f"(khoảng tin cậy 95%: {final.rank_range}).\n",
    "Mốc leaderboard private 2025: hạng 1 = 0.99824 · hạng 3 = 0.99658 · hạng 10 = 0.99461 · hạng 30 = 0.99049.\n",
    "| Mô hình | CV AUC (dev) | Holdout AUC | CI 95% | Hạng tương đương | Khoảng hạng |",
    "|---|---|---|---|---|---|",
]
md += [f"| {r.model} | {r.cv_auc:.5f} | **{r.holdout_auc:.5f}** | {r.ci95_low:.5f}–{r.ci95_high:.5f} | {r.rank_2025} | {r.rank_range} |"
       for r in scores.itertuples()]
md += [
    "",
    f"Trọng số ensemble: {json.dumps({k: round(v, 3) for k, v in weights.items()})}",
    "",
    "**Lưu ý khi so sánh:** holdout được tách từ train của cuộc thi (cùng phân phối), còn private test 2025 là dữ liệu ẩn; "
    "các đội thi chỉ có 1–2 ngày và 10 lượt nộp. Đây là ước lượng mô hình đạt *mức* hạng nào, không phải kết quả thi chính thức.",
]
(OUT / "leaderboard_compare.md").write_text("\n".join(md), encoding="utf-8")
(OUT / "scores.json").write_text(json.dumps(dict(
    verdict=verdict, weights=weights, models=scores.to_dict(orient="records"),
    n_dev=int(len(dev_idx)), n_holdout=int(len(hold_idx)), config=CONFIG,
), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print("\n".join(md))

# %% [markdown]
# ## Phân tích lỗi trên holdout
# Xem các mẫu bị xếp sai nhiều nhất để biết cần thêm mô hình/dữ liệu kiểu gì.

# %%
hold_scores = PREDS["ensemble"]["holdout"]
err = pd.DataFrame({"text": X_hold, "label": y_hold, "score": hold_scores})
false_neg = err[err.label == 1].nsmallest(15, "score").assign(error="jailbreak bị chấm thấp")
false_pos = err[err.label == 0].nlargest(15, "score").assign(error="benign bị chấm cao")
errors = pd.concat([false_neg, false_pos])
errors.to_csv(OUT / "holdout_errors.csv", index=False)
pd.DataFrame({"text": X_dev, "label": y_dev, **{f"oof_{k}": v["oof"] for k, v in PREDS.items()}}).to_csv(OUT / "oof_predictions.csv", index=False)
for row in errors.head(6).itertuples():
    print(f"[{row.error} | score={row.score:.3f}] {row.text[:250]!r}\n")

# %% [markdown]
# ## File submission (không nộp, để sẵn nếu sau này muốn thử Late Submission)

# %%
sample_path = glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)
id_col, target_col = (pd.read_csv(sample_path[0]).columns[:2].tolist() if sample_path else [TEST_ID, "target"])
pd.DataFrame({id_col: test_df[TEST_ID], target_col: PREDS["ensemble"]["test"]}).to_csv(OUT / "submission.csv", index=False)
print("Đã ghi", OUT / "submission.csv")
