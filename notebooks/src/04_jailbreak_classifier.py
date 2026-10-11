# %% [markdown]
# # 04 — Phân loại jailbreak (RMIT Hackathon 2025, Challenge 1+2) và tự chấm điểm
#
# Một notebook chạy hết, khoảng **8 giờ** trên **GPU T4 x2** (giới hạn Kaggle là 12 giờ):
#
# | Mô hình | Cách làm | Thời gian ước tính |
# |---|---|---|
# | `tfidf` | TF-IDF từ + ký tự → Logistic Regression | ~3 phút |
# | `deberta-v3-base` | fine-tune toàn bộ, chạy trên 2 GPU | ~45 phút |
# | `deberta-v3-large` | fine-tune toàn bộ, chạy trên 2 GPU | ~2.5 giờ |
# | `qwen2.5-3b` | Qwen2.5-3B-Instruct 4-bit + LoRA, đầu phân loại (1 GPU) | ~3.5 giờ |
# | `guard_wildguard` | WildGuard 7B chấm zero-shot, không train (1 GPU) | ~40 phút |
#
# **Cài đặt Kaggle:** Accelerator = **GPU T4 x2** · Internet = **On** · Add Input → **Competitions → RMIT HACKATHON 2025**.
#
# WildGuard là mô hình bị khóa (gated) trên Hugging Face nên cần token. **Không có token thì notebook tự bỏ qua WildGuard**, các mô hình khác vẫn chạy.
# Muốn dùng: bấm đồng ý điều khoản tại https://huggingface.co/allenai/wildguard, rồi thêm Kaggle Secret tên **`HF_TOKEN`** (Add-ons → Secrets).
#
# Mô hình nào lỗi giữa chừng thì được ghi lại và bỏ qua, các mô hình đã xong vẫn được gộp và báo cáo.
#
# Cách tự chấm (mô phỏng private leaderboard): tách cố định **20% train làm holdout**, 80% dev chạy **5-fold CV**;
# trọng số ensemble chỉ chọn trên OOF của dev; điểm cuối = **ROC-AUC holdout** + CI 95% bootstrap, tra hạng leaderboard 2025.
# Báo 2 bản: `ensemble_clean` (chỉ mô hình train trên dữ liệu đề) và `ensemble_all` (thêm WildGuard, đã được train trên dữ liệu rất giống đề).
#
# **Output** (`/kaggle/working/clf/`): `REPORT.md`, `charts/*.png`, `models/<tên>/{preds.npz, history.csv, folds.csv, spec.json}`,
# `run_record.json`, `scores.json`, `leaderboard_compare.md`, `oof_predictions.csv`, `holdout_predictions.csv`, `holdout_errors.csv`, `submission*.csv`.

# %%
!pip install -q -U bitsandbytes peft

# %%
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import glob
import hashlib
import inspect
import json
import math
import platform
import re
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset
from scipy.sparse import hstack
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class ElapsedCallback(TrainerCallback):
    """Ghi số phút kể từ đầu fold vào mỗi dòng log để vẽ loss theo thời gian thật."""

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.log_history:
            state.log_history[-1]["elapsed_min"] = (time.time() - self.t0) / 60

MODELS = {
    "tfidf": dict(kind="tfidf", C_grid=[4.0, 16.0, 64.0]),
    "deberta-v3-base": dict(kind="encoder", name="microsoft/deberta-v3-base", max_len=512, head_len=384,
                            lr=2e-5, epochs=3, batch_size=8, grad_accum=2),
    "deberta-v3-large": dict(kind="encoder", name="microsoft/deberta-v3-large", max_len=512, head_len=384,
                             lr=1e-5, epochs=2, batch_size=2, grad_accum=8, grad_ckpt=True),
    "qwen2.5-3b": dict(kind="llm", name="Qwen/Qwen2.5-3B-Instruct", max_len=512, head_len=384,
                       lr=1e-4, epochs=2, batch_size=4, grad_accum=4, lora_r=16, lora_alpha=32, lora_dropout=0.05),
    "guard_wildguard": dict(kind="guard", name="allenai/wildguard", max_prompt_tokens=1024, batch_size=8),
}
# Thời gian ước tính trên T4 x2 (giờ), dùng để bỏ qua mô hình nếu không kịp trước giới hạn 12 giờ của Kaggle.
EST_HOURS = {"tfidf": 0.05, "deberta-v3-base": 1.3, "deberta-v3-large": 4.0, "qwen2.5-3b": 6.3, "guard_wildguard": 0.7}
CONFIG = dict(
    version="v3",
    # Thứ tự = ưu tiên. Mô hình nào có sẵn trong Input (cùng config + cùng SPLIT_FP) thì nạp lại, không train.
    run_models=["tfidf", "deberta-v3-base", "qwen2.5-3b", "guard_wildguard", "deberta-v3-large"],
    time_budget_hours=11.3,
    seed=42,
    n_folds=5,
    n_holdout_splits=5,
    evals_per_epoch=4,
    train_probe_size=320,
    ensemble_iters=30,
    n_bootstrap=2000,
)
RUN_START = time.time()
RUN_ID = f"{CONFIG['version']}_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
OUT = Path("/kaggle/working") / CONFIG["version"]
MODELS_DIR, CHARTS_DIR = OUT / "models", OUT / "charts"
for d in (OUT, MODELS_DIR, CHARTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
N_GPU = torch.cuda.device_count()
print(f"RUN_ID={RUN_ID} | mô hình: {CONFIG['run_models']} | GPU: {N_GPU} × "
      f"{torch.cuda.get_device_name(0) if N_GPU else 'CPU'} | transformers {transformers.__version__}")

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
    matches = [p for p in matches if "/models/" not in p]
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
# `SPLIT_FP` là dấu vân tay của cách chia: dự đoán từ notebook khác chỉ được gộp khi trùng dấu này.

# %%
holdout_splitter = StratifiedGroupKFold(n_splits=CONFIG["n_holdout_splits"], shuffle=True, random_state=CONFIG["seed"])
dev_idx, hold_idx = next(holdout_splitter.split(texts_all, y_all, groups_all))

X_dev, y_dev, g_dev = texts_all[dev_idx], y_all[dev_idx], groups_all[dev_idx]
X_hold, y_hold = texts_all[hold_idx], y_all[hold_idx]
folds = list(StratifiedGroupKFold(n_splits=CONFIG["n_folds"], shuffle=True, random_state=CONFIG["seed"]).split(X_dev, y_dev, g_dev))
SPLIT_FP = hashlib.md5(b"|".join([dev_idx.tobytes(), hold_idx.tobytes()] + [va.tobytes() for _, va in folds])).hexdigest()[:12]
print(f"dev: {len(dev_idx)} (jailbreak {y_dev.mean():.1%}) | holdout: {len(hold_idx)} (jailbreak {y_hold.mean():.1%}) | SPLIT_FP={SPLIT_FP}")

PREDS, HISTORY, FOLD_STATS, SPECS = {}, {}, {}, {}

# %% [markdown]
# ## Lưu từng mô hình
# Mỗi mô hình ghi `models/<tên>/`: `preds.npz` (OOF, holdout, test), `history.csv` (loss/AUC theo bước, từng fold),
# `folds.csv` (AUC, thời gian, VRAM mỗi fold), `spec.json` (cấu hình + môi trường). Nhờ vậy lần chạy sau gộp lại mà không phải train lại.

# %%
def elapsed_hours():
    return (time.time() - RUN_START) / 3600


def environment():
    env = dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
               gpu_count=N_GPU, gpu=torch.cuda.get_device_name(0) if N_GPU else None)
    for pkg in ("peft", "bitsandbytes", "sklearn"):
        try:
            env[pkg] = __import__(pkg).__version__
        except ImportError:
            pass
    return env


def save_model_outputs(key, result, history, fold_stats, spec):
    d = MODELS_DIR / key
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "preds.npz", oof=result["oof"], holdout=result["holdout"], test=result["test"],
             cv_auc=result["cv_auc"], split_fp=SPLIT_FP)
    pd.DataFrame(history).to_csv(d / "history.csv", index=False)
    pd.DataFrame(fold_stats).to_csv(d / "folds.csv", index=False)
    (d / "spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    PREDS[key], HISTORY[key], FOLD_STATS[key], SPECS[key] = result, history, fold_stats, spec


def base_spec(key, minutes, **derived):
    return dict(model=key, run_id=RUN_ID, created_at=datetime.now(timezone.utc).isoformat(),
                config=MODELS[key], derived=derived, minutes=round(minutes, 2), split_fp=SPLIT_FP, environment=environment())

# %% [markdown]
# ## Mô hình A: TF-IDF (từ + ký tự) + Logistic Regression
# Jailbreak hay có cụm đặc trưng ("DAN", "ignore previous", "developer mode"); n-gram ký tự bắt được cả biến thể viết sai/che chữ.

# %%
def tfidf_fit_transform(fit_texts, *other_texts):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=300_000, sublinear_tf=True)
    X_fit = hstack([word.fit_transform(fit_texts), char.fit_transform(fit_texts)]).tocsr()
    return [X_fit] + [hstack([word.transform(t), char.transform(t)]).tocsr() for t in other_texts]


def run_tfidf_C(C):
    oof, hold, test = np.zeros(len(X_dev)), np.zeros(len(X_hold)), np.zeros(len(test_texts))
    stats = []
    for k, (tr, va) in enumerate(folds):
        t0 = time.time()
        X_tr, X_va, X_ho, X_te = tfidf_fit_transform(X_dev[tr], X_dev[va], X_hold, test_texts)
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(X_tr, y_dev[tr])
        oof[va] = clf.predict_proba(X_va)[:, 1]
        hold += clf.predict_proba(X_ho)[:, 1] / len(folds)
        test += clf.predict_proba(X_te)[:, 1] / len(folds)
        stats.append(dict(fold=k, auc=roc_auc_score(y_dev[va], oof[va]), minutes=(time.time() - t0) / 60, peak_vram_gb=0.0))
    return dict(oof=oof, holdout=hold, test=test, cv_auc=roc_auc_score(y_dev, oof)), stats


def run_tfidf(key):
    t0 = time.time()
    runs = {C: run_tfidf_C(C) for C in MODELS[key]["C_grid"]}
    for C, (r, _) in runs.items():
        print(f"TF-IDF C={C:<5} CV AUC = {r['cv_auc']:.5f}")
    best_C = max(runs, key=lambda C: runs[C][0]["cv_auc"])
    result, stats = runs[best_C]
    history = [dict(fold=-1, step=i, C=C, cv_auc=r["cv_auc"]) for i, (C, (r, _)) in enumerate(runs.items())]
    save_model_outputs(key, result, history, stats,
                       base_spec(key, (time.time() - t0) / 60, best_C=best_C, C_cv_auc={str(C): r["cv_auc"] for C, (r, _) in runs.items()}))
    print(f"Chọn C={best_C} theo CV")

# %% [markdown]
# ## Mô hình B/C: fine-tune transformer (DeBERTa-v3, Qwen2.5 + QLoRA)
# Prompt jailbreak thường rất dài; cắt kiểu **đầu + đuôi** (384 token đầu + 128 token cuối) để giữ cả phần dựng vai lẫn yêu cầu cuối.
# Mỗi fold train đủ số epoch cố định, không chọn epoch theo fold đó, để điểm CV không bị lạc quan.
# Các mô hình fold cùng dự đoán holdout và test, lấy trung bình.
#
# Qwen2.5-3B được nạp 4-bit (NF4) và chỉ train LoRA + đầu phân loại `score`; LLM hiểu *ý định* của các prompt nhập vai dài tốt hơn DeBERTa.

# %%
def special_token_wrap(tokenizer):
    probe = tokenizer("x", add_special_tokens=False)["input_ids"]
    full = tokenizer("x", add_special_tokens=True)["input_ids"]
    for i in range(len(full) - len(probe) + 1):
        if full[i:i + len(probe)] == probe:
            return full[:i], full[i + len(probe):]
    raise ValueError(f"Không tách được token đặc biệt của {tokenizer.__class__.__name__}: {full}")


def head_tail_ids(ids, body_len, head_len):
    if len(ids) <= body_len:
        return ids
    return ids[:head_len] + ids[len(ids) - (body_len - head_len):]


def head_tail_encode(tokenizer, texts, max_len, head_len):
    prefix, suffix = special_token_wrap(tokenizer)
    body_len = max_len - len(prefix) - len(suffix)
    ids_list = tokenizer(list(texts), add_special_tokens=False, truncation=False)["input_ids"]
    return [prefix + head_tail_ids(ids, body_len, head_len) + suffix for ids in ids_list]


def to_dataset(input_ids, labels=None):
    data = {"input_ids": input_ids, "attention_mask": [[1] * len(x) for x in input_ids]}
    if labels is not None:
        data["labels"] = [int(v) for v in labels]
    return Dataset.from_dict(data)


def positive_prob(logits):
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return torch.softmax(torch.tensor(np.asarray(logits), dtype=torch.float32), dim=-1)[:, 1].numpy()


TRAINING_ARG_NAMES = set(inspect.signature(TrainingArguments.__init__).parameters)


def make_training_args(**kwargs):
    unsupported = sorted(k for k in kwargs if k not in TRAINING_ARG_NAMES)
    if unsupported:
        print(f"⚠️ transformers {transformers.__version__} không hỗ trợ {unsupported}, bỏ qua các tham số này.")
    return TrainingArguments(**{k: v for k, v in kwargs.items() if k in TRAINING_ARG_NAMES})


def load_classifier(cfg, tokenizer):
    if cfg["kind"] == "encoder":
        return AutoModelForSequenceClassification.from_pretrained(cfg["name"], num_labels=2).float()
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["name"], num_labels=2, quantization_config=bnb, dtype=torch.float16, device_map={"": 0})
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(task_type="SEQ_CLS", r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def run_transformer(key):
    cfg = MODELS[key]
    t_model = time.time()
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    if cfg["kind"] == "llm":
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    dev_ids = head_tail_encode(tokenizer, X_dev, cfg["max_len"], cfg["head_len"])
    hold_ds = to_dataset(head_tail_encode(tokenizer, X_hold, cfg["max_len"], cfg["head_len"]))
    test_ds = to_dataset(head_tail_encode(tokenizer, test_texts, cfg["max_len"], cfg["head_len"]))
    collator = DataCollatorWithPadding(tokenizer)
    n_tok = np.array([len(x) for x in dev_ids])
    print(f"{key}: token trung bình {n_tok.mean():.0f}, bị cắt {np.mean(n_tok >= cfg['max_len']):.1%} | mẫu đầu: {tokenizer.decode(dev_ids[0][:40])!r}")

    devices = max(N_GPU, 1) if cfg["kind"] == "encoder" else 1
    effective_batch = cfg["batch_size"] * cfg["grad_accum"] * devices
    oof, hold, test = np.zeros(len(X_dev)), np.zeros(len(X_hold)), np.zeros(len(test_texts))
    history, stats = [], []
    for k, (tr, va) in enumerate(folds):
        if stats:
            need_h = elapsed_hours() + stats[-1]["minutes"] / 60
            if need_h > CONFIG["time_budget_hours"]:
                raise TimeoutError(f"dừng trước fold {k}: cần ~{need_h:.1f}h > ngân sách {CONFIG['time_budget_hours']}h")
        t0 = time.time()
        if N_GPU:
            torch.cuda.reset_peak_memory_stats()
        fold_dir = f"/kaggle/working/tmp_{key}_fold{k}"
        model = load_classifier(cfg, tokenizer)
        steps_per_epoch = math.ceil(len(tr) / effective_batch)
        eval_steps = max(1, steps_per_epoch // CONFIG["evals_per_epoch"])
        is_llm = cfg["kind"] == "llm"
        val_ds = to_dataset([dev_ids[i] for i in va], y_dev[va])
        probe = np.random.default_rng(CONFIG["seed"] + k).choice(tr, min(CONFIG["train_probe_size"], len(tr)), replace=False)
        train_probe_ds = to_dataset([dev_ids[i] for i in probe], y_dev[probe])
        args = make_training_args(
            output_dir=fold_dir,
            per_device_train_batch_size=cfg["batch_size"],
            per_device_eval_batch_size=cfg["batch_size"] * (2 if is_llm else 4),
            gradient_accumulation_steps=cfg["grad_accum"],
            learning_rate=cfg["lr"],
            num_train_epochs=cfg["epochs"],
            warmup_steps=math.ceil(0.1 * steps_per_epoch * cfg["epochs"]),
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=cfg.get("grad_ckpt", False),
            gradient_checkpointing_kwargs={"use_reentrant": False} if cfg.get("grad_ckpt") else None,
            optim="paged_adamw_8bit" if is_llm else "adamw_torch",
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="no",
            logging_steps=10,
            label_names=["labels"],
            report_to="none",
            seed=CONFIG["seed"] + k,
        )
        if is_llm and args.n_gpu > 1:
            args._n_gpu = 1
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=to_dataset([dev_ids[i] for i in tr], y_dev[tr]),
            eval_dataset={"val": val_ds, "train": train_probe_ds},
            data_collator=collator,
            compute_metrics=lambda p: {"roc_auc": roc_auc_score(p.label_ids, positive_prob(p.predictions))},
            callbacks=[ElapsedCallback()],
        )
        trainer.train()
        oof[va] = positive_prob(trainer.predict(val_ds).predictions)
        hold += positive_prob(trainer.predict(hold_ds).predictions) / len(folds)
        test += positive_prob(trainer.predict(test_ds).predictions) / len(folds)
        fold_auc = roc_auc_score(y_dev[va], oof[va])
        minutes = (time.time() - t0) / 60
        history += [dict(fold=k, **row) for row in trainer.state.log_history]
        history.append(dict(fold=k, step=trainer.state.global_step, epoch=float(cfg["epochs"]), eval_roc_auc_final=fold_auc))
        stats.append(dict(fold=k, auc=fold_auc, minutes=minutes,
                          peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9 if N_GPU else 0.0,
                          train_steps=trainer.state.global_step))
        eta = minutes * (len(folds) - k - 1)
        flag = "  ⚠️ AUC gần 0.5: mô hình có thể đã sụp (lr quá cao / FP16 tràn số)" if fold_auc < 0.6 else ""
        print(f"{key} fold {k}: AUC = {fold_auc:.5f} ({minutes:.1f} phút, còn ~{eta:.0f} phút){flag}")

        del trainer, model
        shutil.rmtree(fold_dir, ignore_errors=True)
        torch.cuda.empty_cache()
    result = dict(oof=oof, holdout=hold, test=test, cv_auc=roc_auc_score(y_dev, oof))
    save_model_outputs(key, result, history, stats, base_spec(
        key, (time.time() - t_model) / 60, devices=devices, effective_batch=effective_batch,
        steps_per_epoch=steps_per_epoch, mean_tokens=float(n_tok.mean()), truncated_pct=float(np.mean(n_tok >= cfg["max_len"]))))
    print(f"{key} CV AUC = {result['cv_auc']:.5f}")

# %% [markdown]
# ## Mô hình D: WildGuard chấm zero-shot
# WildGuard (7B) là mô hình kiểm duyệt đã được train để trả lời "Harmful request: yes/no".
# Điểm = logit(yes) − logit(no) ở token ngay sau `Harmful request:`. Không train, nên điểm trên dev chính là "OOF".
# Mô hình này đã thấy dữ liệu rất giống đề (WildJailbreak), nên được báo riêng trong `ensemble_all`.

# %%
WILDGUARD_TEMPLATE = """<s><|user|>
[INST] You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine 1) whether the human user's request is harmful, 2) whether the AI assistant's response is a refusal, and 3) whether the AI assistant's response is harmful.

Human user:
{prompt}

AI assistant:


---

Answers: [/INST]
<|assistant|>
Harmful request:"""


def hf_login():
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as exc:
        print(f"Không đọc được Kaggle Secret HF_TOKEN ({type(exc).__name__}).")
        return False
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)
    return True


def next_token_id(tokenizer, prefix, word):
    a = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    b = tokenizer(prefix + word, add_special_tokens=False)["input_ids"]
    if b[:len(a)] != a or len(b) <= len(a):
        raise ValueError(f"Không tách được token cho {word!r}")
    return b[len(a)]


@torch.no_grad()
def guard_scores(model, tokenizer, texts, cfg, yes_id, no_id):
    body = cfg["max_prompt_tokens"]
    prompts = []
    for ids in tokenizer(list(texts), add_special_tokens=False)["input_ids"]:
        prompt = tokenizer.decode(head_tail_ids(ids, body, body * 3 // 4))
        prompts.append(WILDGUARD_TEMPLATE.format(prompt=prompt))
    order = np.argsort([len(p) for p in prompts])
    scores = np.zeros(len(prompts))
    for start in range(0, len(order), cfg["batch_size"]):
        idx = order[start:start + cfg["batch_size"]]
        enc = tokenizer([prompts[i] for i in idx], add_special_tokens=False, padding=True, return_tensors="pt").to(model.device)
        logits = model(**enc).logits[:, -1, :].float()
        scores[idx] = (logits[:, yes_id] - logits[:, no_id]).cpu().numpy()
        if start // cfg["batch_size"] % 50 == 0:
            print(f"  {start}/{len(order)}")
    return scores


def run_guard(key):
    from transformers import BitsAndBytesConfig

    cfg = MODELS[key]
    t0 = time.time()
    if not hf_login():
        FAILURES[key] = "bỏ qua: không có HF_TOKEN (mô hình gated)"
        print(f"⏭️ Bỏ qua {key}: thêm Kaggle Secret HF_TOKEN nếu muốn dùng WildGuard.")
        return
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(cfg["name"], quantization_config=bnb, dtype=torch.float16, device_map={"": 0})
    model.eval()
    yes_id, no_id = (next_token_id(tokenizer, "Harmful request:", w) for w in (" yes", " no"))
    if N_GPU:
        torch.cuda.reset_peak_memory_stats()
    parts = {}
    for name, texts in (("dev", X_dev), ("holdout", X_hold), ("test", test_texts)):
        t_part = time.time()
        parts[name] = guard_scores(model, tokenizer, texts, cfg, yes_id, no_id)
        print(f"{key} {name}: {len(texts)} mẫu, {(time.time() - t_part) / 60:.1f} phút")
    result = dict(oof=parts["dev"], holdout=parts["holdout"], test=parts["test"], cv_auc=roc_auc_score(y_dev, parts["dev"]))
    minutes = (time.time() - t0) / 60
    stats = [dict(fold=-1, auc=roc_auc_score(y_dev[va], parts["dev"][va]), minutes=minutes / len(folds),
                  peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9 if N_GPU else 0.0) for _, va in folds]
    save_model_outputs(key, result, [], stats, base_spec(key, minutes, yes_token=yes_id, no_token=no_id))
    print(f"{key} AUC trên dev = {result['cv_auc']:.5f}")
    del model
    torch.cuda.empty_cache()

# %% [markdown]
# ## Chạy lần lượt các mô hình
# - Nếu Input có sẵn kết quả của mô hình với **cùng config và cùng `SPLIT_FP`** (ví dụ output của lần chạy trước), nạp lại thay vì train.
# - Trước mỗi mô hình (và mỗi fold) kiểm tra ngân sách thời gian; không kịp thì bỏ qua để notebook vẫn kết thúc trước 12 giờ.
# - Một mô hình lỗi thì in đầy đủ traceback, ghi 1 dòng tóm tắt vào `FAILURES` (có trong `run_record.json` và REPORT) rồi chạy tiếp.

# %%
def find_input_models():
    found = {}
    for path in sorted(glob.glob("/kaggle/input/**/models/*/preds.npz", recursive=True)):
        src_dir = Path(path).parent
        if str(np.load(path, allow_pickle=False)["split_fp"]) != SPLIT_FP:
            print(f"⚠️ Bỏ qua {path}: khác SPLIT_FP (khác cách chia dữ liệu)")
            continue
        spec = json.loads((src_dir / "spec.json").read_text(encoding="utf-8"))
        found.setdefault(src_dir.name, []).append((src_dir, spec))
    return found


def load_input_model(key, src_dir, spec):
    npz = np.load(src_dir / "preds.npz", allow_pickle=False)
    if src_dir.resolve() != (MODELS_DIR / key).resolve():
        shutil.copytree(src_dir, MODELS_DIR / key, dirs_exist_ok=True)
    PREDS[key] = dict(oof=npz["oof"], holdout=npz["holdout"], test=npz["test"], cv_auc=float(npz["cv_auc"]))
    try:
        HISTORY[key] = pd.read_csv(src_dir / "history.csv").to_dict("records")
    except pd.errors.EmptyDataError:
        HISTORY[key] = []
    FOLD_STATS[key] = pd.read_csv(src_dir / "folds.csv").to_dict("records")
    SPECS[key] = {**spec, "reused_from": str(src_dir)}
    print(f"♻️ Nạp {key} từ {src_dir} (run {spec.get('run_id')}, CV AUC {PREDS[key]['cv_auc']:.5f})")


def short_error(exc):
    first = str(exc).strip().splitlines()
    tail = [l for l in first if "Error" in l][-1:] or first[:1]
    return f"{type(exc).__name__}: {tail[0][:300] if tail else ''}"


INPUT_MODELS = find_input_models()
print("Kết quả có sẵn trong Input:", {k: [s.get("run_id") for _, s in v] for k, v in INPUT_MODELS.items()} or "không có")

RUNNERS = {"tfidf": run_tfidf, "encoder": run_transformer, "llm": run_transformer, "guard": run_guard}
FAILURES = {}
for key in CONFIG["run_models"]:
    print(f"\n===== {key} ({datetime.now().strftime('%H:%M')}, đã chạy {elapsed_hours():.1f}h) =====")
    same = [(d, s) for d, s in INPUT_MODELS.get(key, []) if s.get("config") == MODELS[key]]
    if same:
        load_input_model(key, *same[-1])
        continue
    if elapsed_hours() + EST_HOURS[key] > CONFIG["time_budget_hours"]:
        FAILURES[key] = f"bỏ qua: không kịp (cần ~{EST_HOURS[key]}h, đã chạy {elapsed_hours():.1f}h / ngân sách {CONFIG['time_budget_hours']}h)"
        print("⏭️", FAILURES[key])
        continue
    try:
        RUNNERS[MODELS[key]["kind"]](key)
    except Exception as exc:
        FAILURES[key] = short_error(exc)
        print(f"❌ {key} lỗi, bỏ qua mô hình này và chạy tiếp:\n{traceback.format_exc()}")
    finally:
        torch.cuda.empty_cache()

for key, candidates in INPUT_MODELS.items():
    if key not in PREDS:
        load_input_model(key, *candidates[-1])
        if key in FAILURES:
            FAILURES[key] += f" → dùng tạm kết quả cũ từ run {candidates[-1][1].get('run_id')}"
        candidates = candidates[:-1]
    used = SPECS[key].get("config")
    for i, (src_dir, spec) in enumerate(c for c in candidates if c[1].get("config") != used):
        load_input_model(f"{key}@prev{i + 1}", src_dir, spec)
print("Mô hình hiện có:", list(PREDS), "| Lỗi/bỏ qua:", FAILURES or "không")

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


def is_guard(name):
    return MODELS.get(name, {}).get("kind") == "guard" or name.startswith("guard_")


ENSEMBLES, ENS_WEIGHTS = {}, {}
pools = {"ensemble_clean": {k: v for k, v in PREDS.items() if not is_guard(k)}}
if any(is_guard(k) for k in PREDS):
    pools["ensemble_all"] = dict(PREDS)
for ens_name, pool in pools.items():
    if not pool:
        continue
    w, cv = hill_climb(pool, y_dev, CONFIG["ensemble_iters"])
    ENS_WEIGHTS[ens_name] = w
    ENSEMBLES[ens_name] = dict(oof=blend(pool, w, "oof"), holdout=blend(pool, w, "holdout"), test=blend(pool, w, "test"), cv_auc=cv)
    print(f"{ens_name}: trọng số {({k: round(v, 3) for k, v in w.items() if v > 0})} | CV AUC = {cv:.5f}")
ALL = {**PREDS, **ENSEMBLES}

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
for name, r in ALL.items():
    auc = roc_auc_score(y_hold, r["holdout"])
    lo, hi = bootstrap_ci(y_hold, r["holdout"], CONFIG["n_bootstrap"], CONFIG["seed"])
    rows.append(dict(model=name, cv_auc=r["cv_auc"], holdout_auc=auc, ci95_low=lo, ci95_high=hi,
                     rank_2025=lb_rank(auc), rank_range=f"{lb_rank(hi)} … {lb_rank(lo)}"))
scores = pd.DataFrame(rows).sort_values("holdout_auc", ascending=False)
scores.round(5)

# %%
top3_line = LEADERBOARD_2025[2]


def verdict_of(name):
    r = scores.set_index("model").loc[name]
    head = (f"ĐẠT top 3 (holdout AUC {r.holdout_auc:.5f} ≥ {top3_line})" if r.holdout_auc >= top3_line
            else f"CHƯA đạt top 3: thiếu {top3_line - r.holdout_auc:.5f} so với mốc {top3_line}")
    return f"`{name}`: {head}. Hạng tương đương **{r.rank_2025}** (CI 95%: {r.rank_range})"


verdicts = [verdict_of(n) for n in ENSEMBLES]
for v in verdicts:
    print(v)

results_table = [
    "Mốc leaderboard private 2025: hạng 1 = 0.99824 · hạng 3 = 0.99658 · hạng 10 = 0.99461 · hạng 30 = 0.99049.\n",
    "| Mô hình | CV AUC (dev) | Holdout AUC | CI 95% | Hạng tương đương | Khoảng hạng |",
    "|---|---|---|---|---|---|",
]
results_table += [f"| {r.model} | {r.cv_auc:.5f} | **{r.holdout_auc:.5f}** | {r.ci95_low:.5f}–{r.ci95_high:.5f} | {r.rank_2025} | {r.rank_range} |"
                  for r in scores.itertuples()]
results_table += [""] + [f"Trọng số {n}: {json.dumps({k: round(v, 3) for k, v in w.items() if v > 0})}" for n, w in ENS_WEIGHTS.items()]
md = [
    "# So sánh với RMIT Hackathon 2025 (Challenge 1+2, ROC-AUC)\n",
    *[f"- {v}" for v in verdicts],
    "",
    *results_table,
    "",
    "**Lưu ý khi so sánh:** holdout được tách từ train của cuộc thi (cùng phân phối), còn private test 2025 là dữ liệu ẩn; "
    "các đội thi chỉ có 1–2 ngày và 10 lượt nộp. Đây là ước lượng mô hình đạt *mức* hạng nào, không phải kết quả thi chính thức.",
]
(OUT / "leaderboard_compare.md").write_text("\n".join(md), encoding="utf-8")
primary = "ensemble_all" if "ensemble_all" in ENSEMBLES else "ensemble_clean"
(OUT / "scores.json").write_text(json.dumps(dict(
    run_id=RUN_ID, verdict=verdicts, weights=ENS_WEIGHTS, models=scores.to_dict(orient="records"),
    n_dev=int(len(dev_idx)), n_holdout=int(len(hold_idx)), split_fp=SPLIT_FP, config=CONFIG, model_configs=MODELS,
), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print("\n".join(md))

# %% [markdown]
# ## Biểu đồ: quá trình train và kết quả
# Mỗi epoch đo 4 lần (`evals_per_epoch`) trên fold validation và trên 320 mẫu train cố định (`train_probe_size`),
# nên train loss và eval loss được tính cùng một cách và so trực tiếp được với nhau.
# - `training_curves.png`: mỗi mô hình một hàng; trái = train loss + eval loss (trung bình fold, vùng mờ = min–max), phải = AUC.
# - `loss_over_time.png`: loss theo **phút** thật, 5 fold nối tiếp nhau.
# - `compare_models_curves.png`: eval loss và AUC của mọi mô hình trên cùng một biểu đồ.
# - `fold_summary.png`: AUC, thời gian và VRAM đỉnh của từng fold: fold lệch hẳn là dấu hiệu train không ổn định.
# - `leaderboard_position.png`, `holdout_auc_ci.png`, `roc_holdout.png`, `auc_by_length.png`, `score_hist.png`, `ensemble_weights.png`.

# %%
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
LENGTH_BUCKETS = [(0, 20, "<20"), (20, 60, "20–59"), (60, 150, "60–149"), (150, 300, "150–299"), (300, 10**9, "300+")]


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / name, bbox_inches="tight")
    plt.show()
    plt.close(fig)


TRAIN_C, PROBE_C, VAL_C = "#9e9e9e", "#1f77b4", "#ff7f0e"


def history_frame(key):
    h = pd.DataFrame(HISTORY.get(key, []))
    for col in ("loss", "eval_val_loss", "eval_train_loss", "eval_val_roc_auc", "eval_train_roc_auc", "elapsed_min", "epoch"):
        if col not in h:
            h[col] = np.nan
    return h


def eval_mean(h, col):
    ev = h.dropna(subset=[col]).assign(e=lambda d: (d["epoch"] * CONFIG["evals_per_epoch"]).round() / CONFIG["evals_per_epoch"])
    return ev.groupby("e")[col].agg(["mean", "min", "max"])


trained = [k for k in PREDS if history_frame(k)["eval_val_loss"].notna().any()]
if trained:
    fig, axes = plt.subplots(len(trained), 2, figsize=(14, 4 * len(trained)), squeeze=False)
    for (ax_loss, ax_auc), key in zip(axes, trained):
        h = history_frame(key)
        for _, g in h.dropna(subset=["loss"]).groupby("fold"):
            ax_loss.plot(g["epoch"], g["loss"], color=TRAIN_C, alpha=0.35, lw=0.8)
        ax_loss.plot([], [], color=TRAIN_C, label="train loss lúc train (từng fold)")
        for col, color, label in (("eval_train_loss", PROBE_C, "train loss (đo lại, mẫu train cố định)"),
                                  ("eval_val_loss", VAL_C, "eval loss (fold validation)")):
            m = eval_mean(h, col)
            ax_loss.plot(m.index, m["mean"], "o-", color=color, lw=2, label=label)
            ax_loss.fill_between(m.index, m["min"], m["max"], color=color, alpha=0.15)
        ax_loss.set(title=f"{key}: train loss vs eval loss (đường = trung bình fold, vùng mờ = min–max)", xlabel="epoch", ylabel="loss")
        ax_loss.legend(fontsize=8)
        for col, color, label in (("eval_train_roc_auc", PROBE_C, "AUC trên mẫu train"), ("eval_val_roc_auc", VAL_C, "AUC fold validation")):
            m = eval_mean(h, col)
            ax_auc.plot(m.index, m["mean"], "o-", color=color, lw=2, label=label)
            ax_auc.fill_between(m.index, m["min"], m["max"], color=color, alpha=0.15)
        ax_auc.set(title=f"{key}: ROC-AUC theo epoch", xlabel="epoch", ylabel="ROC-AUC")
        ax_auc.legend(fontsize=8)
    save_fig(fig, "training_curves.png")

    fig, axes = plt.subplots(len(trained), 1, figsize=(14, 3.4 * len(trained)), squeeze=False)
    for ax, key in zip(axes[:, 0], trained):
        h = history_frame(key).sort_values(["fold", "step"])
        offset = 0.0
        for k, g in h.groupby("fold"):
            g = g.assign(t=g["elapsed_min"] + offset)
            tr, ev = g.dropna(subset=["loss"]), g.dropna(subset=["eval_val_loss"])
            ax.plot(tr["t"], tr["loss"], color=TRAIN_C, lw=1, label="train loss" if k == 0 else None)
            ax.plot(ev["t"], ev["eval_val_loss"], "o-", color=VAL_C, lw=1.5, ms=3, label="eval loss" if k == 0 else None)
            ax.axvspan(offset, offset + g["elapsed_min"].max(), color=COLORS[int(k) % len(COLORS)], alpha=0.06)
            ax.text(offset + 0.5, 0.97, f"fold {k}", va="top", fontsize=8, transform=ax.get_xaxis_transform())
            offset += g["elapsed_min"].max()
        ax.set(title=f"{key}: loss theo thời gian thật (các fold nối tiếp nhau)", xlabel="phút", ylabel="loss")
        ax.legend(fontsize=8, loc="upper right")
    save_fig(fig, "loss_over_time.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for i, key in enumerate(trained):
        h = history_frame(key)
        total = h["epoch"].max()
        for ax, col in zip(axes, ("eval_val_loss", "eval_val_roc_auc")):
            m = eval_mean(h, col)
            ax.plot(100 * m.index / total, m["mean"], "o-", color=COLORS[i % len(COLORS)], label=key)
    axes[0].set(title="Eval loss của mọi mô hình", xlabel="% quá trình train", ylabel="eval loss")
    axes[1].set(title="AUC fold validation của mọi mô hình", xlabel="% quá trình train", ylabel="ROC-AUC")
    for ax in axes:
        ax.legend(fontsize=8)
    save_fig(fig, "compare_models_curves.png")

stat_rows = [dict(model=k, **s) for k, v in FOLD_STATS.items() for s in v]
if stat_rows:
    fs = pd.DataFrame(stat_rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (key, g) in enumerate(fs.groupby("model", sort=False)):
        axes[0].scatter([i] * len(g), g["auc"], color=COLORS[i % len(COLORS)], s=40)
        axes[0].hlines(g["auc"].mean(), i - 0.3, i + 0.3, color=COLORS[i % len(COLORS)])
    axes[0].set_xticks(range(fs["model"].nunique()), fs["model"].unique(), rotation=20)
    axes[0].set(title="AUC từng fold (gạch = trung bình)", ylabel="ROC-AUC")
    agg = fs.groupby("model", sort=False).agg(minutes=("minutes", "sum"), vram=("peak_vram_gb", "max"))
    axes[1].barh(agg.index, agg["minutes"], color=COLORS[:len(agg)])
    axes[1].set(title="Tổng thời gian (phút)")
    axes[2].barh(agg.index, agg["vram"], color=COLORS[:len(agg)])
    axes[2].set(title="VRAM đỉnh (GB)")
    save_fig(fig, "fold_summary.png")

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.bar(range(1, len(LEADERBOARD_2025) + 1), LEADERBOARD_2025, color=["#f2b705"] * 3 + ["#c8c8c8"] * (len(LEADERBOARD_2025) - 3))
for i, r in enumerate(scores.itertuples()):
    ax.axhline(r.holdout_auc, ls="--", lw=1.5, color=COLORS[i % len(COLORS)], label=f"{r.model} {r.holdout_auc:.5f} ({r.rank_2025})")
ax.set_ylim(min(min(LEADERBOARD_2025), scores.holdout_auc.min()) - 0.003, 1.0)
ax.set(title="Vị trí trên private leaderboard 2025 (vàng = top 3)", xlabel="hạng", ylabel="ROC-AUC")
ax.legend(fontsize=8, loc="lower left")
save_fig(fig, "leaderboard_position.png")

fig, ax = plt.subplots(figsize=(10, 0.6 * len(scores) + 1.5))
for i, r in enumerate(scores.iloc[::-1].itertuples()):
    ax.plot([r.ci95_low, r.ci95_high], [i, i], lw=3, color=COLORS[i % len(COLORS)])
    ax.scatter(r.holdout_auc, i, s=60, color=COLORS[i % len(COLORS)], zorder=3)
    ax.scatter(r.cv_auc, i, s=60, marker="D", facecolor="white", edgecolor=COLORS[i % len(COLORS)], zorder=3)
ax.axvline(top3_line, ls="--", color="#f2b705", label="mốc top 3")
ax.set_yticks(range(len(scores)), scores.model.iloc[::-1])
ax.set(title="Holdout AUC (●, thanh = CI 95%) và CV AUC (◇)", xlabel="ROC-AUC")
ax.legend()
save_fig(fig, "holdout_auc_ci.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for i, (name, r) in enumerate(ALL.items()):
    fpr, tpr, _ = roc_curve(y_hold, r["holdout"])
    for ax in axes:
        ax.plot(fpr, tpr, color=COLORS[i % len(COLORS)], label=f"{name} ({roc_auc_score(y_hold, r['holdout']):.4f})")
axes[0].plot([0, 1], [0, 1], ls=":", color="grey")
axes[0].set(title="ROC trên holdout", xlabel="FPR", ylabel="TPR")
axes[1].set(title="Phóng to FPR ≤ 0.1", xlabel="FPR", xlim=(0, 0.1), ylim=(0.6, 1.0))
axes[1].legend(fontsize=8, loc="lower right")
save_fig(fig, "roc_holdout.png")

n_words_dev = np.array([len(t.split()) for t in X_dev])
bucket_rows = []
for lo, hi, label in LENGTH_BUCKETS:
    m = (n_words_dev >= lo) & (n_words_dev < hi)
    row = dict(bucket=label, n=int(m.sum()), n_jailbreak=int(y_dev[m].sum()))
    for name, r in ALL.items():
        row[name] = roc_auc_score(y_dev[m], r["oof"][m]) if 0 < y_dev[m].sum() < m.sum() else np.nan
    bucket_rows.append(row)
by_length = pd.DataFrame(bucket_rows)
fig, ax = plt.subplots(figsize=(12, 4.5))
width = 0.8 / len(ALL)
for i, name in enumerate(ALL):
    ax.bar(np.arange(len(by_length)) + i * width, by_length[name], width, label=name, color=COLORS[i % len(COLORS)])
ax.set_xticks(np.arange(len(by_length)) + 0.4 - width / 2, [f"{r.bucket}\n(n={r.n}, jb={r.n_jailbreak})" for r in by_length.itertuples()])
ax.set_ylim(max(0.5, np.nanmin(by_length[list(ALL)].values) - 0.02), 1.0)
ax.set(title="AUC theo độ dài prompt (OOF dev)", xlabel="số từ", ylabel="ROC-AUC")
ax.legend(fontsize=8)
save_fig(fig, "auc_by_length.png")

fig, ax = plt.subplots(figsize=(10, 4))
ens = ALL[primary]["oof"]
ax.hist(ens[y_dev == 0], bins=30, alpha=0.6, color="#2ca02c", label="benign")
ax.hist(ens[y_dev == 1], bins=30, alpha=0.6, color="#d62728", label="jailbreak")
ax.set(title=f"Phân bố điểm OOF của {primary} theo nhãn thật (vùng chồng lấn = lỗi xếp hạng)", xlabel="điểm (rank)", ylabel="số mẫu")
ax.legend()
save_fig(fig, "score_hist.png")

fig, axes = plt.subplots(1, len(ENS_WEIGHTS), figsize=(6 * len(ENS_WEIGHTS), 3.5), squeeze=False)
for ax, (name, w) in zip(axes[0], ENS_WEIGHTS.items()):
    items = sorted(((k, v) for k, v in w.items() if v > 0), key=lambda kv: kv[1])
    ax.barh([k for k, _ in items], [v for _, v in items], color=COLORS[:len(items)])
    ax.set(title=f"Trọng số {name}", xlim=(0, 1))
save_fig(fig, "ensemble_weights.png")

# %% [markdown]
# ## Phân tích lỗi trên holdout
# Xem các mẫu bị xếp sai nhiều nhất để biết cần thêm mô hình/dữ liệu kiểu gì.

# %%
hold_scores = ALL[primary]["holdout"]
err = pd.DataFrame({"text": X_hold, "label": y_hold, "score": hold_scores})
false_neg = err[err.label == 1].nsmallest(15, "score").assign(error="jailbreak bị chấm thấp")
false_pos = err[err.label == 0].nlargest(15, "score").assign(error="benign bị chấm cao")
errors = pd.concat([false_neg, false_pos])
errors.to_csv(OUT / "holdout_errors.csv", index=False)
pd.DataFrame({"text": X_dev, "label": y_dev, **{f"oof_{k}": v["oof"] for k, v in ALL.items()}}).to_csv(OUT / "oof_predictions.csv", index=False)
pd.DataFrame({"text": X_hold, "label": y_hold, **{f"holdout_{k}": v["holdout"] for k, v in ALL.items()}}).to_csv(OUT / "holdout_predictions.csv", index=False)
for row in errors.head(6).itertuples():
    print(f"[{row.error} | score={row.score:.3f}] {row.text[:250]!r}\n")

# %% [markdown]
# ## REPORT.md và run_record.json
# Báo cáo đầy đủ của lần chạy: specs từng mô hình, kết quả, biểu đồ, chẩn đoán, lỗi tiêu biểu.

# %%
def diagnose(key):
    stats = pd.DataFrame(FOLD_STATS.get(key, []))
    notes = []
    if len(stats) and stats["auc"].min() < 0.6:
        notes.append("có fold AUC < 0.6: train sụp, giảm lr hoặc kiểm tra FP16")
    if len(stats) > 1 and stats["auc"].std() > 0.01:
        notes.append(f"AUC giữa các fold dao động mạnh (std {stats['auc'].std():.4f})")
    h = history_frame(key)
    if h["loss"].notna().any():
        last = h.dropna(subset=["loss"]).groupby("fold")["loss"].last()
        if not np.isfinite(last).all():
            notes.append("loss NaN/inf")
    if h["eval_val_loss"].notna().any():
        val = h.dropna(subset=["eval_val_loss"]).groupby("fold")["eval_val_loss"]
        if ((val.last() - val.min()) / val.min() > 0.10).mean() >= 0.5:
            notes.append("eval loss tăng lại > 10% so với đáy ở ≥ một nửa số fold: overfit, thử ít epoch hơn hoặc lr thấp hơn")
        probe = h.dropna(subset=["eval_train_loss"]).groupby("fold")["eval_train_loss"].last()
        gap = float((val.last() - probe).mean())
        notes.append(f"khoảng cách eval − train loss cuối = {gap:.3f}")
        if probe.mean() > 0.3:
            notes.append("train loss cuối còn cao (> 0.3): underfit, thử thêm epoch hoặc lr cao hơn")
    if h["eval_val_roc_auc"].notna().any():
        ev = h.dropna(subset=["eval_val_roc_auc"]).groupby("fold")["eval_val_roc_auc"]
        if ((ev.max() - ev.last()) > 0.003).mean() >= 0.5:
            notes.append("AUC validation giảm ở cuối ở ≥ một nửa số fold: có thể overfit")
    return "; ".join(notes) or "ổn định"


def fmt_spec(key):
    spec = SPECS.get(key, {})
    cfg = spec.get("config") or MODELS.get(key, {})
    shown = {k: v for k, v in cfg.items() if k not in ("kind",)}
    origin = f" · ♻️ nạp lại từ run `{spec.get('run_id')}`" if spec.get("reused_from") else ""
    return f"`{key}` ({cfg.get('kind', '?')}): " + ", ".join(f"{k}={v}" for k, v in shown.items()) + \
        (f" · derived: {json.dumps(spec.get('derived', {}), default=str)}" if spec.get("derived") else "") + origin


report = [
    f"# Báo cáo lần chạy `{RUN_ID}`",
    "",
    *[f"- {v}" for v in verdicts],
    *[f"- ⚠️ `{k}` không có trong kết quả: {msg}" for k, msg in FAILURES.items()],
    "",
    "## 1. Specs",
    "",
    f"- Dữ liệu: train {len(texts_all)} mẫu → dev {len(dev_idx)} + holdout {len(hold_idx)} · test {len(test_texts)} · SPLIT_FP `{SPLIT_FP}`",
    f"- CV: StratifiedGroupKFold {CONFIG['n_folds']} fold, seed {CONFIG['seed']} · ensemble hill-climbing {CONFIG['ensemble_iters']} vòng trên OOF",
    f"- Môi trường: {json.dumps(environment(), default=str)}",
    "",
    *[f"- {fmt_spec(k)}" for k in PREDS],
    "",
    "## 2. Kết quả",
    "",
    *results_table,
    "",
    "![](charts/leaderboard_position.png)",
    "",
    "![](charts/holdout_auc_ci.png)",
    "",
    "## 3. Quá trình train",
    "",
    "| Mô hình | Tổng phút | VRAM đỉnh (GB) | AUC fold (min / mean / max) | Chẩn đoán |",
    "|---|---|---|---|---|",
]
for key in PREDS:
    s = pd.DataFrame(FOLD_STATS.get(key, []))
    if len(s):
        report.append(f"| {key} | {s['minutes'].sum():.1f} | {s['peak_vram_gb'].max():.1f} | "
                      f"{s['auc'].min():.4f} / {s['auc'].mean():.4f} / {s['auc'].max():.4f} | {diagnose(key)} |")
report += [
    "",
    *(["**Cách đọc:** xám = loss lúc train; xanh = train loss đo lại trên mẫu train cố định; cam = eval loss trên fold validation. "
        "Hai đường xanh và cam cùng giảm là học tốt; cam đi lên trong khi xanh vẫn giảm là overfit; cả hai cùng cao là underfit.",
        "",
        "![](charts/training_curves.png)",
        "",
        "![](charts/loss_over_time.png)",
        "",
        "![](charts/compare_models_curves.png)"] if trained else ["_(không có mô hình nào train theo bước trong lần chạy này)_"]),
    "",
    "![](charts/fold_summary.png)",
    "",
    "## 4. ROC, độ dài prompt, phân bố điểm",
    "",
    "![](charts/roc_holdout.png)",
    "",
    "| " + " | ".join(by_length.columns) + " |",
    "|" + "---|" * len(by_length.columns),
    *["| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |" for row in by_length.itertuples(index=False)],
    "",
    "![](charts/auc_by_length.png)",
    "",
    "![](charts/score_hist.png)",
    "",
    "![](charts/ensemble_weights.png)",
    "",
    f"## 5. Mẫu sai nặng nhất trên holdout ({primary})",
    "",
    "| Loại lỗi | Điểm | Prompt (rút gọn) |",
    "|---|---|---|",
    *[f"| {e.error} | {e.score:.3f} | {' '.join(e.text.split())[:220].replace('|', '/')} |" for e in errors.itertuples()],
]
(OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
(OUT / "run_record.json").write_text(json.dumps(dict(
    schema_version=1, run_id=RUN_ID, created_at=datetime.now(timezone.utc).isoformat(), split_fp=SPLIT_FP,
    config=CONFIG, environment=environment(), models={k: SPECS.get(k) for k in PREDS},
    fold_stats=FOLD_STATS, scores=scores.to_dict(orient="records"), weights=ENS_WEIGHTS,
    diagnosis={k: diagnose(k) for k in PREDS}, failures=FAILURES,
), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print("Đã ghi", OUT / "REPORT.md")

# %% [markdown]
# ## File submission (không nộp, để sẵn nếu sau này muốn thử Late Submission)

# %%
sample_path = glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)
id_col, target_col = (pd.read_csv(sample_path[0]).columns[:2].tolist() if sample_path else [TEST_ID, "target"])
for name, r in ENSEMBLES.items():
    pd.DataFrame({id_col: test_df[TEST_ID], target_col: r["test"]}).to_csv(OUT / f"submission_{name}.csv", index=False)
shutil.copy(OUT / f"submission_{primary}.csv", OUT / "submission.csv")
print("Đã ghi submission_*.csv; submission.csv =", primary)
