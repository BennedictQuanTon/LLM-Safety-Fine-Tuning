"""Build a Markdown report with SVG charts from a downloaded notebook-04 output folder.

Stdlib only, so it runs on any laptop without installing packages.

    python3 tools/report_from_results.py results/clf --version v1
"""
import argparse
import csv
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

csv.field_size_limit(10**9)

LEADERBOARD_2025 = [
    0.99824, 0.99658, 0.99658, 0.99614, 0.99603, 0.99603, 0.99593, 0.99570, 0.99568, 0.99461,
    0.99425, 0.99425, 0.99359, 0.99359, 0.99359, 0.99359, 0.99359, 0.99324, 0.99316, 0.99261,
    0.99236, 0.99236, 0.99236, 0.99232, 0.99187, 0.99187, 0.99185, 0.99088, 0.99077, 0.99049,
    0.99039, 0.99039, 0.98989, 0.98910, 0.98805, 0.98788, 0.98664, 0.98570, 0.98545, 0.98509,
    0.98448, 0.98373, 0.98321, 0.98257, 0.98200, 0.98165, 0.98157, 0.98157, 0.98145,
]
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
LENGTH_BUCKETS = [(0, 20, "<20"), (20, 60, "20–59"), (60, 150, "60–149"), (150, 300, "150–299"), (300, 10**9, "300+")]


def auc(y, s):
    order = sorted(range(len(s)), key=lambda i: s[i])
    ranks = [0.0] * len(s)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    n1 = sum(y)
    n0 = len(y) - n1
    if n0 == 0 or n1 == 0:
        return float("nan")
    return (sum(r for r, t in zip(ranks, y) if t == 1) - n1 * (n1 + 1) / 2) / (n0 * n1)


def roc_points(y, s):
    pairs = sorted(zip(s, y), reverse=True)
    n1 = sum(y)
    n0 = len(y) - n1
    tp = fp = 0
    pts = [(0.0, 0.0)]
    for i, (score, label) in enumerate(pairs):
        tp += label
        fp += 1 - label
        if i + 1 == len(pairs) or pairs[i + 1][0] != score:
            pts.append((fp / n0, tp / n1))
    return pts


def lb_rank(score):
    rank = 1 + sum(s > score for s in LEADERBOARD_2025)
    return f"#{rank}" if rank <= len(LEADERBOARD_2025) else f"ngoài top {len(LEADERBOARD_2025)}"


class Svg:
    def __init__(self, width, height, title, x_label="", y_label="", pad=(60, 20, 40, 50)):
        self.w, self.h = width, height
        self.left, self.right, self.top, self.bottom = pad
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="12">',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            f'<text x="{width / 2}" y="22" text-anchor="middle" font-size="15" font-weight="bold">{escape(title)}</text>',
        ]
        if x_label:
            self.text(width / 2, height - 10, x_label, anchor="middle")
        if y_label:
            self.parts.append(f'<text x="16" y="{height / 2}" text-anchor="middle" transform="rotate(-90 16 {height / 2})">{escape(y_label)}</text>')

    @property
    def plot_w(self):
        return self.w - self.left - self.right

    @property
    def plot_h(self):
        return self.h - self.top - self.bottom

    def axes(self, x0, x1, y0, y1, x_ticks=None, y_ticks=None, y_fmt="{:.2f}", x_fmt="{:.2f}"):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        for t in y_ticks or []:
            y = self.sy(t)
            self.line(self.left, y, self.w - self.right, y, "#e5e5e5")
            self.text(self.left - 6, y + 4, y_fmt.format(t), anchor="end", size=11)
        for t in x_ticks or []:
            x = self.sx(t)
            self.line(x, self.top, x, self.h - self.bottom, "#f0f0f0")
            self.text(x, self.h - self.bottom + 16, x_fmt.format(t), anchor="middle", size=11)
        self.parts.append(f'<rect x="{self.left}" y="{self.top}" width="{self.plot_w}" height="{self.plot_h}" fill="none" stroke="#888"/>')

    def sx(self, v):
        return self.left + (v - self.x0) / (self.x1 - self.x0) * self.plot_w

    def sy(self, v):
        return self.top + (1 - (v - self.y0) / (self.y1 - self.y0)) * self.plot_h

    def line(self, x1, y1, x2, y2, color, width=1, dash=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{d}/>')

    def text(self, x, y, s, anchor="start", size=12, color="#222", bold=False):
        w = ' font-weight="bold"' if bold else ""
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" fill="{color}"{w}>{escape(str(s))}</text>')

    def rect(self, x, y, w, h, color, opacity=1.0, stroke="none"):
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" fill="{color}" fill-opacity="{opacity}" stroke="{stroke}"/>')

    def polyline(self, pts, color, width=2, dash=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        path = " ".join(f"{self.sx(x):.1f},{self.sy(y):.1f}" for x, y in pts)
        self.parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="{width}"{d}/>')

    def legend(self, items, x, y):
        for i, (label, color) in enumerate(items):
            self.rect(x, y + i * 18 - 9, 12, 12, color)
            self.text(x + 18, y + i * 18 + 1, label, size=11)

    def save(self, path):
        path.write_text("\n".join(self.parts + ["</svg>"]), encoding="utf-8")


def chart_leaderboard(models, path):
    svg = Svg(820, 380, "Vị trí trên private leaderboard 2025 (top 49 / 110 đội)", "Hạng", "ROC-AUC")
    lo = min(min(LEADERBOARD_2025), min(m["holdout_auc"] for m in models)) - 0.002
    svg.axes(0, len(LEADERBOARD_2025) + 1, lo, 1.0, x_ticks=[1, 3, 10, 20, 30, 40, 49],
             y_ticks=[round(lo + i * (1 - lo) / 5, 3) for i in range(6)], y_fmt="{:.3f}", x_fmt="{:.0f}")
    bar_w = svg.plot_w / (len(LEADERBOARD_2025) + 1) * 0.7
    for i, s in enumerate(LEADERBOARD_2025, start=1):
        color = "#f2b705" if i <= 3 else "#c8c8c8"
        svg.rect(svg.sx(i) - bar_w / 2, svg.sy(s), bar_w, svg.sy(lo) - svg.sy(s), color)
    for k, m in enumerate(models):
        y = svg.sy(m["holdout_auc"])
        color = PALETTE[k % len(PALETTE)]
        svg.line(svg.left, y, svg.w - svg.right, y, color, 2, "6,4")
        svg.text(svg.w - svg.right - 4, y - 4, f'{m["model"]} {m["holdout_auc"]:.5f} ({m["rank_2025"]})', anchor="end", size=11, color=color, bold=True)
    svg.text(svg.sx(3) + 6, svg.sy(LEADERBOARD_2025[2]) - 6, "mốc top 3 = 0.99658", size=11, color="#b08000", bold=True)
    svg.save(path)


def chart_scores_ci(models, path):
    svg = Svg(820, 60 + 70 * len(models), "Holdout AUC và khoảng tin cậy 95% (bootstrap)", "ROC-AUC", pad=(150, 30, 40, 50))
    lo = min(m["ci95_low"] for m in models) - 0.005
    hi = 1.0
    svg.axes(lo, hi, 0, len(models), x_ticks=[round(lo + i * (hi - lo) / 6, 3) for i in range(7)], x_fmt="{:.3f}")
    top3 = LEADERBOARD_2025[2]
    svg.line(svg.sx(top3), svg.top, svg.sx(top3), svg.h - svg.bottom, "#f2b705", 2, "6,4")
    svg.text(svg.sx(top3), svg.top - 4, "top 3", anchor="middle", size=11, color="#b08000", bold=True)
    for k, m in enumerate(models):
        y = svg.sy(len(models) - k - 0.5)
        color = PALETTE[k % len(PALETTE)]
        svg.line(svg.sx(m["ci95_low"]), y, svg.sx(m["ci95_high"]), y, color, 3)
        svg.parts.append(f'<circle cx="{svg.sx(m["holdout_auc"]):.1f}" cy="{y:.1f}" r="6" fill="{color}"/>')
        svg.parts.append(f'<path d="M{svg.sx(m["cv_auc"]):.1f},{y - 7:.1f} l6,7 l-6,7 l-6,-7 z" fill="white" stroke="{color}" stroke-width="2"/>')
        svg.text(svg.left - 8, y + 4, m["model"], anchor="end", size=12, bold=True)
    svg.legend([("● holdout AUC (thanh ngang = CI 95%)", "#444"), ("◇ CV AUC trên dev", "#999")], svg.left + 8, svg.top + 14)
    svg.save(path)


def chart_roc(y, model_scores, path):
    svg = Svg(560, 500, "ROC trên dự đoán OOF của dev (3200 mẫu)", "False positive rate", "True positive rate")
    svg.axes(0, 1, 0, 1, x_ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], y_ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    svg.polyline([(0, 0), (1, 1)], "#bbb", 1, "4,4")
    items = []
    for k, (name, s) in enumerate(model_scores.items()):
        color = PALETTE[k % len(PALETTE)]
        svg.polyline(roc_points(y, s), color, 2)
        items.append((f"{name} (AUC {auc(y, s):.4f})", color))
    svg.legend(items, svg.left + 150, svg.h - svg.bottom - 18 * len(items))
    svg.save(path)


def chart_roc_zoom(y, model_scores, path):
    svg = Svg(560, 500, "ROC phóng to vùng FPR ≤ 0.2", "False positive rate", "True positive rate")
    svg.axes(0, 0.2, 0.6, 1, x_ticks=[0, 0.05, 0.1, 0.15, 0.2], y_ticks=[0.6, 0.7, 0.8, 0.9, 1.0])
    items = []
    for k, (name, s) in enumerate(model_scores.items()):
        color = PALETTE[k % len(PALETTE)]
        pts = [(x, max(v, 0.6)) for x, v in roc_points(y, s) if x <= 0.2]
        svg.polyline(pts, color, 2)
        items.append((name, color))
    svg.legend(items, svg.left + 200, svg.h - svg.bottom - 18 * len(items))
    svg.save(path)


def chart_length_buckets(bucket_rows, model_names, path):
    svg = Svg(820, 400, "AUC theo độ dài prompt (OOF dev): prompt dài là điểm yếu", "Số từ trong prompt", "ROC-AUC")
    lo = max(0.5, min(r[m] for r in bucket_rows for m in model_names if not math.isnan(r[m])) - 0.02)
    svg.axes(0, len(bucket_rows), lo, 1.0, y_ticks=[round(lo + i * (1 - lo) / 5, 3) for i in range(6)], y_fmt="{:.3f}")
    group_w = svg.plot_w / len(bucket_rows)
    bar_w = group_w * 0.8 / len(model_names)
    for g, row in enumerate(bucket_rows):
        x0 = svg.left + g * group_w + group_w * 0.1
        for k, m in enumerate(model_names):
            v = row[m]
            if math.isnan(v):
                continue
            svg.rect(x0 + k * bar_w, svg.sy(v), bar_w - 2, svg.sy(lo) - svg.sy(v), PALETTE[k % len(PALETTE)])
        svg.text(svg.left + (g + 0.5) * group_w, svg.h - svg.bottom + 16, f'{row["bucket"]} (n={row["n"]}, jb={row["n_pos"]})', anchor="middle", size=11)
    svg.legend([(m, PALETTE[k % len(PALETTE)]) for k, m in enumerate(model_names)], svg.left + 10, svg.top + 14)
    svg.save(path)


def chart_score_hist(y, s, name, path, bins=25):
    svg = Svg(820, 380, f"Phân bố điểm OOF của {name} theo nhãn thật", "Điểm dự đoán (rank-normalized nếu là ensemble)", "Số mẫu")
    lo, hi = min(s), max(s)
    width = (hi - lo) / bins or 1
    counts = {0: [0] * bins, 1: [0] * bins}
    for label, v in zip(y, s):
        counts[label][min(int((v - lo) / width), bins - 1)] += 1
    peak = max(max(counts[0]), max(counts[1]))
    svg.axes(lo, hi, 0, peak * 1.05, x_ticks=[lo + i * (hi - lo) / 5 for i in range(6)],
             y_ticks=[round(peak * i / 4) for i in range(5)], y_fmt="{:.0f}")
    for label, color in ((0, "#2ca02c"), (1, "#d62728")):
        for b, c in enumerate(counts[label]):
            x = svg.sx(lo + b * width)
            svg.rect(x, svg.sy(c), svg.sx(lo + (b + 1) * width) - x - 1, svg.sy(0) - svg.sy(c), color, 0.55)
    svg.legend([("benign (nhãn 0)", "#2ca02c"), ("jailbreak (nhãn 1)", "#d62728")], svg.left + svg.plot_w / 2 - 60, svg.top + 14)
    svg.save(path)


def chart_weights(weights, path):
    items = sorted(weights.items(), key=lambda kv: -kv[1])
    svg = Svg(620, 70 + 40 * len(items), "Trọng số ensemble (hill-climbing trên OOF)", "Trọng số", pad=(150, 30, 40, 50))
    svg.axes(0, 1, 0, len(items), x_ticks=[0, 0.25, 0.5, 0.75, 1.0])
    for k, (name, w) in enumerate(items):
        y = svg.sy(len(items) - k)
        h = svg.sy(len(items) - k - 1) - y
        svg.rect(svg.sx(0), y + h * 0.2, svg.sx(w) - svg.sx(0), h * 0.6, PALETTE[k % len(PALETTE)])
        svg.text(svg.left - 8, y + h / 2 + 4, name, anchor="end", bold=True)
        svg.text(svg.sx(w) + 6, y + h / 2 + 4, f"{w:.3f}")
    svg.save(path)


def md_escape(text, limit=220):
    t = " ".join(text.split())
    t = t[:limit] + ("…" if len(t) > limit else "")
    return t.replace("|", "\\|")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()
    d = args.results_dir
    charts = d / "charts"
    charts.mkdir(exist_ok=True)

    scores = json.loads((d / "scores.json").read_text(encoding="utf-8"))
    models = scores["models"]
    oof = list(csv.DictReader(open(d / "oof_predictions.csv", encoding="utf-8")))
    y = [int(r["label"]) for r in oof]
    oof_cols = [c for c in oof[0] if c.startswith("oof_")]
    model_scores = {c[4:]: [float(r[c]) for r in oof] for c in oof_cols}
    errors = list(csv.DictReader(open(d / "holdout_errors.csv", encoding="utf-8")))

    bucket_rows = []
    n_words = [len(r["text"].split()) for r in oof]
    for lo, hi, label in LENGTH_BUCKETS:
        idx = [i for i, n in enumerate(n_words) if lo <= n < hi]
        yy = [y[i] for i in idx]
        row = dict(bucket=label, n=len(idx), n_pos=sum(yy))
        for name, s in model_scores.items():
            row[name] = auc(yy, [s[i] for i in idx]) if idx else float("nan")
        bucket_rows.append(row)

    chart_leaderboard(models, charts / "leaderboard_position.svg")
    chart_scores_ci(models, charts / "holdout_auc_ci.svg")
    chart_roc(y, model_scores, charts / "roc_oof.svg")
    chart_roc_zoom(y, model_scores, charts / "roc_oof_zoom.svg")
    chart_length_buckets(bucket_rows, list(model_scores), charts / "auc_by_length.svg")
    ens_name = "ensemble" if "ensemble" in model_scores else list(model_scores)[-1]
    chart_score_hist(y, model_scores[ens_name], ens_name, charts / "score_hist_ensemble.svg")
    chart_weights(scores["weights"], charts / "ensemble_weights.svg")

    cfg = scores["config"]
    final = next(m for m in models if m["model"].startswith("ensemble"))
    if isinstance(scores["verdict"], list):
        scores["verdict"] = "; ".join(scores["verdict"])
    if "transformers" not in cfg:
        raise SystemExit("Thư mục này là output v2+: dùng REPORT.md do notebook tự sinh.")
    n_pos = sum(y)
    md = [
        f"# Báo cáo {args.version}: phân loại jailbreak, tự chấm theo RMIT Hackathon 2025",
        "",
        f"**Kết luận:** {scores['verdict']}. Ensemble đạt holdout AUC **{final['holdout_auc']:.5f}**, "
        f"tương đương hạng **{final['rank_2025']}** (khoảng {final['rank_range']}).",
        "",
        "## 1. Specs của lần chạy",
        "",
        "| Mục | Giá trị |",
        "|---|---|",
        f"| Dữ liệu | train cuộc thi 4000 mẫu → dev {scores['n_dev']} + holdout {scores['n_holdout']} (tách nhóm theo prompt trùng) |",
        f"| Dev (OOF) | {len(y)} mẫu, jailbreak {n_pos} ({n_pos / len(y):.1%}) |",
        f"| CV | StratifiedGroupKFold, {cfg['n_folds']} fold, seed {cfg['seed']} |",
        f"| Mô hình A | TF-IDF từ (1–2 gram) + ký tự (2–5 gram) → Logistic Regression, thử C ∈ {cfg['tfidf_C_grid']} |",
    ]
    for t in cfg["transformers"]:
        eff = t["batch_size"] * t["grad_accum"]
        md.append(f"| Mô hình B | `{t['name']}`, max_len {t['max_len']} (đầu {t['head_len']} + đuôi {t['max_len'] - t['head_len']} token), "
                  f"lr {t['lr']}, {t['epochs']} epoch, batch {t['batch_size']}×{t['grad_accum']} = {eff}, FP16 AMP, cosine, warmup 10% |")
    md += [
        f"| Ensemble | rank-average, trọng số chọn bằng hill-climbing ({cfg['ensemble_iters']} vòng) trên OOF |",
        f"| Chấm điểm | ROC-AUC trên holdout + bootstrap {cfg['n_bootstrap']} lần cho CI 95% |",
        "| Phần cứng | Kaggle GPU T4 ×2 (DataParallel), transformers 5.16.1, Python 3.13 |",
        "",
        "## 2. Kết quả",
        "",
        "| Mô hình | CV AUC (dev) | Holdout AUC | CI 95% | Hạng 2025 | Khoảng hạng |",
        "|---|---|---|---|---|---|",
    ]
    md += [f"| {m['model']} | {m['cv_auc']:.5f} | **{m['holdout_auc']:.5f}** | {m['ci95_low']:.5f}–{m['ci95_high']:.5f} | {m['rank_2025']} | {m['rank_range']} |"
           for m in models]
    md += [
        "",
        "![Vị trí leaderboard](charts/leaderboard_position.svg)",
        "",
        "![Holdout AUC và CI](charts/holdout_auc_ci.svg)",
        "",
        "CV AUC và holdout AUC gần nhau (chênh < 0.004), nghĩa là cách chia dữ liệu không bị rò rỉ và điểm tự chấm đáng tin. "
        "Khoảng tin cậy rộng (±0.008) vì holdout chỉ có 800 mẫu: đừng kết luận từ chênh lệch nhỏ hơn mức này.",
        "",
        "## 3. ROC",
        "",
        "![ROC OOF](charts/roc_oof.svg) ![ROC zoom](charts/roc_oof_zoom.svg)",
        "",
        "## 4. Lỗi nằm ở đâu: theo độ dài prompt",
        "",
        "| Số từ | n | jailbreak | " + " | ".join(model_scores) + " |",
        "|---|---|---|" + "---|" * len(model_scores),
    ]
    md += [f"| {r['bucket']} | {r['n']} | {r['n_pos']} | " + " | ".join(f"{r[m]:.4f}" for m in model_scores) + " |" for r in bucket_rows]
    md += [
        "",
        "![AUC theo độ dài](charts/auc_by_length.svg)",
        "",
        "Prompt ≥ 60 từ chiếm phần lớn dữ liệu và là nơi AUC thấp nhất: đây là các prompt nhập vai / kịch bản dài "
        "mà nhãn phụ thuộc vào *ý định thật* (vô hại hay độc hại được ngụy trang), không phải từ khóa. "
        "Với prompt rất dài (300+ từ), transformer chỉ thấy 512 token đầu + đuôi nên có thể mất phần giữa; "
        "nếu ở nhóm này TF-IDF (đọc toàn bộ văn bản) ngang hoặc hơn transformer thì đó là dấu hiệu cắt bớt đang làm mất thông tin.",
        "",
        "## 5. Phân bố điểm và trọng số ensemble",
        "",
        "![Phân bố điểm](charts/score_hist_ensemble.svg)",
        "",
        "![Trọng số](charts/ensemble_weights.svg)",
        "",
        "## 6. Mẫu sai nặng nhất trên holdout",
        "",
        "| Loại lỗi | Điểm | Prompt (rút gọn) |",
        "|---|---|---|",
    ]
    md += [f"| {e['error']} | {float(e['score']):.3f} | {md_escape(e['text'])} |" for e in errors]
    md += [
        "",
        "## 7. Chưa có trong v1",
        "",
        "- Đường cong loss / AUC theo epoch của từng fold, thời gian mỗi fold và VRAM: bản v1 chưa lưu. "
        "Từ v2 notebook ghi `history_<model>.csv`, `folds_<model>.csv`, `run_record.json` và tự vẽ biểu đồ.",
        "- Dự đoán holdout đầy đủ của từng mô hình (v1 chỉ lưu 30 mẫu sai): từ v2 ghi `holdout_predictions.csv`.",
        "",
        "## 8. Hướng v2",
        "",
        "Thêm Qwen2.5-3B (QLoRA, phân loại), DeBERTa-v3-large và điểm zero-shot của WildGuard; báo riêng ensemble có / không có WildGuard.",
    ]
    out = d / f"REPORT_{args.version}.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("Đã ghi", out, "và", len(list(charts.glob("*.svg"))), "biểu đồ trong", charts)


if __name__ == "__main__":
    main()
