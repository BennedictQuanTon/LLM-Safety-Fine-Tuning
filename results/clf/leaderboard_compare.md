# So sánh với RMIT Hackathon 2025 (Challenge 1+2, ROC-AUC)

- `ensemble_clean`: CHƯA đạt top 3: thiếu 0.00556 so với mốc 0.99658. Hạng tương đương **#28** (CI 95%: #7 … #41)

Mốc leaderboard private 2025: hạng 1 = 0.99824 · hạng 3 = 0.99658 · hạng 10 = 0.99461 · hạng 30 = 0.99049.

| Mô hình | CV AUC (dev) | Holdout AUC | CI 95% | Hạng tương đương | Khoảng hạng |
|---|---|---|---|---|---|
| ensemble_clean | 0.98983 | **0.99102** | 0.98473–0.99598 | #28 | #7 … #41 |
| qwen2.5-3b | 0.98172 | **0.99042** | 0.98533–0.99474 | #31 | #10 … #40 |
| deberta-v3-base | 0.98338 | **0.98364** | 0.97490–0.99096 | #43 | #28 … ngoài top 49 (/110) |
| tfidf | 0.96050 | **0.96207** | 0.94806–0.97393 | ngoài top 49 (/110) | ngoài top 49 (/110) … ngoài top 49 (/110) |

Trọng số ensemble_clean: {"tfidf": 0.2, "deberta-v3-base": 0.4, "qwen2.5-3b": 0.4}

**Lưu ý khi so sánh:** holdout được tách từ train của cuộc thi (cùng phân phối), còn private test 2025 là dữ liệu ẩn; các đội thi chỉ có 1–2 ngày và 10 lượt nộp. Đây là ước lượng mô hình đạt *mức* hạng nào, không phải kết quả thi chính thức.