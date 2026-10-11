# Kế hoạch v3: phân loại jailbreak

> Kết quả v2 nằm ở [clf/REPORT.md](clf/REPORT.md). Các con số v3 trong file này là **dự đoán**, phải chờ chạy xong mới biết thật.

## 1. v2 đang ở đâu

| | v1 | v2 |
|---|---|---|
| Ensemble ROC-AUC (holdout 800 mẫu) | 0.98480 | **0.99102** |
| Hạng tương đương leaderboard 2025 | #41 | **#28** (CI 95%: #7 … #41) |
| Accuracy (ngưỡng chọn trên dev) | — | **95.75%** (sai 34/800) |
| Thiếu bao nhiêu so với top 3 (0.99658) | 0.0118 | **0.0056** |

Ở v2 có 3 mô hình chạy được: TF-IDF, DeBERTa-base và **Qwen2.5-3B**. Qwen là mô hình mạnh nhất, một mình đạt 0.99042. Có 2 mô hình không chạy:
- **DeBERTa-large**: hết bộ nhớ GPU (OutOfMemory trên GPU 0, đã dùng hết 14.5 GB).
- **WildGuard**: không có `HF_TOKEN` nên tự bỏ qua.

## 2. Bằng chứng từ biểu đồ v2, và v3 sửa gì

### 2.1. Qwen chưa học xong → train 2 epoch

**Biểu đồ v2 cho thấy gì:**
- Đến cuối epoch 1, AUC trên validation vẫn đang đi lên (0.971 → 0.979 → 0.982), chưa chững lại.
- Khoảng cách giữa eval loss và train loss chỉ 0.088. Nghĩa là mô hình chưa học thuộc lòng, vẫn còn chỗ để học thêm.

**v3 làm gì:** train Qwen **2 epoch** (v2 chỉ 1).

**Thêm một điểm cộng:** Qwen v2 (1 epoch) không bị bỏ đi mà được gộp chung dưới tên `qwen2.5-3b@prev1`. Hai phiên bản của cùng một mô hình ở hai thời điểm train khác nhau thường sai hơi khác nhau, nên gộp lại có lợi.

### 2.2. DeBERTa-large hết bộ nhớ → chia nhỏ batch

**Vì sao hết bộ nhớ:** mô hình large to gấp khoảng 2.4 lần bản base. Mỗi GPU phải giữ cùng lúc 4 prompt dài 512 token, cộng toàn bộ kết quả trung gian, nên vượt 15 GB.

**v3 sửa:**

| | v2 | v3 |
|---|---|---|
| Số prompt mỗi lượt trên 1 GPU | 4 | **2** |
| Gom bao nhiêu lượt rồi mới cập nhật | 4 | **8** (nên tổng số prompt mỗi lần cập nhật vẫn như cũ) |
| Gradient checkpointing | tắt | **bật**: không giữ hết kết quả trung gian, khi cần thì tính lại. Chậm hơn khoảng 30% nhưng đỡ tốn bộ nhớ nhiều |
| `expandable_segments` | tắt | **bật**: giảm tình trạng bộ nhớ GPU bị phân mảnh |
| Số epoch | 3 | **2**, vì bản base đã cho thấy sau epoch 2 bắt đầu học thuộc lòng (xem 2.3) |

### 2.3. DeBERTa-base overfit sau epoch 2 → giữ nguyên, dùng lại kết quả v2

**Biểu đồ v2 cho thấy gì:** eval loss chạm đáy khoảng 0.36 ở epoch 2 rồi tăng lại khoảng 0.41 ở epoch 3, trong khi train loss vẫn giảm. AUC thì đi ngang.

**Vì sao v3 vẫn giữ nguyên:** sửa thành 2 epoch chỉ tiết kiệm thời gian chứ AUC gần như không đổi, trong khi train lại tốn 75 phút. Giữ nguyên config thì v3 **dùng lại kết quả v2** mà không phải train lại. TF-IDF cũng vậy.

### 2.4. Chống chạy quá 12 giờ

Nếu Kaggle chạy quá 12 giờ, notebook bị dừng và **có thể mất toàn bộ output**. v3 thêm một "đồng hồ":
- **Trước mỗi mô hình:** nếu thời gian đã chạy cộng thời gian ước tính của mô hình vượt 11.3 giờ thì bỏ qua mô hình đó.
- **Trước mỗi fold:** nếu không đủ thời gian cho thêm một fold nữa thì dừng mô hình đó lại.
- Các mô hình chạy theo **thứ tự ưu tiên**: Qwen trước (đóng góp nhiều nhất), rồi WildGuard, cuối cùng là DeBERTa-large. Nếu thiếu giờ thì chỉ mất mô hình ít quan trọng nhất.

Tính sơ bộ: Qwen khoảng 6.3 giờ, WildGuard khoảng 0.7 giờ, DeBERTa-large khoảng 4 giờ, tổng khoảng 11 giờ. **DeBERTa-large có thể bị bỏ qua nếu Qwen chạy chậm hơn dự tính.** Khi đó chạy thêm một lần nữa (gắn output v3 làm Input) thì chỉ phải train riêng DeBERTa-large.

### 2.5. Những thay đổi khác

| Thay đổi | Lý do |
|---|---|
| Thêm **accuracy, precision, recall, F1** vào REPORT, với ngưỡng chọn trên dev | để trả lời câu hỏi "đúng bao nhiêu %" bên cạnh AUC |
| Lỗi chỉ ghi 1 dòng trong REPORT (traceback đầy đủ vẫn có trong log) | REPORT v2 bị dán cả trăm dòng traceback |
| Output ghi vào `/kaggle/working/v3/` thay vì `clf/` | để không bị ghi đè kết quả cũ như lần trước |

## 3. Kỳ vọng

| Kịch bản | Ensemble AUC | Hạng tương đương |
|---|---|---|
| Qwen 2 epoch không giúp thêm, large bị bỏ qua | khoảng 0.991 | khoảng #28 |
| Qwen 2 epoch tăng nhẹ, gộp thêm Qwen v2 | 0.992–0.993 | #20–#25 |
| Thêm DeBERTa-large chạy được | 0.993–0.995 | #10–#20 |
| Thêm WildGuard (có token) | có thể ≥ 0.995 | gần top 10 (chỉ tính ở `ensemble_all`) |

Top 3 (≥ 0.99658) vẫn khó. Riêng việc khoảng tin cậy của holdout đã rộng ±0.005, nên dù v3 cho điểm sát mốc thì cũng chưa thể kết luận chắc là đạt top 3.

## 4. Cách chạy v3 trên Kaggle

1. **Import** `notebooks/04_jailbreak_classifier.ipynb` mới vào notebook cũ (File → Import Notebook).
2. **Gắn kết quả v2:** Add Input → *Your Work* → *Notebooks* → chọn chính notebook này (bản v2). Nếu không gắn được, tạo một Kaggle Dataset từ thư mục `results/clf/models/` rồi Add Input dataset đó.
3. (Tùy chọn) Thêm Secret `HF_TOKEN` để chạy WildGuard. Trước đó phải bấm đồng ý điều khoản tại https://huggingface.co/allenai/wildguard.
4. Chọn **GPU T4 x2**, bật **Internet**, rồi **Save Version → Save & Run All**.
5. Ở đầu log, kiểm tra có các dòng `♻️ Nạp tfidf …` và `♻️ Nạp deberta-v3-base …`. Có thì nghĩa là đã dùng lại kết quả v2.
6. Khi xong, tải thư mục **`v3/`** về và đặt vào **`results/v3/`**. **Đừng ghi đè lên `results/clf/`.**

## 5. Rủi ro

- **Code v3 chưa chạy thử.** Máy local không có GPU và thư viện ML, nên mình chỉ kiểm tra được cú pháp và build. Nếu lỗi, gửi log.
- **DeBERTa-large vẫn có thể hết bộ nhớ** dù đã giảm batch. Nếu vậy notebook ghi lỗi rồi chạy tiếp, không ảnh hưởng các mô hình khác.
- **Qwen 2 epoch có thể bắt đầu học thuộc lòng.** REPORT sẽ tự chẩn đoán nếu eval loss tăng lại sau khi chạm đáy.
