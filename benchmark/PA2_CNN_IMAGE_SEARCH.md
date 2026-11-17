# PA2 — Truy xuất ảnh bằng CNN pretrained

Mở http://127.0.0.1:8000/image-search → **Dữ liệu ảnh** → **PA2 · Truy xuất ảnh bằng CNN**.

## Mô hình

| Model | Checkpoint cố định | Tham số toàn checkpoint | Vector |
|---|---|---:|---:|
| MobileNetV3-Large | Torchvision IMAGENET1K_V2 | 5.483.032 | 960 |
| ResNet-50 | Torchvision IMAGENET1K_V2 | 25.557.032 | 2048 |
| EfficientNet-B0 | Torchvision IMAGENET1K_V1 | 5.288.548 | 1280 |

Đây là số tham số của mạng phân loại pretrained gốc. Sau khi bỏ classifier, số tham số backbone thực được lưu riêng tại `model.feature_parameters`. Không dùng logits phân loại làm vector.

ResNet thay `fc` bằng Identity; MobileNet/EfficientNet thay toàn bộ `classifier` bằng Identity. Forward trả đặc trưng sau Global Average Pooling/flatten. Dùng đúng `weights.transforms()` của checkpoint cho resize/crop/normalize; không áp dụng resize 128×128 của PA1 lên CNN. Giữ ảnh RGB từ asset gốc/canonical PNG. Vector đầu ra float32, L2; Qdrant dùng Cosine.

Tất cả model `.eval()`, `requires_grad_(False)`, `torch.inference_mode()`; không train/fine-tune, không thay bằng trọng số ngẫu nhiên nếu tải lỗi. CPU mặc định tối đa 4 thread (`IMAGE_CPU_THREADS`), batch 2. Chỉ nạp một model tại một thời điểm và unload sau mỗi model/job. Không đo VRAM.

## Quy trình

1. Chọn một index PA1 có ảnh thật làm Gallery chuẩn.
2. Bấm **Mã hóa Gallery bằng cả 3 CNN**. Backend chốt danh sách tối đa **2 ảnh/địa điểm** ngay khi gửi yêu cầu, tạo baseline PA1 riêng bằng cách sao chép vector thực, rồi chạy từng CNN.
3. Mỗi CNN lưu collection riêng. Metadata/model/transforms/revision URL/dimension/count/timing/RAM lưu PostgreSQL. Gallery hash và image_ids giống nhau giữa bốn index; không tự thêm ảnh nhập sau đó vào bộ so sánh đã chốt.
4. Trong **Tìm kiếm ảnh**, chọn MobileNetV3-Large/ResNet-50/EfficientNet-B0, tải ảnh Query lên, chọn Top-K và tìm kiếm. Query dùng chính preprocessing và model của index. Kết quả hiển thị ảnh/địa điểm/cosine, load/preprocess/embedding/search/total latency và RAM.
5. Trong **Vector đặc trưng**, chọn model và ảnh để mã hóa hoặc đọc vector từ index. Xem số chiều thực, từng phần tử, biểu đồ, norm, min/max/mean/std, pooling và checkpoint.
6. Trong **Đánh giá truy vấn test**, tạo manifest từ một trong bốn index cùng Gallery; upload Query độc lập, gán nhãn thị giác 0–3 và xác nhận. Chọn cả PA1 snapshot và ba CNN để chạy cùng tập test. Có thể chọn một truy vấn hoặc tất cả.

Visual dùng Recall@10/MRR@10/nDCG@10/Precision@10; Exact Place là mục tiêu riêng. Chưa có nhãn thì chất lượng là null, không được suy ra từ source place_id hoặc kết quả xếp hạng. Run lưu lịch sử, per-query/repeat/Top-K, cold/warm timing, RAM và provenance. Không kết luận CNN tốt hơn PA1 trước khi có nhãn xác nhận.

## API và file

`POST /api/ai/images/cnn-indexes`:

```json
{"source_index_id":"image_index_PA1_ready","batch_size":2}
```

Trả job; poll `/api/jobs/{id}`. Result có baseline_index_id, gallery_images, gallery_sha256 và danh sách ba CNN index/dimension/count. Hủy dừng tại ranh giới batch/model; bước tải model hiện tại phải kết thúc trước.

Phần điều phối: `backend/images/cnn.py`; adapter: `encoders.py`; API: `api.py`; giao diện: `frontend/src/CNNPanel.jsx` và ImageLab. Không thay thế pipeline PA1 hoặc BM25/Dense văn bản.

## Lần triển khai hiện tại

Đã chốt 939 ảnh nguồn thực từ PA1 (tối đa 2 ảnh/địa điểm). Job và bằng chứng máy nằm trong runtime/cnn-preparation-job.json và runtime/cnn-real-validation.json sau khi kiểm tra xong. Thư mục runtime không commit.

Đã hoàn tất cả ba CNN, mỗi index đúng 939 vector, cùng Gallery hash với baseline PA1. Kiểm tra pretrained frozen, dimension 960/2048/1280, L2 norm và upload tìm Top-10 thực đều đạt. Đã chạy 12 mẫu đo (4 phương án × 3 lần lặp) trên một ảnh Query nguồn thật, chưa gán nhãn. Run `c56449ab870c40b1a579da7eb89875e2`, bộ test `6841e3416fcfcb924b0b3c58`. Quality vẫn null; Official đã được kiểm tra chặn khi thiếu nhãn. Không dùng run này để kết luận chất lượng CNN.

Build frontend đạt; bộ phần mềm có 55 tests, 50 đạt / 5 opt-in bỏ qua. Lượt thực tế trên không dùng mock/random weights hoặc fixture làm kết quả nghiên cứu.

Nguồn chính thức: [MobileNetV3-Large](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html), [ResNet-50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html), [EfficientNet-B0](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html).
