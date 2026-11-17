# PA1 Image Search Lab — đặc trưng truyền thống

Mở **http://127.0.0.1:8000/image-search**. Nhấn Ctrl+F5 sau khi cập nhật giao diện.

## Thuật toán và cấu hình

Ảnh được xử lý EXIF, alpha trên nền trắng, RGB, resize bicubic 128×128. Không huấn luyện hoặc fine-tune model.

| Cấu hình ablation | Nhóm đặc trưng | Chiều |
|---|---|---:|
| 1 | HSV | 128 |
| 2 | HSV + LBP | 138 |
| 3 | HSV + LBP + GLCM | 186 |
| 4 | HSV + LBP + GLCM + HOG | 1950 |
| 5 | HSV + LBP + GLCM + HOG + Canny | 2014 |

Mỗi nhóm chuẩn hóa L2, nhân trọng số không âm, ghép theo thứ tự cấu hình rồi chuẩn hóa L2 lần cuối. Trọng số mặc định đều bằng 1. Nhóm rỗng có block 0; vector toàn 0 bị từ chối. Không chọn trọng số dựa trên test labels. Điểm cosine của vector ghép bằng tổng tích vô hướng của từng block; giao diện hiển thị đóng góp và cosine riêng của nhóm. Trọng số ảnh hưởng đóng góp theo bình phương, không phải tuyến tính.

Qdrant lưu collection riêng theo cấu hình, namespace và revision; mỗi điểm gắn image_id / place_id. Chỉ báo ready khi số điểm vật lý khớp Gallery. Replay không tạo bản sao; nếu collection mất hoặc thiếu điểm, có thể tạo lại từ các asset bất biến. Xóa vector không xóa ảnh gốc và run đã lưu.

## Bốn màn hình

### Dữ liệu ảnh

Duyệt database địa điểm, chọn ảnh đại diện hoặc upload ảnh gắn place_id. Nút import nguồn dùng hàng đợi trên máy chủ: chuyển màn hình / đóng tab không dừng công việc. Có giới hạn số ảnh, tùy chọn cả album, hủy sau ảnh hiện tại, báo cáo ảnh mới / tái sử dụng / lỗi và liên kết nguồn. Chạy lại tái sử dụng ảnh đã import. Lỗi không được thay bằng ảnh giả.

Ảnh nhập từ database hiện là cache tối đa 960×640. File upload giữ kích thước decode gốc và lưu canonical PNG. Metadata / nhãn / run lưu PostgreSQL; PNG nằm trong runtime/image_blobs. Khi sao lưu cần giữ cả PostgreSQL và thư mục ảnh.

Chọn cùng Gallery và bấm **Tạo đủ 5 cấu hình PA1 cùng Gallery** để chuẩn bị ablation; có thể tạo riêng từng cấu hình và chỉnh trọng số. Để so sánh công bằng phải dùng cùng Gallery / Query / nhãn / protocol. Không thêm ảnh vào collection đang dùng cho manifest cũ; backend chặn benchmark nếu Gallery không còn khớp.

### Tìm kiếm ảnh

Upload ảnh hoặc chọn Query đã lưu, chọn cấu hình PA1 và Top-K. Mặc định exact cosine, gộp nhiều ảnh theo place_id bằng điểm lớn nhất. Kết quả có ảnh, tên, địa chỉ / mô tả nếu liên kết database, score và giải thích đóng góp các nhóm. Tổng latency gồm decode, load, embedding, Qdrant, giải thích và metadata; thời gian search riêng chỉ đo truy vấn Qdrant ban đầu. Hình ảnh tải trên trình duyệt không nằm trong latency backend.

### Vector đặc trưng

Mã hóa ảnh bằng cấu hình đã chọn hoặc đọc đúng vector Qdrant. Có bảng từng chiều, biểu đồ 128 chiều/vùng, dimension, norm, min/max/mean/std, block offsets, trọng số và tỷ lệ năng lượng. Hiển thị ảnh RGB/xám/LBP/GLCM/HOG/Canny thực từ pipeline. GLCM là heatmap trung bình 8 ma trận; các ảnh minh họa không thay thế vector dùng tìm kiếm.

### Đánh giá truy vấn test

Upload ảnh Query độc lập với Gallery, gán nhãn 0–3 cùng lý do, xác nhận rồi lưu phiên bản mới. Exact Place và Visual Similarity là hai mục tiêu khác nhau. Không tự suy ra nhãn thị giác từ place_id; không biến nhãn trống thành 0. Chặn Query trùng pixel với Gallery; người dùng vẫn cần kiểm tra ảnh gần trùng / crop / recompress để tránh leakage.

Có thể chọn một truy vấn hoặc cả bộ test và so sánh năm cấu hình. Visual: Recall@10, MRR@10, nDCG@10, Precision@10. Exact: Recall@1/5, MRR@10. Nhãn dương theo threshold 1; nDCG dùng gain 2^relevance−1. Official Visual yêu cầu đủ nhãn trên toàn Gallery target. Thăm dò có trạng thái / độ phủ và số nhãn thiếu; metric không xác định vẫn là null. Khi không có nhãn liên quan, Recall không xác định.

Kết quả có biểu đồ chất lượng / latency, từng query / repeat, trace đóng góp DCG, thứ hạng đầu tiên đúng, Top-K bằng ảnh, lịch sử PostgreSQL và export JSON / CSV. Cold start và warm-up tách khỏi warm latency; RAM là RSS tiến trình Python, không gồm PostgreSQL/Qdrant. Không sử dụng VRAM trên giao diện CPU.

## Dữ liệu thực đã kiểm tra

Đã chạy năm cấu hình trên **cùng 16 ảnh Gallery nguồn thực**, một ảnh Query nguồn khác pixel, **3 lần lặp mỗi cấu hình** (15 mẫu đo). Đã kiểm tra vector 128/138/186/1950/2014 chiều, chuẩn L2, sáu ảnh trung gian, replay import và chặn Official khi thiếu nhãn.

Tập test `afed4a07925cb8991b06a2bb`; run hiệu năng `af5d07c97ee24c058c7452986543a7ce`. Đây là **run thăm dò chưa có nhãn**, không phải kết quả chất lượng chính thức. Recall/MRR/nDCG đang null; cần người dùng kiểm tra ảnh Query và gán nhãn 16 địa điểm trước khi tính chính thức. Ảnh khác pixel không bảo đảm loại hết near-duplicate.

Bằng chứng máy tại runtime/pa1-real-validation.json (không commit dữ liệu runtime). Bộ kiểm thử phần mềm dùng schema PostgreSQL và thư mục tạm; fixture không đưa vào Gallery nghiên cứu.

## Khởi chạy lại

```powershell
# Từ benchmark, cài môi trường theo requirements-image.txt nếu chưa có.
python -m pip install -r requirements-image.txt
Set-Location frontend
npm.cmd ci
npm.cmd run build
Set-Location ..
.\start.ps1
```

API mới: `POST /api/ai/images/ablation`, `POST /import-source`, `GET /imports`, `GET /assets/{id}/pa1-preview?kind=...` (các đường dẫn rút gọn đều có prefix `/api/ai/images`). Benchmark nhận thêm query_ids để chọn phạm vi test. Judgment hỗ trợ reason và review_status=confirmed.
