# Kiểm tra triển khai — 17/11/2026

Đây là ghi nhận kiểm thử phần mềm, không phải kết quả đánh giá chất lượng trên dữ liệu nghiên cứu.

## Đã kiểm tra

- Bộ suite có bật `RUN_SERVICE_TESTS`, `RUN_IMAGE_SERVICE_TESTS`, `RUN_IMAGE_MODEL_TESTS`: 27 tests tại thời điểm chạy, 26 đạt; 1 test tải embedding **văn bản** để skip. Test pretrained ảnh chạy tuần tự bốn checkpoint: MobileNetV3-Large, DINOv2-small, CLIP ViT-B/32, SigLIP ViT-B/16. Tất cả đúng dimension, số hữu hạn và chuẩn L2; không dùng random weights hoặc mock model ở test này.
- Sau đó bổ sung test retry khi tạo collection thất bại: bộ image có 10 tests, 8 đạt, 2 opt-in skip trong lượt chạy local. Test retry mới đạt; sửa duy nhất pipeline tạo collection khi retry.
- Qdrant local engine thực: đủ năm handcrafted ablation, trọng số, index/re-index/delete, place dedup, hai mục tiêu benchmark, Visual image target và nhãn thiếu.
- Qdrant server 1.15.5: index ảnh thật từ fixture → search → benchmark → export JSON/CSV. Collections test có namespace riêng và được dọn sau test; dữ liệu/ảnh fixture nằm trong thư mục tạm.
- Hồi quy Elasticsearch BM25 và Qdrant text service tests đạt. API text registry vẫn trả M1–M4, dữ liệu và index text không bị thay thế.
- Frontend `npm.cmd run build` đạt. JavaScript của trang ảnh vượt `node --check`. `/image-search` và `/api/ai/images/models` trả HTTP 200; registry có 10 encoder. Runtime Image Assets còn rỗng; fixture không được đưa lên dashboard nghiên cứu.

## Môi trường quan sát

Python 3.14, torch 2.13.0+cpu, torchvision 0.28.0+cpu, transformers 5.14.1, OpenCV 4.13.0, scikit-image 0.26.0. CUDA không khả dụng trên môi trường kiểm thử này.

Qdrant client đang cài là 1.20.0, server compose là 1.15.5: thư viện phát cảnh báo lệch version dù các test trên đều đạt. `requirements-image.txt` giới hạn client 1.15–1.16 cho lần cài venv mới, phù hợp server được cung cấp. Không tắt cảnh báo để che vấn đề tương thích.

CLIP/SigLIP full checkpoint có các key text không sử dụng: loader vision báo unexpected text keys, không có missing vision keys. Đây là hệ quả có chủ đích của việc chỉ khởi tạo image encoder. Lỗi thiếu/mismatch vision weights bị chặn.

Hugging Face cache mặc định từng lỗi symlink Windows khi tải SigLIP. Đã sửa sang snapshot `local_dir` trong workspace; SigLIP và lượt kiểm thử lại bốn checkpoint đều đạt. Không thay đổi quyền hay thiết lập bảo mật Windows.

## Chưa xác minh

- Weights thật của ResNet-50, EfficientNet-B0, DINOv2-base/large và CLIP ViT-B/16 chưa được tải/chạy trong lượt này. Adapter và registry có triển khai; test opt-in nhận toàn bộ model key để chạy tiếp.
- Chưa kiểm thử CUDA, dataset quy mô lớn hay tải nhiều người dùng. Collapse Top-K places hiện lấy toàn bộ image hits, cần tối ưu thêm nếu Gallery lớn.
- Giao diện đã kiểm tra build, syntax và HTTP, chưa xác minh trực quan bằng browser trong lượt này.
- Chưa có Gallery/Query/Ground Truth chính thức: không thể kết luận model nào tốt nhất, cải thiện bao nhiêu, hoặc xuất bảng metric nghiên cứu có ý nghĩa. Những metric từ test fixture chỉ dùng để xác minh công thức/pipeline và không được lưu như run chính thức.
