# PA2 — Dense Semantic Search trên CPU

Mở http://127.0.0.1:8000, chọn **PA2 · Dense Semantic Search**. Dùng lại PostgreSQL chứa 6.052 địa điểm, ảnh và 100 truy vấn; vector lưu trong Qdrant server. Không thay đổi dữ liệu gốc hoặc index BM25.

## Quy trình

1. Chọn dataset và mô hình MiniLM / E5-base / E5-large / BGE-M3.
2. Chọn cùng trường văn bản cho các phương án so sánh. `place_id` là khóa truy xuất bản ghi gốc, không đưa vào embedding.
3. Máy CPU: bắt đầu batch 2. Chọn giới hạn token; BGE-M3 trên trang PA2 mặc định 512 để tiết kiệm RAM, giới hạn checkpoint là 8192. Giới hạn thực tế lưu trong cấu hình. Nội dung vượt giới hạn bị cắt bởi tokenizer, chưa chia chunk.
4. Bấm mã hóa. Tiến trình hiển thị số địa điểm, log và nút hủy sau batch. Checkpoint tải lần đầu vào cache Hugging Face; các định dạng ONNX/OpenVINO và trọng số phụ không được tải. Chỉ một model thường trú.
5. Mỗi index tạo ba collection Cosine / Dot / Euclid từ cùng vector, kiểm tra đủ số điểm trước khi báo ready. Revision được cố định theo commit. Vector cũng lưu thành artifact `.npy`.
6. Chọn **Nhập truy vấn & xem ảnh kết quả** để kiểm chứng định tính, hoặc **Benchmark Runner** để chạy cùng tập truy vấn, Ground Truth và K.

## Đo lường

- Search hiển thị thời gian nạp model, embedding truy vấn, vector search, tổng phản hồi và RAM tiến trình Python. RAM không bao gồm container Qdrant/PostgreSQL.
- Benchmark lưu từng lần chạy vào PostgreSQL, có cold start riêng, warmup, số lần lặp và latency P50/P95/P99. Không hiển thị VRAM trên máy CPU.
- Recall/MRR/nDCG chỉ dùng nhãn confirmed. Thiếu nhãn trả về chưa đủ dữ liệu theo điều kiện từng metric, không thay bằng 0.
- Giữ cùng dataset, thứ tự trường, split và chính sách token khi so sánh; sử dụng exact vector search để loại sai số ANN.
- Khi L2 bật, Dot/Cosine/Euclid trên vector đơn vị có thứ hạng tương đương về toán học (trừ sai số số học và hòa điểm). Tắt L2 để khảo sát ảnh hưởng độ lớn; Cosine của Qdrant vẫn chuẩn hóa vector riêng.

## Checkpoint chính thức

| Model | Chiều | Token tối đa | Quy tắc |
|---|---:|---:|---|
| [MiniLM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | 384 | 128 | Mean pooling |
| [E5-base](https://huggingface.co/intfloat/multilingual-e5-base) | 768 | 512 | query: / passage:, masked mean |
| [E5-large](https://huggingface.co/intfloat/multilingual-e5-large) | 1024 | 512 | query: / passage:, masked mean |
| [BGE-M3](https://huggingface.co/BAAI/bge-m3) | 1024 | 8192 | Chỉ Dense, CLS, không thêm prefix |

BGE-M3 hỗ trợ Sparse/Multi-vector trong checkpoint gốc; PA2 này chỉ khảo sát Dense theo yêu cầu. Mô hình lớn có thể cần giải phóng thêm RAM và chạy lâu trên CPU; trạng thái ready chỉ xuất hiện khi mã hóa và ghi Qdrant thành công.
