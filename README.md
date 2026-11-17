# Semantic Search — Research & Evaluation Lab

## Mục tiêu
Ứng dụng Streamlit có giao diện kiểm tra dữ liệu, gán nhãn relevance, chia truy vấn train/validation/test, chạy TF-IDF và 4 mô hình embedding, so sánh nDCG/Recall/MRR/Precision, xem kết quả từng câu và xuất CSV.

## Cài đặt
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Mở URL localhost do Streamlit hiển thị. Chạy TF-IDF trước để thử giao diện. Các model SentenceTransformer sẽ tải weights qua mạng khi chạy lần đầu; có thể cần GPU/RAM đáng kể cho E5-large/BGE-M3.

## Định dạng dữ liệu
- `places.csv`: place_id,name,city,category,description (mỗi place_id duy nhất)
- `queries.csv`: query_id,query,group (mỗi query_id duy nhất)
- `qrels.csv`: query_id,place_id,relevance (mỗi cặp duy nhất, 0–3)

Dữ liệu đi kèm là **dữ liệu minh họa, không phải bộ dữ liệu thực đã được kiểm chứng**. Ground truth còn thiếu rất nhiều nhãn; mặc định ứng dụng loại các query có kết quả Top-K chưa được gán nhãn, nên có thể chưa tính được metrics. Dùng tab Gán nhãn để hoàn thiện nhãn rồi tải CSV về, sau đó upload lại để dùng trong toàn bộ phiên. Chức năng 'Dùng nhãn vừa chỉnh sửa' chỉ giữ trong phiên Streamlit và có thể bị thay thế khi upload mới.

## Thiết kế thí nghiệm đề xuất
1. Chuẩn hóa corpus thực tế 1.000–3.000 địa điểm, khóa snapshot; loại bản ghi trùng, lỗi vị trí, mô tả thiếu.
2. Xây 150–300 truy vấn đa dạng (semantic, paraphrase, exact-name, location, constraints, hard-negative, no-answer, code-switching); gán nhãn theo query_id × place_id.
3. Chia truy vấn train 60%, validation 20%, test 20% có phân tầng theo nhóm. Train dùng phát triển nhãn/quy tắc, validation dùng chọn mô hình/siêu tham số, test chỉ báo cáo cuối.
4. Pool top 10–20 từ mọi mô hình, gán nhãn toàn bộ pool; thêm mẫu hard-negative và mẫu ngẫu nhiên. Nhờ 2 người chấm một phần, hòa giải bất đồng.
5. So sánh TF-IDF char n-gram, multilingual MiniLM, E5-base, E5-large, BGE-M3 Dense trên cùng corpus bằng exact cosine. E5 sử dụng prefix chuẩn.
6. Báo cáo nDCG@10 (primary), MRR@10, Precision@5, Recall@10 (nếu qrels đủ đầy), theo nhóm query; thêm P50/P95 online latency đo riêng với warmed-up model, RAM/VRAM.
7. Sau khi chọn dense model, làm ablation BGE-M3 sparse/hybrid, Qdrant ANN và reranker; không gộp vào thí nghiệm so sánh embedding đầu tiên.

## Giới hạn kỹ thuật
- Chạy exact cosine trong RAM, chưa tích hợp Qdrant hoặc API FastAPI, chưa đo end-to-end latency.
- Chế độ unjudged=0 chỉ để minh họa và có thiên lệch; chế độ nghiêm ngặt loại truy vấn có Top-K chưa gán nhãn, nhưng Recall vẫn có thể thiên lệch nếu chưa tìm đủ nhãn dương trong toàn corpus.
- Số liệu encode_seconds_total gồm tải/khởi tạo mô hình và toàn bộ corpus; không được gọi là latency của một truy vấn.
- Mỗi lần chạy không cố định seed mô hình hay kiểm tra khác biệt thống kê; nên chạy lại và báo cáo CI/bootstrap khi có test đủ lớn.
