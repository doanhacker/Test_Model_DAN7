# Tính Recall@K, MRR@K và nDCG@K từng truy vấn

Menu **Tính Recall / MRR / nDCG** được thêm vào giao diện hiện có. Không cần tạo bộ thực nghiệm mới.

1. Chọn dataset, query set và Ground Truth đúng phiên bản.
2. Chọn một câu hỏi và phương án đã có index. Chọn K (mặc định 10) và ngưỡng relevance (1–3).
3. Bấm **Lấy Top-20 để gán nhãn**. Nếu K lớn hơn 20, hệ thống lấy ít nhất K kết quả.
4. Gán relevance 0–3 và lý do cho các ứng viên. Để trống là chưa biết. Có thể tìm thêm địa điểm ngoài pool và gán nhãn cho chúng.
5. Bấm **Xác nhận nhãn & tính Recall / MRR / nDCG**. Chỉ các giá trị do người dùng chọn trong phiên này mới được xác nhận; không tự gán nhãn 0 cho các dòng còn lại. Snapshot nhãn và kết quả tính được lưu riêng trong PostgreSQL.

## Công thức

- Recall@K = số địa điểm có relevance >= ngưỡng trong Top-K / tổng số địa điểm liên quan trong tập nhãn đánh giá.
- RR@K = 1/rank của kết quả liên quan đầu tiên; không có kết quả liên quan trong Top-K đã được gán nhãn thì bằng 0. MRR@K là trung bình RR@K trên các truy vấn hợp lệ; với một truy vấn, hai giá trị bằng nhau.
- DCG@K = tổng (2^relevance − 1)/log2(rank + 1).
- IDCG@K là DCG với thứ tự lý tưởng theo relevance giảm dần. nDCG@K = DCG/IDCG; IDCG=0 thì quy ước nDCG=0.

nDCG dùng mức relevance gốc 0–3, không nhị phân hóa bằng ngưỡng Recall/MRR. Precision@K vẫn được lưu để tương thích bộ benchmark hiện có.

## Nhãn thiếu và phạm vi chỉ số

Không có nhãn xác nhận thì cả ba giá trị để null. Một nhãn proposed/pending không được dùng như nhãn thật.

**MRR chính xác** không cần gán nhãn toàn corpus: nếu đã biết kết quả liên quan đầu tiên và tất cả vị trí đứng trước nó, RR đã xác định. Các nhãn thiếu phía sau không làm thay đổi vị trí đúng đầu tiên. Nếu chưa xác định được tiền tố đó, MRR để trống. Nếu tất cả kết quả Top-K đã gán nhãn và đều không liên quan, RR=0.

**Recall/nDCG chính thức** cần biết tổng tập liên quan / thứ tự lý tưởng đáng tin cậy. Cờ “Chỉ tính Recall/nDCG khi đủ nhãn toàn corpus” giữ hai giá trị null cho tới khi corpus có đủ nhãn xác nhận; MRR vẫn có thể hiện nếu đã đủ bằng chứng.

**Recall/nDCG theo pool**: khi bỏ cờ trên, hệ thống yêu cầu duyệt hết pool kết quả đã lấy (mặc định Top-20), rồi tính theo các nhãn xác nhận hiện có, gồm cả địa điểm bổ sung ngoài pool. UI ghi rõ “Tạm tính theo nhãn đã xác nhận”. Không tính Recall/nDCG chỉ từ Top-10 khi còn ứng viên Top-20 chưa duyệt, để tránh Recall=1 do mẫu số bị giới hạn vào chính Top-K. Pool BM25 vẫn thiên lệch theo BM25 và chưa chứng minh đã tìm hết mọi địa điểm liên quan trong 6.052 địa điểm.

Không có nhãn dương: Recall không xác định và giữ null. Nhãn thiếu không bị chuyển thành 0. K đánh giá không được vượt K của phiên tìm kiếm đã lưu. Query text và dataset phải trùng phiên tìm kiếm gốc.

## Lưu và tái lập

`POST /api/evaluation/query` nhận `job_id,query_id,qrels_id,k,threshold,official`. Kết quả có từng thành phần: hits_at_k, known_relevant, first_relevant_rank, dcg, idcg, nhãn thiếu Top-K/pool/corpus và bảng đóng góp DCG từng rank. Trạng thái từng metric (`exact`, `pooled`, `needs_labels`…) được lưu trong payload.

Kết quả tạo run ID mới trong các bảng benchmark_runs/run_metrics/query_metrics/search_results và snapshot đầy đủ. Timing/RAM lấy từ phiên tìm kiếm nguồn, không giả vờ là phép đo mới. “Chạy lại cấu hình” của run đánh giá chỉ đánh giá lại đúng truy vấn nguồn, không tự chạy cả split. Xem/xuất kết quả qua màn hình Kết quả & biểu đồ hoặc lịch sử database hiện có.

Batch Benchmark Runner vẫn giữ chế độ chính thức yêu cầu nhãn toàn corpus. Sau khi có nhiều query đã duyệt, có thể chạy thăm dò để tổng hợp trung bình trên các query hợp lệ; luôn báo số mẫu/nhãn đủ điều kiện cho từng metric.

Kiểm thử công thức, MRR tiền tố, nhãn thiếu, pool chưa đủ, bảo vệ phiên bản/query/K, lưu PostgreSQL và chạy lại: `python -m unittest discover -s tests -v`. Dữ liệu kiểm thử công thức chỉ nằm trong schema tạm; không chèn nhãn test vào dữ liệu nghiên cứu.
