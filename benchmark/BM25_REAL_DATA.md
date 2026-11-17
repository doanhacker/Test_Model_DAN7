# BM25 với Excel địa điểm thực tế

Hiện dữ liệu và lịch sử được lưu trong PostgreSQL riêng tại PostgreSQL server, kết hợp Elasticsearch experiment index; không sửa Excel hoặc index khác. JSON cũ đã được nhập một lần và giữ làm bản legacy. Xem [hướng dẫn database](DATABASE.md).

Nguồn: `../data/danang_merged_serpapi_da_lam_sach.xlsx`, sheet `Địa điểm chung`. SHA256 `b318647e55536a58b02fd49ac6470fe9fba810aa2eb2da1cc0bfe9bc813b6bd3`.

- 6.312 dòng nguồn: Đà Nẵng 6.053, Hội An 245, Huế 12, Nha Trang 2.
- Scope thí nghiệm: chỉ các dòng có thành phố Đà Nẵng. Không sửa hay xóa 259 dòng thành phố khác trong nguồn.
- ID `62796186-d6d8-49e7-94cb-d41b99bdf5f0` xuất hiện hai lần (Gia Lợi Hotel), cùng tên/địa chỉ nhưng khác mô tả/tiện ích. Snapshot gộp các giá trị khác nhau, lưu cả hai dòng nguồn; không tự gộp ID khác tên/địa chỉ.
- 6.052 địa điểm duy nhất, không có dòng thiếu ID/tên trong scope. ES `_count` xác nhận 6.052 tài liệu.
- Amenities dạng Python dict trong Excel được parse bằng `ast.literal_eval` và làm phẳng; không thực thi nội dung. Tất cả ID gốc được giữ nguyên.
- `place_id` là ID tài liệu Elasticsearch, không phải trường text tính điểm. BM25 tìm trên `name,category,description,address,tags,amenities`. Category giữ cả loại địa điểm và danh mục nguồn. k1=1.2, b=.75; analyzer standard/lowercase giữ dấu; dùng cùng analyzer cho index/query.

## Dùng giao diện hiện có

Mở http://127.0.0.1:8000. Bộ dữ liệu, 100 truy vấn, pool và run đo hiệu năng được chọn sẵn khi tải trang (lấy từ preparation report).

1. **Quản lý dữ liệu → Địa điểm**: chọn phiên bản để xem trước; số hợp lệ/đã index hiển thị cạnh phần import. Với file mới, chọn file, xem preview rồi bấm **Nhập Excel du lịch · Đà Nẵng · xử lý ID trùng**. Không dùng import chuẩn nếu chưa xử lý duplicate ID.
2. **Tập truy vấn**: chọn phiên bản `100 truy vấn Đà Nẵng · nháp cần duyệt`. Mỗi nhóm có 20 câu: tên riêng, từ khóa trực tiếp, mô tả đặc điểm, ngữ nghĩa, nhiều điều kiện. Nhãn split 60/20/20 phân bố đều trong từng nhóm; run kiểm tra hiện dùng `split=all` trên 100 câu. Bộ này tái sử dụng cho embedding, không phải query set riêng cho BM25.
3. **Ground Truth**: bấm mở bộ gán nhãn. Danh sách mặc định là pool BM25 Top20 cho query đã chọn. Tìm theo tên/địa chỉ/mô tả để bổ sung địa điểm ngoài pool; xem tối đa 200 kết quả lọc mỗi lần. Chọn 0–3, ghi lý do, đánh dấu **Tôi xác nhận nhãn này**, rồi lưu phiên bản Ground Truth mới. Đổi relevance/lý do sẽ đưa nhãn về pending để xác nhận lại. Không tự gán 0 cho phần còn lại.
4. **Benchmark Runner**: dùng BM25 index đã tạo, query/GT version khớp, Top-K=10. Chọn split all nếu muốn chạy đủ 100 câu. Chế độ thăm dò chỉ sử dụng nhãn confirmed; chính thức kiểm tra đủ nhãn toàn corpus. Kết quả hiển thị số mẫu hợp lệ và null khi thiếu nhãn, không lấp bằng 0.
5. **Xuất báo cáo → Excel**: Summary, Per query, Ground Truth (có reason/status), Queries, Source mapping, Rankings và Provenance. Metadata JSON dài được chia nhiều dòng để tránh giới hạn ô Excel. CSV/JSON vẫn có sẵn.

## Ground Truth và giới hạn suy luận

Schema mở rộng: `query_id,place_id,relevance,reason,review_status`. Trạng thái: `pending` chưa gán/duyệt, `proposed` đề xuất, `confirmed` đã được người dùng xác nhận. Import nhãn không có review_status mặc định pending; chỉ confirmed mới vào metric. Dữ liệu cũ đã lưu trước phiên bản này vẫn được đọc tương thích theo trạng thái cũ.

Pool hiện có 1.998 cặp: 99 query trả 20 kết quả khớp, `ten_rieng-08` trả 18. Không tự thêm hai địa điểm không khớp chỉ để làm tròn 2.000. Pool không có relevance tự động hoặc đề xuất từ điểm BM25. Query evidence lưu anchor/source term để kiểm tra việc sinh câu; anchor không phải nhãn dương.

Hiện chưa có nhãn nào được người dùng xác nhận. Run thực đã đo latency/RAM nhưng Recall@10, MRR@10, nDCG@10 đều null. Không thể xác minh luồng tính metric **chính thức trên dữ liệu thực** trước khi người dùng gán/xác nhận đủ nhãn. Các công thức và cổng xác nhận đã được kiểm thử tự động, không đánh tráo bằng nhãn test vào dữ liệu thực.

Top20 BM25 có thiên lệch theo BM25; không chứng minh đã bao phủ tất cả địa điểm liên quan. Chỉ duyệt 1.998 ứng viên chưa đủ cho Recall chính thức toàn corpus. Hiện kiểm tra đầy đủ yêu cầu 6.052×100 = 605.200 cặp xác nhận cho cả 100 query. Có thể bắt đầu xem kết quả thăm dò sau khi duyệt đủ Top-K, nhưng Recall/nDCG dựa trên tập nhãn hiện có phải được báo cáo là thăm dò. Khi thêm embedding, nên mở rộng pool bằng kết quả các model và tiếp tục rà soát độc lập, không tự coi ngoài pool là sai.

Truy vấn được tạo từ đặc điểm có trong file, dùng các mẫu ngôn ngữ cố định và paraphrase tiếng Việt; không gọi LLM ngoài hệ thống. Cần người dùng xem lại ý định truy vấn, đặc biệt dữ liệu nguồn có các tag/description chưa được xác minh đầy đủ. Không khẳng định nguồn là ground truth ngoài thực địa.

## API bổ sung

- `POST /api/real-data/import-excel`: multipart file, city mặc định Đà Nẵng.
- `POST /api/real-data/queries`: dataset_id → 100 query có kiểm tra corpus.
- `POST /api/real-data/pool`: index_id, queries_id → job Top20.
- `GET /api/real-data/pool?dataset_id=...&queries_id=...`: ứng viên/rank/score/timing.
- `GET /api/real-data/coverage?dataset_id=...&queries_id=...&qrels_id=...`: confirmed/pending/missing_corpus.
- `GET /api/real-data/source/{dataset_id}`: filename, SHA256, sheet và các dòng gốc cho từng place_id.
- `GET /api/real-data/query-evidence/{queries_id}`: bằng chứng nguồn của câu hỏi.
- `GET /api/real-data/preparation`: các ID dataset/index/query/pool/run đang chuẩn bị.

Chạy lại chuẩn bị bằng `python -X utf8 -m backend.prepare_real_bm25` trong thư mục benchmark. Giữ nguyên nguồn; snapshot dữ liệu/truy vấn có hash ổn định, tái sử dụng index cùng cấu hình nếu sẵn sàng. Script có tạo pool pending/run hiệu năng mới, không xác nhận nhãn thay người dùng. Không chạy lại script sau khi đã bắt đầu gán nhãn nếu không muốn tạo thêm bản pool nháp.

File sửa: backend/data.py (schema/review), retrieval.py (đếm index), runner.py (lọc confirmed/coverage), main.py (API/Excel), frontend/src/main.jsx (bổ sung vào màn hình hiện có), templates/qrels.csv; thêm real_data.py, prepare_real_bm25.py và tests/test_real_data.py; cập nhật test_api.py/test_retrieval.py tương ứng schema/count.
