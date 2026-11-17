# PostgreSQL cho dữ liệu và lịch sử thực nghiệm

Backend dùng **PostgreSQL 16** qua psycopg, service `postgres` trong Docker Compose. Không dùng SQLite để đọc/ghi dữ liệu ứng dụng. File cũ chỉ giữ làm bản migration dự phòng.

## Chạy và kết nối

```powershell
python -m pip install -r requirements.txt
docker compose up -d postgres elasticsearch qdrant
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

`.env` chứa POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_PORT và DATABASE_URL. Mật khẩu local được tạo riêng, không hardcode hoặc trả qua API. `.env.example` cần thay mật khẩu mẫu. Host `127.0.0.1`, port mặc định `5432`, database/user mặc định `benchmark`; password xem `.env`. Dùng các giá trị đó để kết nối DataGrip/pgAdmin. Dữ liệu nằm trong Docker volume `postgres-data`.

## Migration và lưu trữ

Khởi động lần đầu tạo schema và nhập toàn bộ document từ SQLite cũ trong một transaction, tái tạo các bảng quan hệ rồi đánh dấu migration. Ưu tiên SQLite cũ vì chứa run/nhận xét mới hơn JSON. Nếu không có file cũ, nhập JSON legacy. Restart không nhập lại và không fallback SQLite khi PostgreSQL lỗi.

Đã chuyển và đối chiếu 6.052 địa điểm, 100 truy vấn, 1.998 cặp Ground Truth, 6.053 liên kết dòng Excel, cấu hình index, lịch sử chạy và jobs. File nguồn và SQLite cũ giữ nguyên. Chưa đủ nhãn thì metric chất lượng vẫn là SQL NULL.

Bảng: dataset_versions, places, queries, ground_truth, source_links, search_indexes, benchmark_runs, run_metrics, query_metrics, search_results, documents, schema_meta. Views: latest_places, benchmark_history. Payload tái lập và bảng quan hệ ghi trong cùng transaction; khóa ngoại/unique kiểm tra tính nhất quán và rollback khi lỗi.

Ảnh địa điểm nằm trong bảng `place_images`: liên kết `dataset_id,place_id`, URL nguồn, ảnh đại diện, thứ tự gallery và file/sheet/dòng Excel. Dữ liệu hiện có 7.965 liên kết cho 5.878 địa điểm; 174 địa điểm không có ảnh nguồn. Khi mở ảnh lần đầu, backend tải URL đã kiểm tra, tạo JPEG xem trước tối đa 960×640 và lưu **nội dung ảnh BYTEA**, MIME và SHA256 ngay trong PostgreSQL. Các lần sau đọc cache database. Không tải toàn bộ ảnh gốc trước; URL nguồn vẫn được giữ để truy vết. URL hỏng không được thay bằng ảnh khác địa điểm.

API `/api/place-images?dataset_id=...&place_id=...` trả gallery; `/api/place-images/{image_id}` trả ảnh; `/api/place-images/stats?dataset_id=...` trả số URL/địa điểm/ảnh cached/ảnh lỗi. Chỉ HTTPS và các host ảnh trong Excel được cho phép; không nhận một URL tùy ý từ query string để tải. Ảnh quá lớn/không giải mã được trả 404, lỗi được lưu và có thời gian chờ retry 5 phút. Bản backup PostgreSQL bao gồm cả các byte ảnh đã cache.

`backend/storage_schema.py` chứa schema/projection, `backend/database.py` quản lý psycopg và migration. PostgreSQL lưu địa điểm/nhãn/cấu hình/kết quả; Elasticsearch vẫn xử lý BM25, Qdrant vẫn xử lý vector.

## Giao diện và API

Tổng quan → **Cơ sở dữ liệu địa điểm & lịch sử thực nghiệm** hiển thị PostgreSQL, lọc lịch sử, xem kết quả, chạy lại cấu hình với run ID mới và tải backup `.sql`. Kiểm chứng định tính lưu nhận xét riêng trong PostgreSQL, không biến thành Ground Truth.

- GET `/api/database/info`
- GET `/api/database/places?dataset_id=...&page=1&page_size=50&search=...`
- GET `/api/database/history?method=BM25&page=1&page_size=50`
- POST `/api/database/runs/{run_id}/rerun`
- GET `/api/database/backup` → SQL schema + COPY data từ snapshot repeatable-read.

Phục hồi backup bằng psql vào database mới/rỗng. Không sửa riêng một bảng mà bỏ qua documents; dùng API để giữ snapshot nhất quán. `docker compose stop` giữ volume; không xóa volume khi chỉ muốn dừng chương trình.

```sql
SELECT place_id, name, address FROM latest_places;
SELECT run_id, status, method, recall_at_k, mrr_at_k, ndcg_at_k,
       average_ms, p95_ms, ram_mb
FROM benchmark_history ORDER BY created_at DESC;
```

## Kiểm thử

`python -m unittest discover -s tests -v` cần PostgreSQL chạy và .env hợp lệ. Tests dùng máy chủ thật với schema test_<hash> riêng của thư mục tạm, không ghi vào public. Schema test do tiến trình tạo được dọn sau khi hoàn tất; không fallback SQLite.

BM25 bỏ qua similarity metric; Dense chấp nhận Cosine/Dot/Euclid và mặc định Cosine khi để trống. Truy vấn trống và Top-K ngoài 1–1000 trả lỗi 422 rõ ràng trước khi tạo job.

Tham chiếu: [Psycopg transactions](https://www.psycopg.org/psycopg3/docs/basic/transactions.html), [PostgreSQL Docker image](https://hub.docker.com/_/postgres).
