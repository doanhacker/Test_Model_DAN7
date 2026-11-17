# AI Search Benchmark — PA1 & PA2

Menu **Tính Recall / MRR / nDCG** hỗ trợ gán nhãn và tính từng truy vấn, xem mẫu số và đóng góp từng hạng; xem [hướng dẫn đánh giá](EVALUATION.md).

Dữ liệu và lịch sử mỗi lần chạy lưu trên **máy chủ PostgreSQL 16** qua Docker Compose; xem [hướng dẫn database](DATABASE.md). Dữ liệu cũ đã được chuyển sang PostgreSQL; các file cũ chỉ giữ làm bản dự phòng.

Menu **Kiểm chứng định tính** cho phép chọn phương án/index, nhập truy vấn tự do và so sánh Top-K cạnh nhau; hiển thị điểm, thứ hạng, mô tả, địa chỉ, tags, tiện ích, timing và RAM. Chỉ các phương án đã index đúng dataset mới chọn được. Nhận xét từng kết quả và nhận xét chung lưu trong PostgreSQL (`documents`, group `qualitative_reviews`); mở lại qua lịch sử. Những nhận xét này độc lập với Ground Truth và không tự tham gia metric chính thức.

Kết quả Search Playground và Kiểm chứng định tính kèm ảnh đại diện/gallery từ Excel, có nút xem ảnh trước/sau. Ảnh tải lần đầu được cache byte trong PostgreSQL. Địa điểm thiếu ảnh hoặc URL hết hiệu lực có trạng thái rõ ràng; không dùng ảnh giả. Timing tìm kiếm bao gồm bước gắn metadata ảnh, không bao gồm thời gian browser tải/render ảnh qua request riêng.

Image-to-Image Search PA1–PA4 được bổ sung độc lập tại `/image-search`; xem [hướng dẫn Image Search](IMAGE_SEARCH.md) và cài `requirements-image.txt` để dùng các image encoder.

Khu thực nghiệm độc lập: React + Vite + Tailwind, FastAPI, Elasticsearch BM25, Qdrant dense retrieval. Không thay đổi `../app.py`, `../engine.py` hoặc dữ liệu Excel nguồn. Không có Hybrid, RRF, sparse hay multi-vector. Không tự import dữ liệu hay tạo kết quả nghiên cứu.

## Chạy trên Windows

Tại thư mục `Test_Model_DAN7/benchmark`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
cd frontend
npm.cmd ci
npm.cmd run build
cd ..
docker compose up -d
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Mở http://127.0.0.1:8000. OpenAPI: http://127.0.0.1:8000/docs. Docker Desktop phải đang chạy. Có thể dùng `./start.ps1` sau khi cài thư viện và build; `-SkipDocker` nếu đã có dịch vụ bên ngoài. Dừng server bằng Ctrl+C; `docker compose stop` để dừng hai dịch vụ và giữ dữ liệu. Dev UI: `npm.cmd run dev` trong frontend; Vite proxy `/api` tới cổng 8000.

Backend tự nạp `benchmark/.env` khi khởi động. Sao chép `.env.example` thành `.env` nếu chưa có; chỉnh URL dịch vụ hoặc đường dẫn dữ liệu tại đó rồi khởi động lại backend. Biến môi trường đã đặt trên hệ điều hành được ưu tiên hơn `.env`. File `.env` được bỏ qua bởi Git; `.env.example` dùng làm mẫu chia sẻ.

Biến môi trường tùy chọn: `ELASTICSEARCH_URL`, `QDRANT_URL`, `BENCHMARK_DATA_DIR` (mặc định `runtime/`). `QDRANT_LOCAL_PATH` chỉ dành cho kiểm thử embedded, để unset khi dùng Qdrant server. Server chỉ bind loopback. Cấu hình compose dành cho môi trường thực nghiệm local, không phải cấu hình public production.

## Quy trình

1. Quản lý dữ liệu → import địa điểm (CSV/XLSX/JSON), xem preview và mapping `tên schema → tên cột nguồn`. Các ID được đọc như chuỗi, giữ số 0 đầu. Xác nhận số bản ghi và trường thiếu.
2. Import truy vấn với `query_id,query_text,query_type,split`; split phải là `train`, `validation` hoặc `test`. Các nhóm có thể là tên địa điểm, danh mục, hoạt động, ngữ nghĩa, sở thích, nhiều điều kiện.
3. Import Ground Truth có `query_id,place_id,relevance`, chọn đúng dataset và query version. Hoặc mở giao diện gán nhãn thủ công. Relevance nguyên 0–3; để trống là chưa gán nhãn. Mỗi lần lưu tạo phiên bản mới, không sửa snapshot cũ.
4. Cấu hình & Indexing → chọn dataset/trường, tạo BM25 và từng model dense. Theo dõi tác vụ phía dưới. Mỗi cấu hình tạo index mới; index lỗi không được chọn để tìm kiếm. Dense tải weights từ Hugging Face lần đầu, có thể cần nhiều GB RAM/VRAM/dung lượng. Chỉ một model thường trú.
5. Playground → chọn một/nhiều index cùng dataset/cùng thứ tự trường, nhập truy vấn, chọn K/metric/exact. Chọn truy vấn từ Ground Truth nếu muốn hiện nhãn cạnh kết quả. Điểm BM25 không so trực tiếp với cosine/dot/khoảng cách.
6. Benchmark Runner → chọn query/GT version khớp index, split, K, threshold, warm-up và số lần lặp. Cold start đo riêng, các warm-up không tính vào mẫu. Chạy chính thức yêu cầu nhãn toàn corpus cho mỗi query và exact search. Chế độ thăm dò vẫn không gán nhãn thiếu thành 0.
7. Chọn run tại Kết quả, Phân tích lỗi hoặc Xuất báo cáo. Dùng validation để chọn tham số; chỉ báo cáo test sau khi khóa cấu hình. Không dùng nhãn test để tự điều chỉnh mô hình.

Template CSV rỗng trong `templates/`; API `/api/templates/{places|queries|qrels}?format=csv|xlsx|json` tạo template các định dạng. JSON dùng mảng object; `tags`/`amenities` là mảng chuỗi; CSV/XLSX có thể dùng chuỗi JSON hoặc dấu `;`. Chỉ `place_id,name` bắt buộc cho địa điểm; các trường còn lại để trống được thống kê. JSON template là mảng rỗng; schema được mô tả tại đây và trên UI.

```json
{"place_id":"place-001","name":"Tên địa điểm","category":"ATTRACTION","city":"Đà Nẵng","address":"","description":"","tags":["biển"],"amenities":[]}
```

Ví dụ này chỉ minh họa schema, không được cài làm dataset. File Excel đã có trong repository chưa được sử dụng vì chưa được xác nhận là dataset chính thức.

## Các quyết định thực nghiệm

- BM25: mặc định k1=1.2, b=.75. Custom similarity được gắn lên từng trường; query multi_match `most_fields`. Mỗi lần đổi thông số tạo index tên mới. Analyzer `standard + lowercase`, tùy chọn `asciifolding`; cùng analyzer dùng cho index và query. Đây chưa phải bộ phân đoạn từ ghép tiếng Việt. Unicode NFC và khoảng trắng chuẩn hóa nhất quán. Cấu hình analyzer được lưu nguyên trong metadata.
- M1 MiniLM: 384 chiều, 128 token, mean pooling. M2 E5-base: 768 chiều; M3 E5-large: 1024 chiều, 512 token, masked mean pooling, tiền tố `query: ` và `passage: `. M4 BGE-M3: 1024 chiều, 8192 token, CLS pooling; chỉ dense. SentenceTransformer tải cấu trúc checkpoint, không áp một pooling thủ công chung.
- Revision nhập mặc định `main`, được phân giải thành commit hash khi index; search tải đúng hash đó. Thiết bị, batch, max length, normalization và kiến trúc thực tế được lưu. Các module Normalize nội bộ được bỏ trước khi áp cờ normalize để tắt L2 có hiệu lực.
- Một lần encode corpus dùng chung cho ba collection Cosine/Dot/Euclid. Tệp `.npy` giữ embedding nguồn. Qdrant tự chuẩn hóa vector với Cosine, vì vậy cờ L2 không thay đổi ý nghĩa cosine. Với vector L2: cosine=dot, Euclid²=2−2·dot; thứ hạng về lý thuyết tương đương (có thể khác cách phá hòa điểm).
- Exact Qdrant (`SearchParams(exact=True)`) mặc định; có thể chạy ANN trên Playground/thăm dò. Metric là cấu hình truy vấn chọn collection tương ứng, không thay đổi khoảng cách collection đã tạo.
- nDCG dùng gain `2^rel−1`, không áp threshold lên gain. Precision/Recall/MRR dùng threshold 1–3. Duplicate result ID bị từ chối. Precision chia K kể cả khi trả ít hơn K. Không có nhãn dương: Recall=null, MRR=0; IDCG=0 thì nDCG=0. Summary hiển thị số mẫu hợp lệ cho từng metric, không lấp null bằng 0. Các query không đủ nhãn vẫn được lưu trong detail.
- Official yêu cầu nhãn toàn corpus, nên Recall/IDCG có mẫu số đã biết. Thăm dò chỉ tính query có đủ nhãn Top-K; Recall/nDCG dựa trên nhãn hiện có, có thiên lệch. Tất cả run và export ghi rõ official/provisional.
- Total đo load (nếu có), embedding và search sau khi đọc metadata local. Search gồm round-trip client đến engine. Mẫu cold của mỗi phương án được lưu riêng; cold model là model bị unload, không phải xóa OS/disk/server cache. ES cache không bị flush. Warm-up dùng truy vấn đầu tiên. RAM là RSS toàn tiến trình, VRAM là torch allocated; peak lấy mẫu 100 ms trong benchmark, có thể bỏ lỡ spike ngắn. Chi phí model load không gộp warm latency. Indexing time bao gồm encode và upsert. BM25 index size lấy `_stats/store`; Qdrant API chuẩn không có disk bytes đáng tin cậy nên `index_size_bytes=null`, lưu riêng kích thước tệp vector và ghi chú.

## Lưu trữ và API

PostgreSQL server: dataset, địa điểm, truy vấn, nhãn, index metadata, jobs và lịch sử run được ghi trong transaction PostgreSQL. Các thư mục JSON cũ chỉ còn là bản legacy sau migration. `originals`: bytes upload nguyên bản. `vectors`: embeddings. Snapshot có SHA256 nội dung; GT liên kết đúng hai version. Gói JSON chứa cấu hình, hash, môi trường, Ground Truth, query set, cold sample và mọi thứ hạng/timing thô. CSV xuất ZIP gồm bảng + provenance JSON. XLSX gồm bảng tổng hợp, chi tiết, thứ hạng, provenance; PNG là biểu đồ thực tế, không điền số liệu khi chưa có run.

API chính: `GET /api/health`, `/models`, `/datasets`, `/indexes`, `/jobs`, `/runs`; `POST /api/preview`, `/import`, `/labels`, `/indexes`, `/models/load`, `/models/unload`, `/search`, `/benchmarks`, `/jobs/{id}/cancel`; `GET /api/runs/{id}/export?format=...`. Xem schema trên `/docs`.

Tác vụ model/index/search/benchmark chạy trong một worker thread nền để tránh tải nhiều model cùng lúc và giữ request HTTP ngắn. Hủy có hiệu lực sau thao tác model/batch/query đang chạy; không cưỡng bức ngắt CUDA hoặc HTTP request. Không chạy nhiều Uvicorn worker với local storage/model manager này. Server restart đánh dấu tác vụ/index/run dang dở là interrupted/failed; không tự resume hay báo hoàn tất giả. Đây là hàng đợi local đơn tiến trình; dữ liệu lớn/phân tán có thể thay bằng durable queue sau này.

## Kiểm thử

```powershell
python -m unittest discover -s tests -v
cd frontend
npm.cmd run build
```

Tests dùng dữ liệu tạm đánh dấu fixture trong test, không import vào runtime. Unit tests công thức tính tay, nhãn thiếu, no-relevant, duplicate, ID/foreign key/relevance, các định dạng import và mapping. API test dùng mock retrieval để kiểm tra pipeline job → đánh giá → export; không tuyên bố mock là chất lượng BM25/model thực. Integration tests tùy chọn ghi rõ điều kiện dịch vụ.

Tài liệu tham chiếu: [Elasticsearch similarity](https://www.elastic.co/docs/reference/elasticsearch/index-settings/similarity), [Qdrant search](https://qdrant.tech/documentation/search/search/), [Qdrant collections](https://qdrant.tech/documentation/manage-data/collections/), [MiniLM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), [E5-base](https://huggingface.co/intfloat/multilingual-e5-base), [E5-large](https://huggingface.co/intfloat/multilingual-e5-large), [BGE-M3](https://huggingface.co/BAAI/bge-m3).
# PA2 Dense Semantic Search

Mục **PA2 · Dense Semantic Search** dùng bốn mô hình trên CPU và ba collection Qdrant cho Cosine/Dot/Euclid. Xem [hướng dẫn PA2](PA2_DENSE_SEARCH.md) để lập chỉ mục dữ liệu thật và kiểm chứng trên giao diện.
