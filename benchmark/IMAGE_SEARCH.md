# Image-to-Image Search — PA1, PA2, PA3, PA4

Cập nhật lưu trữ: `backend.data` hiện dùng PostgreSQL cho metadata, manifest và lịch sử run thay JSON rời. Ảnh và model cache vẫn là file. Các API ảnh giữ nguyên; xem [DATABASE.md](DATABASE.md).

## Khảo sát và phạm vi thay đổi

Repository gốc chỉ có Streamlit (`app.py`, `engine.py`) và dữ liệu Excel; chưa có image search. Khu `benchmark` đã có FastAPI, React, JSON snapshots, queue một thread, Elasticsearch BM25, Qdrant text vectors, metric engine và cấu hình `.env`. Không phát hiện API ảnh hay database địa điểm production để tích hợp trực tiếp.

Tái sử dụng `backend.data` cho persistence atomic và hash, `jobs` cho queue/progress/cancel, `retrieval.qclient` cho kết nối Qdrant/timeout, `retrieval.resources/environment` cho tài nguyên và `metrics.evaluate_query/latency_summary` cho công thức. Bộ image mới nằm trong `backend/images`; ảnh, index, manifest và run nằm trong các group lưu trữ riêng. Không thay thế hoặc sửa pipeline Semantic Text Search.

Các file mới: `images/encoders.py`, `store.py`, `service.py`, `benchmark.py`, `api.py`, `playground.html`, `tests/test_images.py`, `requirements-image.txt`, tài liệu này. `main.py` chỉ gắn router và trang ảnh; `jobs.py` bổ sung khôi phục trạng thái tác vụ ảnh khi restart; `.env.example` bổ sung giới hạn upload và timeout download.

## Khởi chạy

Tại `Test_Model_DAN7/benchmark`, dùng môi trường Python có phiên bản torch/torchvision tương thích:

```powershell
python -m pip install -r requirements-image.txt
# Nếu chưa có .env:
Copy-Item .env.example .env
docker compose up -d qdrant
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Trang thử ảnh: http://127.0.0.1:8000/image-search. API schema và thử multipart: http://127.0.0.1:8000/docs, nhóm **Image-to-Image Search**. Dashboard văn bản vẫn ở `/`.

Ảnh JPEG/PNG/WEBP tối đa 20 MB, mỗi cạnh ≥16 px và tối đa 25 triệu pixel; tùy chỉnh `IMAGE_MAX_UPLOAD_MB`, `IMAGE_MAX_PIXELS` trong `.env`, restart backend. Tổng bytes một batch mặc định tối đa 100 MB (`IMAGE_MAX_BATCH_MB`). URL Qdrant dùng `QDRANT_URL` hiện có; không hardcode key. `QDRANT_LOCAL_PATH` tùy chọn cho embedded engine khi kiểm thử. `HF_HUB_DOWNLOAD_TIMEOUT`/`HF_HUB_ETAG_TIMEOUT` tùy chỉnh thời gian đọc download/model metadata; Qdrant request timeout hiện có là 120 giây. Không giới hạn cứng tổng thời gian job lớn; có hủy tại ranh giới batch/query qua API jobs.

`requirements-image.txt` giới hạn qdrant-client 1.15–1.16 để tương thích server 1.15.5 trong compose. Môi trường có sẵn client 1.20 vẫn chạy được các API kiểm thử nhưng phát cảnh báo version; nên cài dependencies vào venv theo hướng dẫn để tránh lệch phiên bản.

## Kiến trúc pipeline

```text
multipart image + place_id
  → decode/validate → EXIF transpose → alpha composite white → RGB
  → content-addressed asset (canonical lossless PNG + source/pixel SHA256)
  → queue job → model-specific preprocessing → frozen encoder → L2
  → Qdrant Cosine collection per resolved encoder/config/namespace
  → metadata + gallery hash + dimension + resources

query image → same decoder/preprocessor/encoder → Qdrant exact cosine
  → ranked image hits → optional max-score collapse by place_id → Top-K
```

Collection `img-{config-hash}` được suy ra từ model, revision đã phân giải, device, batch size, nhóm/trọng số PA1, namespace và algorithm version. Không cho trộn vector khác mô hình/cấu hình. `collection` trong API là namespace hợp lệ `[a-zA-Z0-9_-]{1,40}`, không phải quyền ghi đè một collection tùy ý. Query chọn `index_id` đã lưu, không chọn encoder độc lập có nguy cơ lệch index.

`image_id` là hash pixel sau xử lý EXIF/RGB/alpha, độc lập tên tệp; cùng ảnh cùng place/dataset trả lại asset cũ. Ảnh trùng pixel nhưng khác place/dataset bị từ chối để phát hiện liên kết mâu thuẫn. Không phát hiện near-duplicate/crop/recompressed lossy; cần curate bộ nghiên cứu. Qdrant point ID là UUID cố định từ image_id; upsert lại an toàn. Index lại cùng configuration/gallery chỉ bổ sung asset chưa có; vector trùng không được tạo. Một collection có thể chứa ảnh từ nhiều địa điểm, mỗi ảnh cần place_id. `dataset_id` tùy chọn kiểm tra place_id với dataset văn bản; nếu không cung cấp, place_id là ID hệ thống gọi khai báo.

Mỗi lần index ghi trạng thái building trước khi upsert, failed nếu lỗi, ready chỉ khi hoàn tất. Retry dùng cùng điểm cố định. Search không dùng index failed/building/interrupted. Hủy không đảo ngược các upsert đã xong; retry batch để hoàn tất. Delete yêu cầu image_id và index_id, chỉ xóa vector trong collection đó. Asset ảnh bất biến được giữ lại để các manifest/run cũ còn có bằng chứng. Gallery hash thay đổi sau bổ sung/xóa; benchmark không cho dùng collection có Gallery khác manifest.

## Image encoders

| PA | Model key | Checkpoint | Vector |
|---|---|---|---|
| PA1 | handcrafted | handcrafted-v1 | 128–2014, theo ablation |
| PA2 | mobilenet_v3_large | Torchvision IMAGENET1K_V2 | 960 |
| PA2 | resnet50 | Torchvision IMAGENET1K_V2 | 2048 |
| PA2 | efficientnet_b0 | Torchvision IMAGENET1K_V1 | 1280 |
| PA3 | dinov2_small | facebook/dinov2-small | 384 |
| PA3 | dinov2_base | facebook/dinov2-base | 768 |
| PA3 | dinov2_large | facebook/dinov2-large | 1024 |
| PA4 | clip_b32 | openai/clip-vit-base-patch32 | 512 |
| PA4 | clip_b16 | openai/clip-vit-base-patch16 | 512 |
| PA4 | siglip_b16 | google/siglip-base-patch16-224 | 768 |

PA1 resize 128×128 bicubic. HSV histogram 8×4×4; uniform LBP 8 neighbors, radius 1, 10 bins; GLCM 16 gray levels, khoảng cách [1,3], góc [0,45,90,135] độ, sáu thống kê; HOG 9 orientations, cell 16×16, block 2×2, L2-Hys; Canny thresholds 100/200, density grid 8×8. GLCM contrast/dissimilarity chia giới hạn lý thuyết 225/15, các thuộc tính còn lại đã bị chặn. Mỗi nhóm L2 → nhân trọng số không âm → concat → L2. Không fit scaler từ test data. Trọng số là multiplier của block, nên đóng góp vào tích vô hướng tỷ lệ bình phương trọng số. Không tự học trọng số theo ground truth.

Ablation đúng thứ tự: `[hsv]`, `[hsv,lbp]`, `[hsv,lbp,glcm]`, thêm `hog`, thêm `canny`. Kích thước lần lượt 128,138,186,1950,2014. Nếu một nhóm rỗng thì block là vector 0; toàn vector rỗng bị từ chối.

CNN lấy feature global average pooling trước classifier: ResNet thay `fc`, MobileNet/EfficientNet thay toàn `classifier` bằng Identity. Tránh vô tình lấy logits hoặc hidden classifier. Dùng chính xác `weights.transforms()` của checkpoint; URL weights cố định và repr transform được lưu. Revision người dùng chỉ áp dụng model Hugging Face, CNN dùng weights enum cố định.

DINOv2 lấy `last_hidden_state[:,0]` (CLS). CLIP chỉ khởi tạo `CLIPVisionModelWithProjection` và lấy projected `image_embeds`. SigLIP khởi tạo `SiglipVisionModel`, lấy `pooler_output` của learned vision pooling head. Không khởi tạo text encoder hoặc chạy text inference. Processor tải theo checkpoint cụ thể; commit hash được lưu và dùng lại khi search. Các model `.eval()`, `requires_grad_(False)`, `inference_mode()`, không fine-tune. L2 chuẩn hóa tất cả model để truy xuất cosine.

Vision snapshots Hugging Face được lưu dưới `runtime/image_model_cache/{model}/{commit}` bằng `local_dir`, không cần quyền tạo symlink trên Windows. Có thể đổi thư mục bằng `IMAGE_MODEL_CACHE_DIR`. Checkpoint thiếu vision weights bị báo lỗi, không âm thầm dùng tham số khởi tạo ngẫu nhiên.

Device auto dùng CUDA nếu khả dụng, CPU nếu không; không âm thầm fallback khi yêu cầu cuda nhưng thiếu GPU. CUDA synchronize trước/sau forward để timing không chỉ đo enqueue. Batch áp dụng indexing. Dùng chung queue một worker với text; unload text trước image job và unload image sau job. Vì vậy không giữ đồng thời toàn bộ checkpoint. Một benchmark giữ một image encoder giữa các warm queries; Playground mỗi job có thể cold-load lại. Không chạy nhiều Uvicorn workers cho kiến trúc local này.

## API contract

Tất cả endpoint indexing/search/delete trả job `{id,status,...}`. Poll `GET /api/jobs/{id}`, kết quả tại `result`; hủy bằng `POST /api/jobs/{id}/cancel`. Exception/log/traceback nằm trong trạng thái job; input validation trả 422. Ảnh không hỗ trợ URL/base64 vì API hiện có không có chuẩn đó và multipart đáp ứng yêu cầu.

| Method | Endpoint | Input / output |
|---|---|---|
| GET | `/api/ai/images/models` | registry, dimensions, năm ablation |
| POST | `/api/ai/images/assets` | multipart `file,place_id?,name?,dataset_id?`; lưu Gallery/Query asset |
| GET | `/api/ai/images/assets` | metadata assets |
| GET | `/api/ai/images/assets/{id}/content` | canonical PNG |
| POST | `/api/ai/images/index` | multipart `file,place_id,config(JSON),name?,dataset_id?` |
| POST | `/api/ai/images/index-batch` | multipart `files` lặp, `metadata(JSON array)`, `config(JSON)`; tối đa 256 ảnh/batch |
| POST | `/api/ai/images/index-assets` | JSON `{image_ids:[],config:{...}}`; dùng lại Gallery với encoder khác |
| GET | `/api/ai/images/indexes` | cấu hình và trạng thái collection |
| POST | `/api/ai/images/search` | multipart `file,index_id,top_k=10,deduplicate=true,exact=true` |
| DELETE | `/api/ai/images/{image_id}?index_id=...` | xóa vector trong collection được chỉ định |
| POST/GET | `/api/ai/images/datasets` | lưu/list manifest bất biến Gallery/Query/GT |
| POST | `/api/ai/images/benchmarks` | dataset_id, index_ids, objective, repeats, warmup, threshold, official |
| GET | `/api/ai/images/runs` | danh sách image run |
| GET | `/api/ai/images/runs/{id}` | full kết quả thực tế |
| GET | `/api/ai/images/runs/{id}/export?format=json\|csv` | JSON đầy đủ hoặc ZIP chứa CSV + provenance/rankings JSON |

Ví dụ config (JSON string trong multipart hoặc object ở index-assets):

```json
{"model":"handcrafted","groups":["hsv","lbp"],"weights":{"hsv":1,"lbp":1},"device":"auto","batch_size":8,"revision":"main","collection":"tourism"}
```

Batch metadata cùng thứ tự files: `[{"place_id":"p1","name":"Địa điểm 1"},{"place_id":"p2","name":"Địa điểm 2"}]`.

## Tạo bộ benchmark

1. Upload Gallery assets cùng place_id, index bằng cấu hình đầu tiên. Dùng image_ids đó qua index-assets cho các model/ablation tiếp theo.
2. Upload Query assets riêng, KHÔNG index chúng vào Gallery. Không cần place_id khi upload Query; nhãn exact được khai báo trong manifest.
3. Tạo manifest trên UI hoặc API. Ví dụ cấu trúc, không phải dữ liệu nghiên cứu:

```json
{
  "name":"Tourism evaluation v1",
  "gallery_image_ids":["gallery_asset_1","gallery_asset_2"],
  "queries":[{"query_id":"q1","image_id":"query_asset_1","exact_place_ids":["place-001"]}],
  "visual_target":"place",
  "judgments":[
    {"query_id":"q1","target_id":"place-001","relevance":3},
    {"query_id":"q1","target_id":"place-002","relevance":1}
  ]
}
```

Manifest kiểm tra IDs, query trùng, exact places ngoài Gallery, nhãn trùng và Query/Gallery trùng pixel. `visual_target=image` cho nhãn theo image_id thay vì place_id. Nhãn Visual 0–3 phải gán thủ công độc lập: ảnh khác địa điểm có thể rất giống về cảnh quan, không được suy ra relevance=0.

4. Chạy benchmark:

```json
{"dataset_id":"manifest_id","index_ids":["img-config1","img-config2"],"objective":"both","repeats":3,"warmup":1,"threshold":1,"official":true}
```

Exact task: gộp image hits về place trước khi đánh giá. Recall@1/5 chia số exact_place_ids được khai báo, MRR@10 theo exact identity. Mặc định một địa điểm đúng/query. Việc coi place khác là sai chỉ áp dụng **identity task**, hoàn toàn không tạo nhãn Visual.

Visual task: nDCG@10 theo gain 2^rel−1; Precision@10 và Recall@10 theo threshold 1–3. Official yêu cầu đủ nhãn trên toàn Gallery target. Thăm dò giữ null nếu Top-K thiếu nhãn; không mặc định chưa gán thành 0. No-relevant: recall=null, MRR=0, IDCG=0 → nDCG=0. Xuất số mẫu hợp lệ từng metric. Không gộp metric của hai mục tiêu thành một điểm.

Để gộp địa điểm chính xác khi có nhiều ảnh, hiện truy xuất toàn bộ image hits rồi lấy max cosine cho mỗi place. Điều này tốn chi phí O(Gallery) kết quả; là baseline đúng về thứ hạng, chưa phải tối ưu production. `visual_target=image` truy xuất Top10 image hits trực tiếp. Vì hai mục tiêu có pipeline khác, timing được lưu riêng theo objective.

Các collection phải chứa đúng cùng bộ Gallery image_ids như manifest; mọi query, GT, K và thiết bị đo dùng cùng protocol. Benchmark chạy sequential exact Qdrant để loại sai số ANN, cold sample và warm-up nằm ngoài warm latency. Phần decode đọc canonical PNG; `total_ms` gồm decode, model load nếu có, preprocessing, embedding, Qdrant round-trip và collapse. Không tính thời gian upload HTTP hoặc thời gian chờ queue. RAM là RSS tiến trình, VRAM là torch allocated, peak toàn run lấy mẫu 100 ms; đây không phải peak GPU toàn hệ thống. `vector_bytes=count×dimension×4` là float32 vector payload, không phải dung lượng HNSW/payload/disk tổng. Không tự tuyên bố model tốt nhất trước khi có dataset và run thực tế.

## Kiểm thử và giới hạn

```powershell
python -m unittest discover -s tests -v
# Kiểm tra pretrained thực (tải weights lần đầu, dùng CPU/GPU theo config):
$env:RUN_IMAGE_MODEL_TESTS='1'
$env:IMAGE_TEST_MODELS='mobilenet_v3_large,clip_b32'
python -m unittest discover -s tests -p test_images.py -v
```

Để kiểm tra toàn bộ 9 pretrained model, đặt IMAGE_TEST_MODELS thành danh sách 9 model key trong bảng (cách nhau bởi dấu phẩy). Tests chuẩn dùng ảnh fixture tổng hợp trong thư mục tạm, Qdrant local thực và handcrafted thực; kiểm tra API/queue, ablation, duplicate, replay/delete, leakage và cả hai mục tiêu benchmark. Không nhập fixture vào runtime nghiên cứu. Tests pretrained opt-in kiểm tra kích thước vector, số hữu hạn và L2, không suy ra chất lượng tìm kiếm thực từ ảnh test.

Không có dữ liệu/ground truth chính thức, nên chưa thể trả lời CNN cải thiện bao nhiêu, DINOv2 có tốt hơn CNN hay CLIP/SigLIP có tốt hơn DINOv2. Không thêm fine-tuning, text-to-image hoặc tự sinh nhãn/metric. Backend production địa điểm không tồn tại trong repo này; tích hợp hiện qua place_id và dataset_id tùy chọn. Những checkpoint lớn cần đủ RAM/dung lượng, download có thể lâu; xem job status thay vì giữ request HTTP mở.

Tham chiếu: [Torchvision pretrained transforms](https://docs.pytorch.org/vision/main/models.html), [MobileNetV3](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_large.html), [DINOv2 CLS](https://huggingface.co/docs/transformers/main/model_doc/dinov2), [CLIP vision projection](https://huggingface.co/docs/transformers/model_doc/clip), [SigLIP vision model](https://huggingface.co/docs/transformers/model_doc/siglip), [scikit-image features](https://scikit-image.org/docs/stable/api/skimage.feature.html).
