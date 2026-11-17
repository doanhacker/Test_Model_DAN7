import io
from pathlib import Path
import pandas as pd
import streamlit as st
from engine import MODELS, retrieve, evaluate, split_queries

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Semantic Search Lab', page_icon='🔎', layout='wide')
st.title('🔎 Semantic Search — Research & Evaluation Lab')
st.caption('Vietnamese Tourism · So sánh mô hình · Gán nhãn · Đánh giá · Phân tích lỗi')

@st.cache_data
def sample(filename): return pd.read_csv(ROOT/'data'/filename)

def load_csv(upload, fallback):
    return pd.read_csv(upload) if upload is not None else sample(fallback).copy()

with st.sidebar:
    st.header('Cấu hình dữ liệu')
    up_places=st.file_uploader('places.csv', type='csv',key='places')
    up_queries=st.file_uploader('queries.csv',type='csv',key='queries')
    up_qrels=st.file_uploader('qrels.csv',type='csv',key='qrels')
    st.info('Dữ liệu mặc định chỉ để minh họa, chưa được thẩm định. Tải CSV thật lên để đánh giá.')
places=load_csv(up_places,'places.csv')
queries=load_csv(up_queries,'queries.csv')
qrels=load_csv(up_qrels,'qrels.csv')
for df,cols,title in [(places,['place_id','name','city','category','description'],'places'),(queries,['query_id','query','group'],'queries'),(qrels,['query_id','place_id','relevance'],'qrels')]:
    missing=set(cols)-set(df.columns)
    if missing: st.error(f'{title} thiếu cột: {sorted(missing)}'); st.stop()
for df,ids in [(places,['place_id']),(queries,['query_id']),(qrels,['query_id','place_id'])]:
    if df.duplicated(ids).any(): st.error(f'Trùng khóa {ids}; hãy xử lý trước.'); st.stop()
places['place_id']=places.place_id.astype(str)
queries['query_id']=queries.query_id.astype(str)
qrels['query_id']=qrels.query_id.astype(str)
qrels['place_id']=qrels.place_id.astype(str)
qrels['relevance']=pd.to_numeric(qrels.relevance,errors='coerce')
if qrels.relevance.isna().any() or not qrels.relevance.between(0,3).all(): st.error('relevance phải là số nguyên 0–3.'); st.stop()
if (~qrels.query_id.isin(queries.query_id)).any() or (~qrels.place_id.isin(places.place_id)).any(): st.error('qrels chứa query_id/place_id không tồn tại.'); st.stop()

summary_tab, data_tab, label_tab, run_tab, eval_tab, guide_tab=st.tabs(['Tổng quan','Dữ liệu & Chia tập','Gán nhãn','Chạy mô hình','Kết quả & Phân tích','Tiêu chí & Phương pháp'])
with summary_tab:
    a,b,c,d=st.columns(4)
    a.metric('Địa điểm',len(places)); b.metric('Truy vấn',len(queries)); c.metric('Nhãn đã gán',len(qrels)); d.metric('Truy vấn có nhãn',qrels.query_id.nunique())
    st.warning('Ground Truth hiện chưa đầy đủ: nếu Top-K có địa điểm chưa gán nhãn, không được mặc định là sai. Chế độ đánh giá nghiêm ngặt chỉ tính truy vấn có đủ nhãn trong Top-K.')
    st.subheader('Giải pháp đối chứng')
    st.dataframe(pd.DataFrame([{'Giải pháp':n,'Model ID':v or 'scikit-learn TF-IDF','Vai trò':'Lexical baseline' if v is None else 'Dense embedding'} for n,v in MODELS.items()]),hide_index=True,use_container_width=True)
with data_tab:
    st.subheader('Danh mục địa điểm'); st.dataframe(places,use_container_width=True)
    st.subheader('Truy vấn và nhóm'); st.dataframe(queries,use_container_width=True)
    st.subheader('Phân chia theo truy vấn (60/20/20)')
    seed=st.number_input('Random seed',value=42,step=1)
    split=split_queries(queries,int(seed))
    st.dataframe(split.groupby(['group','split']).size().unstack(fill_value=0),use_container_width=True)
    st.download_button('Tải queries_split.csv',split.to_csv(index=False).encode('utf-8-sig'),'queries_split.csv','text/csv')
    st.caption('Train: xây dựng nhãn và tinh chỉnh. Validation: chọn mô hình, tham số. Test: chỉ chạy sau khi khóa cấu hình. Địa điểm vẫn thuộc cùng corpus truy xuất; chia theo truy vấn để tránh rò rỉ câu hỏi.')
with label_tab:
    st.subheader('Gán nhãn relevance 0–3')
    st.caption('0: không liên quan · 1: liên quan yếu · 2: phù hợp · 3: rất phù hợp. Cần kiểm chứng thông tin địa điểm trước khi gán.')
    chosen=st.selectbox('Truy vấn',queries.query_id.tolist(),format_func=lambda x:f'{x} — {queries.loc[queries.query_id==x,"query"].iloc[0]}')
    rows=places[['place_id','name','category','description']].copy()
    existing=qrels[qrels.query_id==chosen].set_index('place_id')['relevance'].to_dict()
    rows['relevance']=rows.place_id.map(existing)
    st.caption('Để trống nghĩa là chưa được đánh giá, KHÔNG phải relevance = 0.')
    edited=st.data_editor(rows,hide_index=True,use_container_width=True,disabled=['place_id','name','category','description'],column_config={'relevance':st.column_config.NumberColumn('Relevance (0–3)',min_value=0,max_value=3,step=1)},key=f'edit_{chosen}')
    newq=qrels[qrels.query_id!=chosen].copy()
    selected=edited.dropna(subset=['relevance'])[['place_id','relevance']].copy()
    selected['query_id']=chosen
    selected=selected[['query_id','place_id','relevance']]
    combined=pd.concat([newq,selected],ignore_index=True)
    st.download_button('Tải qrels đã chỉnh sửa',combined.to_csv(index=False).encode('utf-8-sig'),'qrels_annotated.csv','text/csv')
    if st.button('Dùng nhãn vừa chỉnh sửa trong phiên này'):
        st.session_state['edited_qrels']=combined
        st.success('Đã dùng nhãn mới trong phiên; tải CSV để lưu lâu dài.')
with run_tab:
    st.subheader('Chạy Retrieval')
    models=st.multiselect('Chọn mô hình',list(MODELS),default=['TF-IDF (baseline)'])
    k=st.slider('Top-K',1,min(20,len(places)),min(5,len(places)))
    batch=st.number_input('Batch size',min_value=1,max_value=128,value=16)
    st.caption('Mô hình embedding sẽ tải weights ở lần chạy đầu. TF-IDF chạy CPU không cần tải mô hình. Thời gian encode bao gồm tải/khởi tạo model, KHÔNG đại diện online latency.')
    if st.button('▶ Chạy benchmark',type='primary'):
        if not models: st.warning('Chọn ít nhất một mô hình.')
        else:
            runs=[]; timings=[]
            for name in models:
                with st.spinner(f'Đang chạy {name}...'):
                    try:
                        res,timing=retrieve(places,queries,name,k,batch)
                        runs.append(res);timings.append(timing)
                    except Exception as e: st.error(f'{name}: {e}')
            if runs:
                st.session_state['runs']=pd.concat(runs,ignore_index=True)
                st.session_state['timings']=pd.DataFrame(timings)
                st.session_state['run_k']=k
                st.success('Hoàn tất. Xem tab Kết quả & Phân tích.')
with eval_tab:
    if 'runs' not in st.session_state:
        st.info('Hãy chạy benchmark ở tab Chạy mô hình.')
    else:
        results=st.session_state['runs']; k=int(st.session_state['run_k'])
        active_qrels=st.session_state.get('edited_qrels',qrels)
        mode=st.radio('Cách xử lý ảnh/địa điểm chưa được gán nhãn', ['Nghiêm ngặt: bỏ truy vấn có Top-K chưa được gán nhãn','Minh họa: coi chưa gán nhãn là 0 (thiên lệch)'],horizontal=False)
        unjudged='exclude' if mode.startswith('Nghiêm') else 'zero'
        scores,per=evaluate(results,active_qrels,queries,k,unjudged)
        if scores.empty:
            st.warning('Không có truy vấn đủ điều kiện để tính metric theo chế độ này. Hãy gán nhãn toàn bộ Top-K (hoặc chuyển sang chế độ minh họa có thiên lệch).')
        else:
            st.subheader('So sánh chỉ số')
            st.dataframe(scores,use_container_width=True,hide_index=True)
            st.bar_chart(scores.set_index('model')[['ndcg_at_k','mrr_at_k','recall_at_k']])
            st.download_button('Tải metrics.csv',scores.to_csv(index=False).encode('utf-8-sig'),'metrics.csv','text/csv')
            st.subheader('Theo từng nhóm truy vấn')
            st.dataframe(per.groupby(['model','group'])[['ndcg@k','recall@k','mrr@k']].mean(),use_container_width=True)
        st.subheader('Thời gian chạy (không phải online latency)')
        st.dataframe(st.session_state['timings'],use_container_width=True,hide_index=True)
        st.subheader('Kiểm tra lỗi từng truy vấn')
        qid=st.selectbox('Query ID',queries.query_id.tolist(),key='review_query')
        view=results[results.query_id==qid].copy()
        view=view.merge(active_qrels[['query_id','place_id','relevance']],on=['query_id','place_id'],how='left')
        st.dataframe(view[['model','rank','name','score','relevance','place_id']],use_container_width=True,hide_index=True)
        st.download_button('Tải retrieval_results.csv',results.to_csv(index=False).encode('utf-8-sig'),'retrieval_results.csv','text/csv')
with guide_tab:
    st.markdown('''### Ma trận tiêu chí
| Nhóm | Chỉ số | Cách dùng |
|---|---|---|
| Chất lượng | nDCG@10 | **Chính**: phù hợp nhãn 0–3 và thứ hạng |
| Chất lượng | Recall@10 | Phụ: cần nhãn dương tương đối đầy đủ trong corpus |
| Chất lượng | MRR@10 | Phụ: vị trí kết quả phù hợp đầu tiên |
| Chất lượng | Precision@5 | Phụ: tỷ lệ phù hợp trong top 5 |
| Hiệu năng | Query encode P50/P95 | Đo riêng sau warm-up, không tính tải model |
| Hiệu năng | Search P50/P95 | Đo riêng Qdrant, cố định HNSW và phần cứng |
| Tài nguyên | RAM/VRAM, model size | Theo dõi khi chạy thực tế |
| Độ ổn định | Điểm theo nhóm query | Phân tích truy vấn dễ/khó, có dấu/không dấu |

### Quy tắc đánh giá
1. **Khóa tập test** trước khi chọn mô hình. Không sửa nhãn test để cải thiện điểm.
2. Dùng cùng một corpus, cùng các trường nội dung, không lọc rating/rerank trong vòng so sánh embedding.
3. Với E5 áp dụng tiền tố `query:` và `passage:` theo cách dùng của mô hình.
4. Chạy Dense exact cosine trước; sau đó mới kiểm tra tác động ANN/Qdrant và hybrid.
5. Nên dùng pooling Top-K của tất cả mô hình để gán nhãn; ưu tiên đánh giá chéo bởi 2 người.
6. **Chưa gán nhãn ≠ không liên quan**. nDCG/Recall trên qrels không đầy đủ có thể thiên lệch.
7. Tập minh họa 20 địa điểm/30 câu **không đủ để kết luận mô hình tốt nhất**.
8. Truy vấn không có địa điểm phù hợp cần đánh giá riêng theo tỷ lệ từ chối/không trả kết quả, không gộp Recall.
''')
