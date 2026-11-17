import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

MODELS = {
    'TF-IDF (baseline)': None,
    'MiniLM multilingual': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'Multilingual E5 base': 'intfloat/multilingual-e5-base',
    'Multilingual E5 large': 'intfloat/multilingual-e5-large',
    'BGE-M3 dense': 'BAAI/bge-m3',
}

def document_text(places):
    return (places['name'].fillna('') + '. ' + places['category'].fillna('') + '. ' + places['city'].fillna('') + '. ' + places['description'].fillna('')).tolist()

def retrieve(places, queries, model_name, k=10, batch_size=16):
    docs = document_text(places)
    qs = queries['query'].fillna('').astype(str).tolist()
    t0 = time.perf_counter()
    if MODELS[model_name] is None:
        vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), sublinear_tf=True)
        d = vec.fit_transform(docs)
        q = vec.transform(qs)
        encode_s = time.perf_counter() - t0
        scores = (q @ d.T).toarray()
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODELS[model_name])
        e5 = 'E5' in model_name
        d_texts = ['passage: ' + s for s in docs] if e5 else docs
        q_texts = ['query: ' + s for s in qs] if e5 else qs
        d = model.encode(d_texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        q = model.encode(q_texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        encode_s = time.perf_counter() - t0
        scores = np.asarray(q) @ np.asarray(d).T
    search_start = time.perf_counter()
    records = []
    k = min(int(k), len(places))
    for i, query in queries.reset_index(drop=True).iterrows():
        top = np.argsort(-scores[i], kind='stable')[:k]
        for rank, j in enumerate(top, 1):
            p = places.iloc[j]
            records.append({'model': model_name, 'query_id': str(query.query_id), 'group': str(query.group), 'query': str(query.query), 'rank': rank, 'place_id': str(p.place_id), 'name': str(p['name']), 'score': float(scores[i, j])})
    search_s = time.perf_counter() - search_start
    return pd.DataFrame(records), {'model': model_name, 'encode_seconds_total': round(encode_s, 4), 'retrieval_seconds_total': round(search_s, 4), 'queries': len(queries), 'documents': len(places)}

def evaluate(results, qrels, queries, k=10, unjudged='exclude'):
    # Only queries with >= 1 positive judgement; no claim of unbiased recall with incomplete labels.
    labels = {(str(r.query_id), str(r.place_id)): int(r.relevance) for _, r in qrels.iterrows()}
    all_qids = set(queries.query_id.astype(str))
    per = []
    for (model, qid), group in results.groupby(['model', 'query_id'], sort=False):
        if qid not in all_qids: continue
        gt = {pid: rel for (q, pid), rel in labels.items() if q == qid}
        n_positive = sum(v > 0 for v in gt.values())
        if not n_positive: continue
        g = group.sort_values('rank').head(k)
        rels = [gt.get(str(pid), None) for pid in g.place_id]
        if unjudged == 'exclude' and any(r is None for r in rels):
            continue
        rels = [0 if r is None else r for r in rels]
        binary = np.asarray([int(r > 0) for r in rels])
        precision = binary.sum() / k
        recall = binary.sum() / n_positive
        dcg = sum((2**r-1)/np.log2(i+2) for i,r in enumerate(rels))
        ideal = sorted(gt.values(), reverse=True)[:k]
        idcg = sum((2**r-1)/np.log2(i+2) for i,r in enumerate(ideal))
        ndcg = dcg/idcg if idcg else 0
        first = next((1/(i+1) for i,r in enumerate(rels) if r>0),0)
        per.append({'model':model,'query_id':qid,'group':str(g.iloc[0]['group']), 'precision@k':precision,'recall@k':recall,'ndcg@k':ndcg,'mrr@k':first,'judged_top_k':sum(r is not None for r in [gt.get(str(pid)) for pid in g.place_id])})
    detail = pd.DataFrame(per)
    if detail.empty: return pd.DataFrame(), detail
    summary = detail.groupby('model',as_index=False).agg(queries_evaluated=('query_id','nunique'), precision_at_k=('precision@k','mean'), recall_at_k=('recall@k','mean'), ndcg_at_k=('ndcg@k','mean'), mrr_at_k=('mrr@k','mean'))
    return summary, detail

def split_queries(queries, seed=42):
    # Query-level, group-stratified, reproducible split; small groups may not populate every split.
    rng = np.random.default_rng(seed)
    out=[]
    for group, df in queries.groupby('group'):
        ix = rng.permutation(len(df))
        n=len(df)
        ntrain=int(round(n*.6)); nval=int(round(n*.2))
        for pos, row_i in enumerate(ix):
            split='train' if pos<ntrain else ('validation' if pos<ntrain+nval else 'test')
            row=df.iloc[row_i].to_dict(); row['split']=split; out.append(row)
    return pd.DataFrame(out).sort_values('query_id').reset_index(drop=True)
