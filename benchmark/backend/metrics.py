"""No implicit zero labels. Official evaluation requires complete corpus judgments."""
import math
import numpy as np


def evaluate_query(ranking, labels, corpus_ids, k=10, threshold=1, official=True):
    if k < 1 or threshold not in (1, 2, 3):
        raise ValueError('K/ngưỡng relevance không hợp lệ')
    if any(not isinstance(v, int) or not 0 <= v <= 3 for v in labels.values()):
        raise ValueError('Nhãn phải là số nguyên 0–3')
    if len(ranking) != len(set(ranking)):
        raise ValueError('Kết quả tìm kiếm bị trùng place_id')
    corpus = set(corpus_ids)
    if set(ranking) - corpus or set(labels) - corpus:
        raise ValueError('place_id ngoài dataset')
    top = ranking[:k]
    coverage = len(labels) / len(corpus) if corpus else 0
    complete_corpus=set(labels)==corpus
    complete_top=bool(labels) and all(p in labels for p in top)
    rels = [labels.get(p) for p in top]
    positives = sum(v >= threshold for v in labels.values())
    hits = sum(v is not None and v >= threshold for v in rels)
    first=next((i+1 for i,r in enumerate(rels) if r is not None and r>=threshold),None)
    prefix_known=bool(first) and all(r is not None for r in rels[:first])
    rr=1/first if prefix_known else 0 if complete_top and first is None else None
    dcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels)) if complete_top else None
    ideal = sorted(labels.values(), reverse=True)[:k]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    quality_ready=complete_top and (complete_corpus or not official)
    trace=[dict(rank=i+1,place_id=pid,relevance=labels.get(pid),binary_relevant=None if pid not in labels else labels[pid]>=threshold,
                gain=None if pid not in labels else 2**labels[pid]-1,discount=math.log2(i+2),
                dcg_contribution=None if pid not in labels else (2**labels[pid]-1)/math.log2(i+2)) for i,pid in enumerate(top)]
    return dict(coverage=coverage,judged_top_k=sum(p in labels for p in top),returned=len(top),
                missing_top_k=sum(p not in labels for p in top),missing_corpus=len(corpus)-len(labels),judged_total=len(labels),
                hits_at_k=hits,known_relevant=positives,first_relevant_rank=first if prefix_known else None,
                dcg=dcg,idcg=idcg if quality_ready else None,trace=trace,
                recall_status='exact' if quality_ready and complete_corpus and positives else 'pooled' if quality_ready and positives else 'no_relevant' if quality_ready else 'needs_labels',
                ndcg_status=('exact' if complete_corpus else 'pooled') if quality_ready else 'needs_labels',
                mrr_status='exact' if rr is not None else 'needs_prefix_labels',
                status='incomplete_labels' if not quality_ready else 'no_relevant' if positives==0 else 'official' if official else 'provisional',
                precision=hits/k if complete_top else None,recall=hits/positives if quality_ready and positives else None,
                mrr=rr,ndcg=(dcg/idcg if idcg else 0) if quality_ready else None)


def latency_summary(values):
    if not values:
        return {key: None for key in ('average_ms', 'p50_ms', 'p95_ms', 'p99_ms')}
    return {'average_ms': float(np.mean(values)), **{f'p{p}_ms': float(np.percentile(values, p)) for p in (50, 95, 99)}}
