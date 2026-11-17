"""Prepare the user's real Excel data and reviewable queries; never fabricate judgments."""
import ast
import hashlib
import io
import json
from collections import Counter

import pandas as pd

from . import data, retrieval

EXCEL_MAPPING={'place_id':'Mã địa điểm','name':'Tên địa điểm','category':'Danh mục','city':'Thành phố','address':'Địa chỉ','description':'Mô tả','tags':'Thẻ sở thích','amenities':'Tiện ích và đặc điểm'}
SEARCH_FIELDS=['name','category','description','address','tags','amenities']


def flatten(value):
    if isinstance(value,dict):
        return [s for v in value.values() for s in flatten(v)]
    if isinstance(value,list):
        return [s for v in value for s in flatten(v)]
    if not value:
        return []
    if isinstance(value,str) and value.strip().startswith(('{','[')):
        try:
            return flatten(ast.literal_eval(value))
        except (SyntaxError,ValueError):
            pass
    return [s.strip() for s in str(value).split(';') if s.strip()]


def import_excel(raw,filename,city='Đà Nẵng'):
    workbook=pd.ExcelFile(io.BytesIO(raw))
    sheet='Địa điểm chung' if 'Địa điểm chung' in workbook.sheet_names else workbook.sheet_names[0]
    frame=pd.read_excel(workbook,sheet_name=sheet,dtype=str,keep_default_na=False)
    required={'Mã địa điểm','Tên địa điểm','Thành phố'}
    if required-set(frame):
        raise ValueError('Excel không đúng schema du lịch: thiếu '+str(sorted(required-set(frame))))
    report=dict(filename=filename,source_sha256=hashlib.sha256(raw).hexdigest(),sheet=sheet,source_rows=len(frame),city=city,
                city_counts=frame['Thành phố'].value_counts().to_dict(),excluded_city=0,invalid_rows=[],duplicate_ids=[],source_links={})
    rows={}
    for number,source in enumerate(frame.to_dict('records'),2):
        if city and source['Thành phố'].strip()!=city:
            report['excluded_city']+=1
            continue
        row={field:str(source.get(column,'')).strip() for field,column in EXCEL_MAPPING.items()}
        if not row['place_id'] or not row['name']:
            report['invalid_rows'].append(number)
            continue
        row['category']='; '.join(dict.fromkeys([source.get('Loại địa điểm',''),row['category']])).strip('; ')
        row['tags']=flatten(row['tags'])
        row['amenities']=flatten(row['amenities'])
        key=row['place_id']
        report['source_links'].setdefault(key,[]).append({'sheet':sheet,'row':number,'source_place_id':key})
        if key in rows:
            if rows[key]['name']!=row['name'] or rows[key]['address']!=row['address']:
                raise ValueError(f'ID {key} trùng nhưng khác tên/địa chỉ; cần xử lý thủ công, không tự gộp')
            report['duplicate_ids'].append(key)
            for field in ('description','category'):
                if row[field] and row[field]!=rows[key][field]:
                    rows[key][field]+='\n'+row[field]
            for field in ('tags','amenities'):
                rows[key][field]=list(dict.fromkeys(rows[key][field]+row[field]))
        else:
            rows[key]=row
    snapshot=data.import_records('places',list(rows.values()),f'{filename} · {city or "tất cả"}',original=raw,mapping={})
    report.update(dataset_id=snapshot['id'],valid_places=len(rows),merged_duplicate_rows=len(report['duplicate_ids']))
    data.save('source_reports',snapshot['id'],report)
    from . import place_media
    images=place_media.import_excel(raw,filename,snapshot['id'],city)
    # Source links kept separately so place records retain their established schema.
    return {'dataset_id':snapshot['id'],'valid_places':len(rows),'source_rows':len(frame),'excluded_city':report['excluded_city'],
            'merged_duplicate_rows':report['merged_duplicate_rows'],'invalid_rows':len(report['invalid_rows']),'source_sha256':report['source_sha256'],'images':images}


# Each recipe is checked against the imported corpus before it is used.
RECIPES=[
 ('bãi biển','bãi biển','địa điểm có bãi cát và không gian ven biển','Tôi muốn ra biển đi dạo và tận hưởng không khí ven bờ'),
 ('công viên','công viên','không gian công viên để dạo chơi','Tôi muốn đi bộ thư giãn trong không gian công viên'),
 ('bảo tàng','bảo tàng','địa điểm trưng bày hiện vật và giới thiệu văn hóa','Tôi muốn tìm hiểu văn hóa qua các hiện vật được trưng bày'),
 ('chùa','chùa','địa điểm tham quan mang nét văn hóa Phật giáo','Tôi muốn ghé một nơi thờ Phật trong chuyến tham quan'),
 ('cầu','cầu','công trình cầu để tham quan','Tôi muốn ghé xem những cây cầu trong thành phố'),
 ('hải sản','nhà hàng hải sản','địa điểm ăn uống chuyên về hải sản','Tôi muốn thưởng thức các món ăn từ hải sản'),
 ('nhà hàng Việt Nam','nhà hàng Việt Nam','địa điểm phục vụ món ăn Việt','Tôi muốn ăn các món truyền thống Việt Nam'),
 ('ăn chay','địa điểm ăn chay','địa điểm có lựa chọn ăn chay','Tôi đang muốn tìm một bữa ăn chay'),
 ('quán cà phê','quán cà phê','địa điểm uống cà phê','Tôi muốn ngồi thưởng thức một ly cà phê'),
 ('tiệm bánh','tiệm bánh','cửa hàng bán bánh','Tôi muốn tìm một cửa hàng để mua bánh'),
 ('bể bơi','khách sạn có bể bơi','nơi lưu trú có tiện ích bơi lội','Tôi muốn nghỉ ở một nơi có hồ để bơi'),
 ('spa','địa điểm có spa','địa điểm có dịch vụ spa','Tôi muốn dành thời gian thư giãn với dịch vụ spa'),
 ('Wi-Fi miễn phí','chỗ có Wi-Fi miễn phí','địa điểm cung cấp kết nối Wi-Fi miễn phí','Tôi cần một nơi có thể dùng mạng không dây miễn phí'),
 ('Bữa sáng miễn phí','khách sạn có bữa sáng miễn phí','nơi lưu trú bao gồm bữa sáng miễn phí','Tôi muốn nghỉ ở nơi không phải trả thêm tiền bữa sáng'),
 ('Đỗ xe miễn phí','địa điểm có đỗ xe miễn phí','địa điểm có chỗ đỗ xe không thu phí','Tôi đi ô tô và muốn tìm nơi đỗ xe không mất tiền'),
 ('Xe đưa đón ra sân bay','khách sạn có xe đưa đón sân bay','nơi lưu trú có dịch vụ đưa đón sân bay','Tôi cần chỗ nghỉ hỗ trợ di chuyển ra sân bay'),
 ('Bể bơi ngoài trời','khách sạn có bể bơi ngoài trời','nơi lưu trú có khu bơi ngoài trời','Tôi muốn bơi ở hồ ngoài trời trong thời gian lưu trú'),
 ('homestay','homestay','nơi lưu trú dạng homestay','Tôi muốn tìm chỗ ở theo kiểu homestay'),
 ('chợ','chợ','địa điểm chợ và mua sắm','Tôi muốn dạo chợ và xem các mặt hàng được bán'),
 ('nhà hàng gia đình','nhà hàng gia đình','địa điểm ăn uống theo mô hình nhà hàng gia đình','Tôi muốn tìm một nhà hàng cho bữa ăn gia đình'),
]


def generate_queries(dataset_id):
    ds=data.get('datasets',dataset_id)
    if ds['kind']!='places':
        raise ValueError('Cần dataset địa điểm')
    rows=ds['records']
    queries=[]
    evidence=[]
    names=set()
    for n,(term,label,feature,semantic) in enumerate(RECIPES):
        matches=[r for r in rows if term.casefold() in data.text(r,['name','category','tags','amenities']).casefold()]
        if label.startswith('khách sạn'):
            matches=[r for r in matches if 'hotel' in r['category'].casefold() or 'khách sạn' in r['category'].casefold()]
        if not matches:
            raise ValueError(f'Không đủ dữ liệu cho chủ đề {term}; cần chọn chủ đề khác từ corpus')
        anchor=next((r for r in matches if r['name'] not in names),matches[0])
        names.add(anchor['name'])
        extra=next((s for s in anchor['amenities']+anchor['tags'] if s.casefold()!=term.casefold() and s.casefold() not in label.casefold()),None)
        multi=f'Tìm {label} có {extra} ở Đà Nẵng' if extra else f'Tìm {label} ở địa chỉ {anchor["address"]}'
        texts=[anchor['name'],f'Tìm {label} ở Đà Nẵng',f'Tìm {feature} tại Đà Nẵng',semantic+' ở Đà Nẵng',multi]
        for group,text in zip(('ten_rieng','tu_khoa_truc_tiep','mo_ta_dac_diem','ngu_nghia','nhieu_dieu_kien'),texts):
            qid=f'{group}-{n+1:02d}'
            queries.append(dict(query_id=qid,query_text=text,query_type=group,split='train' if n<12 else 'validation' if n<16 else 'test'))
            evidence.append(dict(query_id=qid,anchor_place_id=anchor['place_id'],matched_term=term,extra=extra,
                                 evidence={f:anchor[f] for f in SEARCH_FIELDS},review_status='pending',
                                 note='Truy vấn nháp từ corpus; anchor không phải Ground Truth và không tự mang nhãn dương.'))
    if len({q['query_text'] for q in queries})!=100:
        raise ValueError('Truy vấn bị trùng; cần xem lại chủ đề')
    snapshot=data.import_records('queries',queries,'100 truy vấn Đà Nẵng · nháp cần duyệt',dataset_id=dataset_id)
    data.save('query_evidence',snapshot['id'],dict(id=snapshot['id'],dataset_id=dataset_id,records=evidence))
    return {'queries_id':snapshot['id'],'count':100,'groups':dict(Counter(q['query_type'] for q in queries))}


def create_pool(index_id,queries_id,job):
    index=data.get('indexes',index_id)
    qs=data.get('datasets',queries_id)
    if index['config']['method']!='BM25' or qs['kind']!='queries' or qs.get('dataset_id') not in (None,index['config']['dataset_id']):
        raise ValueError('Cần BM25 index và đúng query set của dataset')
    candidates=[]
    timings=[]
    for i,q in enumerate(qs['records']):
        job.check()
        answer=retrieval.search(index_id,q['query_text'],20)
        timings.append({'query_id':q['query_id'],**answer['timing'],**answer['resources']})
        candidates.extend(dict(query_id=q['query_id'],place_id=r['place_id'],rank=r['rank'],score=r['score'],relevance=None,
                               reason='',review_status='pending') for r in answer['results'])
        job.update(i+1,len(qs['records']),'Tạo ứng viên Top20 · '+q['query_id'])
    gt=data.import_records('qrels',candidates,'BM25 Top20 · chưa gán nhãn',index['config']['dataset_id'],queries_id)
    pool=dict(id=gt['id'],dataset_id=index['config']['dataset_id'],queries_id=queries_id,index_id=index_id,
              created_at=data.now(),records=candidates,timings=timings)
    data.save('candidate_pools',gt['id'],pool)
    return {'qrels_id':gt['id'],'candidates':len(candidates),'queries':len(qs['records'])}


def coverage(dataset_id,queries_id,qrels_id):
    ds=data.get('datasets',dataset_id)
    qs=data.get('datasets',queries_id)
    gt=data.get('datasets',qrels_id)
    if gt['dataset_id']!=dataset_id or gt['queries_id']!=queries_id:
        raise ValueError('Ground Truth không khớp phiên bản')
    confirmed=[r for r in gt['records'] if r.get('review_status','confirmed')=='confirmed' and r['relevance'] is not None]
    counts=Counter(r['query_id'] for r in confirmed)
    return dict(confirmed=len(confirmed),pending=len(gt['records'])-len(confirmed),total_possible=len(ds['records'])*len(qs['records']),
                missing_corpus=len(ds['records'])*len(qs['records'])-len(confirmed),per_query=dict(counts),
                warning='Pool BM25 Top20 không đảm bảo Recall trên toàn corpus. Nhãn chưa xác nhận không được dùng làm Ground Truth.')
