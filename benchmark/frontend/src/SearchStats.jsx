import React from 'react';
import {Clock3,MemoryStick,ListOrdered,ChevronDown} from 'lucide-react';

const fmt=value=>value==null?'—':Number(value).toLocaleString('vi-VN',{maximumFractionDigits:2});
export default function SearchStats({timing,resources,indexId,count}){
 return <div className="search-stats"><div className="search-stat-chips"><span><ListOrdered size={14}/><b>{count}</b> kết quả</span><span><Clock3 size={14}/>Tổng <b>{fmt(timing.total_ms)} ms</b></span><span><MemoryStick size={14}/>RAM <b>{fmt(resources?.ram_mb)} MB</b></span></div><details><summary>Chi tiết hiệu năng <ChevronDown size={12}/></summary><div><span>Search: <b>{fmt(timing.search_ms)} ms</b></span>{timing.embedding_ms>0&&<span>Embedding: <b>{fmt(timing.embedding_ms)} ms</b></span>}{timing.model_loading_ms>0&&<span>Load model: <b>{fmt(timing.model_loading_ms)} ms</b></span>}<small>Index: {indexId}</small></div></details></div>;
}
