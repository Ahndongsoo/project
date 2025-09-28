# server.py
import os, time, json, sqlite3, threading
from typing import List, Optional
from fastapi import FastAPI, Body, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DB_PATH = "meta.sqlite"
INDEX_PATH = "rag.index"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

app = FastAPI(title="Hearim RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- DB ----------
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = db()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS speech (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id TEXT NOT NULL,
      text TEXT NOT NULL,
      context TEXT,
      tags TEXT,       -- JSON array string
      updated_at REAL
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS rx (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id TEXT NOT NULL,
      date TEXT,
      dx TEXT,
      drug TEXT,
      dose TEXT,
      doctor TEXT
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS meta (
      k TEXT PRIMARY KEY,
      v TEXT
    );
    """)
    conn.commit()
    conn.close()

init_db()

# ---------- Embedding / Index ----------
_model = SentenceTransformer(MODEL_NAME)
_index = None               # faiss.IndexFlatIP
_id_map = []                # list of (row_id)
_dim = 768                  # model output dim (MiniLM-L12-v2 is 384~768 계열, 모델에 맞춰 확인)

def _embed(texts: List[str]) -> np.ndarray:
    vecs = _model.encode(texts, normalize_embeddings=True)  # cosine=inner product if normalized
    return np.array(vecs).astype("float32")

def _load_index():
    global _index, _id_map, _dim
    if os.path.exists(INDEX_PATH) and os.path.exists("id_map.json"):
        _index = faiss.read_index(INDEX_PATH)
        with open("id_map.json","r") as f: _id_map = json.load(f)
    else:
        _index = faiss.IndexFlatIP(_dim)
        _id_map = []

def _save_index():
    faiss.write_index(_index, INDEX_PATH)
    with open("id_map.json","w") as f: json.dump(_id_map, f)

_load_index()

def _set_status(key, value):
    conn = db()
    conn.execute("INSERT INTO meta(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (key, value))
    conn.commit(); conn.close()

def _get_status(key, default=""):
    conn = db()
    cur = conn.execute("SELECT v FROM meta WHERE k=?", (key,))
    row = cur.fetchone()
    conn.close()
    return row["v"] if row else default

# ---------- Schemas ----------
class SpeechIn(BaseModel):
    patient_id: str
    text: str
    context: Optional[str] = ""
    tags: Optional[List[str]] = []

class UpsertIn(BaseModel):
    items: List[SpeechIn]

class SearchIn(BaseModel):
    patientId: str
    query: Optional[str] = ""
    topK: int = 5

# ---------- Helpers ----------
def _speech_row_to_doc(r) -> str:
    # 임베딩 입력: text + context + tags를 합쳐 의미 보강
    tags = r["tags"] or "[]"
    return " ".join([
        r["text"] or "",
        r["context"] or "",
        " ".join(json.loads(tags))
    ])

def _add_to_index(row_id: int, text_for_embed: str):
    global _index, _id_map
    v = _embed([text_for_embed])
    _index.add(v)
    _id_map.append(row_id)

def _rebuild_index():
    global _index, _id_map
    conn = db()
    cur = conn.execute("SELECT * FROM speech ORDER BY id")
    rows = cur.fetchall(); conn.close()
    docs = [_speech_row_to_doc(r) for r in rows]
    vecs = _embed(docs)
    _index = faiss.IndexFlatIP(vecs.shape[1])
    _index.add(vecs.astype("float32"))
    _id_map = [int(r["id"]) for r in rows]
    _save_index()
    _set_status("last_updated", str(time.time()))

# ---------- API ----------
@app.get("/api/rag/status")
def status():
    last = _get_status("last_updated", "0")
    return {
        "last_updated": float(last),
        "count": len(_id_map)
    }

@app.post("/api/rag/upsert")
def upsert(payload: UpsertIn):
    conn = db()
    cur = conn.cursor()
    inserted_ids = []
    for it in payload.items:
        cur.execute(
            "INSERT INTO speech(patient_id,text,context,tags,updated_at) VALUES(?,?,?,?,?)",
            (it.patient_id, it.text, it.context or "", json.dumps(it.tags or []), time.time())
        )
        rid = cur.lastrowid
        inserted_ids.append(rid)
        # 인덱스에 즉시 증분 반영
        _add_to_index(rid, _speech_row_to_doc({
            "text": it.text, "context": it.context or "", "tags": json.dumps(it.tags or [])
        }))
    conn.commit(); conn.close()
    _save_index()
    _set_status("last_updated", str(time.time()))
    return {"ok": True, "inserted": inserted_ids, "count": len(_id_map)}

@app.post("/api/rag/search")
def search(q: SearchIn):
    # 1) 전체에서 TopN 뽑고, 2) patient_id로 1차 필터, 3) 쿼리 키워드로 리랭크(가벼운 가산)
    if len(_id_map) == 0:
        return {"items": []}
    query_vec = _embed([q.query or ""])[0:1]
    D, I = _index.search(query_vec, min(50, len(_id_map)))  # 넉넉히 뽑아둠
    ids = [ _id_map[i] for i in I[0] ]
    conn = db()
    rows = []
    for rid, score in zip(ids, D[0]):
        r = conn.execute("SELECT * FROM speech WHERE id=?", (rid,)).fetchone()
        if not r: continue
        if r["patient_id"] != q.patientId:  # 환자 필터
            continue
        rows.append((r, float(score)))
    conn.close()

    # 간단 리랭크: query 단어가 text/context/tags에 포함되면 가산
    key = (q.query or "").lower().strip()
    ranked = []
    for r, s in rows:
        text = _speech_row_to_doc(r).lower()
        bonus = 0.05 if key and key in text else 0.0
        ranked.append((r, s + bonus))
    ranked.sort(key=lambda x: x[1], reverse=True)
    ranked = ranked[: max(1, min(q.topK, 10))]
    items = [{
        "text": r["text"],
        "context": r["context"],
        "tags": json.loads(r["tags"] or "[]"),
        "score": round(float(s), 4)
    } for r, s in ranked]
    return {"items": items}

@app.get("/api/rag/prescriptions")
def rx(patientId: str = Query(...), q: str = Query("", alias="q")):
    conn = db()
    cur = conn.execute("SELECT date,dx,drug,dose,doctor FROM rx WHERE patient_id=?", (patientId,))
    rows = [dict(x) for x in cur.fetchall()]
    conn.close()
    if q:
        ql = q.lower()
        rows = [r for r in rows if ql in json.dumps(r, ensure_ascii=False).lower()]
    return {"rows": rows}

@app.post("/api/rag/reindex")
def reindex():
    _rebuild_index()
    return {"ok": True, "count": len(_id_map), "last_updated": float(_get_status("last_updated","0"))}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

<script>
// (1) 처방 이력
async function fetchPrescriptions(patientId, query){
  const r = await fetch(`/api/rag/prescriptions?patientId=${encodeURIComponent(patientId)}&q=${encodeURIComponent(query||"")}`);
  const j = await r.json();
  return j.rows || [];
}

// (2) RAG 검색
async function fetchSpeechRAG(patientId, query, topK=5){
  const r = await fetch('/api/rag/search', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({patientId, query, topK})
  });
  const j = await r.json();
  return (j.items||[]).map(o=>({
    text:o.text, context:o.context, tags:o.tags, score:o.score
  }));
}

// (3) 상태 뱃지(선택)
async function refreshStatusBadge(){
  const r = await fetch('/api/rag/status'); const j = await r.json();
  const t = new Date((j.last_updated||0)*1000);
  document.querySelector('#rxCount')?.setAttribute('title',
    `인덱스 문서: ${j.count} · 업데이트: ${t.toLocaleString()}`
  );
}
</script>



<!-- HEADER 오른쪽 버튼 옆에 추가 -->
<button id="btnSync" class="px-4 py-2 rounded-xl bg-emerald-700 hover:bg-emerald-600">🔄 RAG 업데이트</button>

<script>
// 데모용: 현재 화면의 patient와 임시 입력 예시를 업서트 (운영에선 관리자/배치 파이프라인이 호출)
document.querySelector('#btnSync').addEventListener('click', async ()=>{
  const pid = document.querySelector('#patientSelect').value;
  const payload = {
    items: [
      { patient_id: pid, text:"물 한 잔만 주이소", context:"대기실", tags:["음료요청"] },
      { patient_id: pid, text:"배가 칼로 쑤십니더", context:"진료 대기", tags:["통증","복통"] }
    ]
  };
  const r = await fetch('/api/rag/upsert', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(payload)
  });
  await r.json();
  await runRAG();          // 화면 갱신
  await refreshStatusBadge();
});
</script>

