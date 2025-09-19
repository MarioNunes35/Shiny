
# =============================================================================
# origin_assistant_complete.py — Finalized single-file Streamlit app
# =============================================================================
import os, io, re, hmac, json, math, base64, hashlib, secrets, datetime as dt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import streamlit as st

try:
    import anthropic  # type: ignore
    _ANTHROPIC_AVAILABLE = True
except Exception:
    _ANTHROPIC_AVAILABLE = False

st.set_page_config(page_title="Origin Assistant · RAG + Claude", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")

def now_iso() -> str: return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def read_secret(path: List[str], default: Optional[str] = None) -> Optional[str]:
    try:
        node = st.secrets
        for k in path: node = node[k]
        return str(node)
    except Exception:
        return default

PBKDF2_ALGO="pbkdf2_sha256"; PBKDF2_ITERATIONS=210_000; SALT_BYTES=16; KEY_LEN=32
_b64 = lambda b: base64.b64encode(b).decode("utf-8")
_b64d = lambda s: base64.b64decode(s.encode("utf-8"))
def hash_password(p:str,*,iterations:int=PBKDF2_ITERATIONS)->str:
    salt=secrets.token_bytes(SALT_BYTES)
    dk=hashlib.pbkdf2_hmac("sha256", p.encode(), salt, iterations, dklen=KEY_LEN)
    return f"{PBKDF2_ALGO}${iterations}${_b64(salt)}${_b64(dk)}"
def verify_password(p:str, stored:str)->bool:
    try:
        algo,iters_s,salt_b64,hash_b64=stored.split("$")
        if algo!=PBKDF2_ALGO: return False
        iters=int(iters_s); salt=_b64d(salt_b64); expected=_b64d(hash_b64)
        test=hashlib.pbkdf2_hmac("sha256", p.encode(), salt, iters, dklen=len(expected))
        return hmac.compare_digest(expected, test)
    except Exception: return False

def load_users_from_secrets()->Dict[str,str]:
    users={}
    try:
        raw=st.secrets.get("auth",{}).get("users",[])
        for u in raw or []:
            if isinstance(u,dict) and "username" in u and "password_hash" in u:
                users[str(u["username"]).strip().lower()]=str(u["password_hash"])
    except Exception: pass
    return users
def save_user_in_session(u:str): st.session_state["user"]={"username":u,"login_at":now_iso()}
def current_user()->Optional[str]: return st.session_state.get("user",{}).get("username")
def logout(): st.session_state.pop("user",None); st.session_state.pop("chat",None)
def try_register(u:str,p:str,tok:str)->Tuple[bool,str]:
    exp=read_secret(["auth","admin_token"],"")
    if not exp or tok!=exp: return False,"Admin token inválido."
    if not u or not p: return False,"Usuário e senha são obrigatórios."
    u=u.strip().lower()
    if u in st.session_state.get("user_db",{}): return False,"Usuário já existe."
    st.session_state["user_db"][u]=hash_password(p); return True,f"Usuário '{u}' registrado (sessão)."
def ensure_user_db():
    if "user_db" not in st.session_state:
        st.session_state["user_db"]={}; st.session_state["user_db"].update(load_users_from_secrets())
def login_form():
    ensure_user_db(); st.markdown("### Entrar")
    with st.form("login_form"):
        u=st.text_input("Usuário"); p=st.text_input("Senha",type="password"); ok=st.form_submit_button("Entrar")
    if ok:
        if u and p and u.strip().lower() in st.session_state["user_db"] and verify_password(p, st.session_state["user_db"][u.strip().lower()]):
            save_user_in_session(u.strip().lower()); st.success("Login concluído."); st.rerun()
        else: st.error("Credenciais inválidas ou usuário não encontrado.")
def register_form():
    st.markdown("### Registrar (admin)")
    with st.form("register_form"):
        u=st.text_input("Novo usuário"); p=st.text_input("Nova senha",type="password"); t=st.text_input("Admin token"); ok=st.form_submit_button("Registrar")
    if ok:
        s,msg=try_register(u,p,t); (st.success if s else st.error)(msg)

@dataclass
class ClaudeClient:
    api_key: Optional[str]=field(default=None)
    model: str=field(default="claude-3-5-sonnet-latest")
    max_tokens: int=field(default=1024)
    def available(self)->bool: return bool(self.api_key) and _ANTHROPIC_AVAILABLE
    def complete(self, prompt:str)->str:
        if not self.available(): return "(Simulação Claude) "+prompt[:300]
        try:
            client=anthropic.Anthropic(api_key=self.api_key)
            msg=client.messages.create(model=self.model,max_tokens=self.max_tokens,messages=[{"role":"user","content":prompt}])
            parts=[]
            content=getattr(msg,"content",[])
            for blk in content:
                if isinstance(blk,dict) and blk.get("type")=="text": parts.append(blk.get("text",""))
                elif getattr(blk,"type","")== "text": parts.append(getattr(blk,"text",""))
            return "\n".join([p for p in parts if p]) or "(Claude sem conteúdo)"
        except Exception as e: return f"(Erro Claude) {e}"

TOKEN_PATTERN=re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+", re.UNICODE)
@dataclass
class Chunk: doc_id:str; chunk_id:int; text:str
@dataclass
class TfIdfIndex:
    chunks:List[Chunk]=field(default_factory=list)
    df:Dict[str,int]=field(default_factory=lambda: defaultdict(int))
    idf:Dict[str,float]=field(default_factory=dict)
    norms:List[float]=field(default_factory=list)
    vocab:Dict[str,int]=field(default_factory=dict)
    weights:List[Dict[int,float]]=field(default_factory=list)
    def clear(self): self.chunks.clear(); self.df.clear(); self.idf.clear(); self.norms.clear(); self.vocab.clear(); self.weights.clear()
    @staticmethod
    def tokenize(txt:str)->List[str]: return [t.lower() for t in TOKEN_PATTERN.findall(txt)]
    def add_documents(self, docs:Dict[str,str], *, chunk_size:int=900, overlap:int=120):
        self.clear()
        def split_chunks(t:str)->List[str]:
            toks=self.tokenize(t); out=[]; i=0
            while i<len(toks):
                j=min(i+chunk_size,len(toks)); out.append(" ".join(toks[i:j]))
                if j==len(toks): break
                i=max(0,j-overlap)
            return out
        for doc_id,text in docs.items():
            for i,part in enumerate(split_chunks(text)): self.chunks.append(Chunk(doc_id, i, part))
        for ch in self.chunks:
            seen=set()
            for tok in self.tokenize(ch.text):
                if tok not in self.vocab: self.vocab[tok]=len(self.vocab)
                if tok not in seen: self.df[tok]+=1; seen.add(tok)
        N=max(1,len(self.chunks))
        for tok,df in self.df.items(): self.idf[tok]=math.log((1+N)/(1+df))+1.0
        self.weights=[]; self.norms=[]
        for ch in self.chunks:
            counts=Counter(self.tokenize(ch.text)); vec={}
            for tok,c in counts.items():
                idx=self.vocab.get(tok); 
                if idx is None: continue
                tf=1+math.log(c); idf=self.idf.get(tok,0.0); vec[idx]=tf*idf
            norm=math.sqrt(sum(v*v for v in vec.values())) or 1.0
            self.weights.append(vec); self.norms.append(norm)
    def search(self, query:str, k:int=5)->List[Tuple[float,Chunk]]:
        if not query.strip() or not self.chunks: return []
        q_counts=Counter(self.tokenize(query)); q_vec={}
        for tok,c in q_counts.items():
            idx=self.vocab.get(tok); 
            if idx is None: continue
            tf=1+math.log(c); idf=self.idf.get(tok,0.0); q_vec[idx]=tf*idf
        q_norm=math.sqrt(sum(v*v for v in q_vec.values())) or 1.0
        scores=[]
        for i,vec in enumerate(self.weights):
            dot=sum(val*vec.get(idx,0.0) for idx,val in q_vec.items())
            sim=dot/(q_norm*self.norms[i])
            scores.append((sim,i))
        scores.sort(reverse=True,key=lambda x:x[0])
        return [(s,self.chunks[i]) for s,i in scores[:k]]

def ensure_index():
    if "rag_index" not in st.session_state: st.session_state["rag_index"]=TfIdfIndex()
    if "rag_docs" not in st.session_state: st.session_state["rag_docs"]={}
def add_uploaded_docs(files):
    ensure_index(); docs=st.session_state["rag_docs"]
    for f in files or []:
        name=getattr(f,"name",f"doc_{len(docs)+1}.txt")
        try:
            raw=f.read()
            try: text=raw.decode("utf-8",errors="ignore")
            except Exception: text=raw.decode("latin-1",errors="ignore")
            docs[name]=text
        except Exception as e: st.warning(f"Falha ao ler '{name}': {e}")
    st.session_state["rag_index"].add_documents(docs)
def rebuild_index_if_needed(force=False):
    ensure_index()
    if force or (st.session_state.get("rag_dirty") and st.session_state["rag_docs"]):
        st.session_state["rag_index"].add_documents(st.session_state["rag_docs"]); st.session_state["rag_dirty"]=False

CSS = """
:root { --bg:#0b1324; --panel:#0f1a33; --muted:#95a1c1; --primary:#67a7ff; --accent:#22d3ee; --radius:16px; }
.stApp { background: radial-gradient(1200px 800px at 10% -10%, #0e1b38 0%, var(--bg) 55%); color:#e6edf7; }
.block-container{ padding-top:1.2rem; padding-bottom:2rem; max-width:1250px; }
.chat-user,.chat-bot{ border-radius:16px; padding:.7rem .9rem; margin-bottom:.5rem; white-space:pre-wrap }
.chat-user{ background:rgba(34,211,238,.10); border:1px solid rgba(34,211,238,.25) }
.chat-bot{ background:rgba(59,130,246,.10); border:1px solid rgba(59,130,246,.25) }
.badge{ display:inline-flex; gap:6px; font-size:.78rem; color:#cfe3ff; background:rgba(103,167,255,.12); border:1px solid rgba(103,167,255,.25); border-radius:999px; padding:.25rem .55rem }
@media (min-width: 1000px){ .grid-2{ display:grid; grid-template-columns:1.15fr .85fr; gap:1rem } }
@media (max-width: 999px){ .grid-2{ display:block } }
.footer{ color:var(--muted); text-align:center; margin-top:1rem; font-size:.8rem; opacity:.85 }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
st.markdown("""<script>const t=parent.document.querySelector('.block-container'); if(t){t.scrollTop=t.scrollHeight}</script>""", unsafe_allow_html=True)

def header():
    c1,c2=st.columns([.8,.2])
    with c1:
        st.markdown("## 🧪 Origin Assistant — RAG + Claude")
        st.caption("Auth PBKDF2 • RAG TF‑IDF • Claude opcional")
    with c2:
        if current_user():
            st.markdown(f"<div class='badge'>🔐 {current_user()}</div>", unsafe_allow_html=True)
            if st.button("Sair"): logout(); st.rerun()

def handle_upload_and_index():
    st.subheader("📚 Base de Conhecimento (RAG)")
    up=st.file_uploader("Envie .txt/.md/.csv/.tsv", type=["txt","md","csv","tsv"], accept_multiple_files=True)
    if up: add_uploaded_docs(up); st.success(f"{len(up)} arquivo(s) adicionados.")
    if st.button("Reindexar documentos"): rebuild_index_if_needed(True); st.info("Reindexação concluída.")
    ensure_index()
    if st.session_state["rag_docs"]:
        with st.expander("Ver documentos carregados"):
            for k,v in st.session_state["rag_docs"].items(): st.markdown(f"- `{k}` — {len(v)} caracteres")
    else: st.caption("Nenhum documento na base ainda.")

def format_context(res,top_k=4)->str:
    return "\n\n".join([f"[{ch.doc_id}#{ch.chunk_id} | score={s:.3f}]\n{ch.text}" for s,ch in res[:top_k]])

def chat_section(claude):
    st.subheader("💬 Chat")
    prompt=st.text_area("Sua pergunta ou mensagem", height=120)
    cA,cB,cC=st.columns([.2,.2,.6])
    with cA: top_k=st.number_input("Top-K RAG",1,10,4)
    with cB: use_claude=st.toggle("Usar Claude", value=bool(claude.available()))
    with cC: temperature=st.slider("Temperatura (Claude)",0.0,1.0,0.3,0.1)
    if "chat" not in st.session_state: st.session_state["chat"]=[]
    if st.button("Enviar", type="primary") and prompt.strip():
        st.session_state["chat"].append({"role":"user","content":prompt})
        ensure_index(); results=st.session_state["rag_index"].search(prompt, k=top_k)
        context=format_context(results, top_k=top_k) if results else ""
        sys_hint=("Você é um assistente especializado em materiais, polímeros e ciência. "
                  "Responda de forma direta e cite trechos do contexto com [doc#chunk] quando útil.")
        composed=f"{sys_hint}\n\n## Contexto selecionado\n{context}\n\n## Pergunta\n{prompt}"
        reply=claude.complete(composed) if use_claude else "(Modo local) "+composed[:1200]
        st.session_state["chat"].append({"role":"assistant","content":reply})
    if st.session_state["chat"]:
        for m in st.session_state["chat"][-50:]:
            cls="chat-user" if m["role"]=="user" else "chat-bot"
            st.markdown(f"<div class='{cls}'>{m['content']}</div>", unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1:
        if st.button("Limpar chat"): st.session_state["chat"]=[]; st.rerun()
    with c2:
        if st.button("Copiar última"):
            last=next((m for m in reversed(st.session_state["chat"]) if m["role"]=="assistant"), None)
            st.code(last["content"] if last else "—")
    with c3:
        if st.button("Exportar .json"):
            data=json.dumps(st.session_state["chat"], ensure_ascii=False, indent=2)
            st.download_button("Baixar chat.json", data, file_name="chat_export.json")

def auth_section():
    if not current_user():
        st.info("Entre ou registre um usuário (admin token exigido nos *Secrets*).")
        c1,c2=st.columns(2)
        with c1: login_form()
        with c2: register_form()
        st.stop()

def server():
    if "user_db" not in st.session_state: ensure_user_db()
    if "rag_docs" not in st.session_state: ensure_index()
    api_key=read_secret(["anthropic","api_key"], os.getenv("ANTHROPIC_API_KEY"))
    model=read_secret(["anthropic","model"], "claude-3-5-sonnet-latest") or "claude-3-5-sonnet-latest"
    claude=ClaudeClient(api_key=api_key, model=model, max_tokens=1024)
    header(); auth_section()
    st.markdown("<div class='grid-2'>", unsafe_allow_html=True)
    with st.container(): handle_upload_and_index(); st.markdown("---"); chat_section(claude)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='footer'>Origin Assistant · {now_iso()}</div>", unsafe_allow_html=True)

if __name__=="__main__":
    server()




