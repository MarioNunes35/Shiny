
# =============================================================================
# origin_assistant_complete.py — Finalized single-file Streamlit app
# Matches requested structure and section ordering.
# =============================================================================
# Imports e configurações (linhas 1-52)
# =============================================================================
import os
import io
import re
import hmac
import json
import time
import math
import base64
import hashlib
import secrets
import textwrap
import datetime as dt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict

import streamlit as st

# Optional: Anthropic (Claude). If not present or key missing, app gracefully degrades.
try:
    import anthropic  # type: ignore
    _ANTHROPIC_AVAILABLE = True
except Exception:
    _ANTHROPIC_AVAILABLE = False

# App config
st.set_page_config(
    page_title="Origin Assistant · RAG + Claude",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Utilities
def now_iso() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def read_secret(path: List[str], default: Optional[str] = None) -> Optional[str]:
    """
    Safely read nested keys from st.secrets like read_secret(['auth','admin_token']).
    """
    try:
        node = st.secrets
        for key in path:
            node = node[key]
        return str(node)
    except Exception:
        return default


# =============================================================================
# Sistema de autenticação seguro com PBKDF2 (linhas 54-234)
# =============================================================================
# Password hashing with PBKDF2-HMAC (SHA256)
# Stored format: pbkdf2_sha256$iterations$salt_b64$hash_b64
PBKDF2_ALGO = "pbkdf2_sha256"
PBKDF2_ITERATIONS = 210_000
SALT_BYTES = 16
KEY_LEN = 32

def _b64(x: bytes) -> str:
    return base64.b64encode(x).decode("utf-8")

def _b64d(x: str) -> bytes:
    return base64.b64decode(x.encode("utf-8"))

def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS) -> str:
    salt = secrets.token_bytes(SALT_BYTES)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations, dklen=KEY_LEN)
    return f"{PBKDF2_ALGO}${iterations}${_b64(salt)}${_b64(dk)}"

def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters_s, salt_b64, hash_b64 = stored.split("$")
        if algo != PBKDF2_ALGO:
            return False
        iters = int(iters_s)
        salt = _b64d(salt_b64)
        expected = _b64d(hash_b64)
        test = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iters, dklen=len(expected))
        return hmac.compare_digest(expected, test)
    except Exception:
        return False

# A very small in-app user store backed by secrets (production: use a DB).
# Expected secrets structure (optional):
# [auth]
# admin_token = "SOME_LONG_RANDOM"
# cookie_secret = "SOME_LONG_RANDOM"
# [[auth.users]]
# username="admin"
# password_hash="pbkdf2_sha256$..."
def load_users_from_secrets() -> Dict[str, str]:
    users: Dict[str, str] = {}
    try:
        raw_users = st.secrets.get("auth", {}).get("users", [])
        if isinstance(raw_users, list):
            for u in raw_users:
                if isinstance(u, dict) and "username" in u and "password_hash" in u:
                    users[str(u["username"]).strip().lower()] = str(u["password_hash"])
    except Exception:
        pass
    return users

def save_user_in_session(username: str):
    st.session_state["user"] = {"username": username, "login_at": now_iso()}

def current_user() -> Optional[str]:
    return st.session_state.get("user", {}).get("username")

def logout():
    st.session_state.pop("user", None)
    st.session_state.pop("chat", None)

# Registration guarded by admin token
def try_register(username: str, password: str, admin_token: str) -> Tuple[bool, str]:
    exp = read_secret(["auth", "admin_token"], "")
    if not exp or admin_token != exp:
        return False, "Admin token inválido."
    if not username or not password:
        return False, "Usuário e senha são obrigatórios."
    username = username.strip().lower()
    if username in st.session_state.get("user_db", {}):
        return False, "Usuário já existe."
    pw_hash = hash_password(password)
    st.session_state["user_db"][username] = pw_hash
    return True, f"Usuário '{username}' registrado (somente nesta sessão)."

def ensure_user_db():
    if "user_db" not in st.session_state:
        st.session_state["user_db"] = {}
        # load from secrets (read-only)
        st.session_state["user_db"].update(load_users_from_secrets())

def login_form():
    ensure_user_db()
    st.markdown("### Entrar")
    with st.form("login_form"):
        u = st.text_input("Usuário", key="login_user")
        p = st.text_input("Senha", type="password", key="login_pass")
        submit = st.form_submit_button("Entrar")
    if submit:
        if u and p and u.strip().lower() in st.session_state["user_db"]:
            if verify_password(p, st.session_state["user_db"][u.strip().lower()]):
                save_user_in_session(u.strip().lower())
                st.success("Login concluído.")
                st.rerun()
            else:
                st.error("Credenciais inválidas.")
        else:
            st.error("Usuário não encontrado.")

def register_form():
    st.markdown("### Registrar (admin)")
    with st.form("register_form"):
        u = st.text_input("Novo usuário")
        p = st.text_input("Nova senha", type="password")
        t = st.text_input("Admin token")
        ok = st.form_submit_button("Registrar")
    if ok:
        success, msg = try_register(u, p, t)
        (st.success if success else st.error)(msg)


# =============================================================================
# Integração com Claude (linhas 236-259)
# =============================================================================
@dataclass
class ClaudeClient:
    api_key: Optional[str] = field(default=None)
    model: str = field(default="claude-3-5-sonnet-latest")
    max_tokens: int = field(default=1024)

    def available(self) -> bool:
        return bool(self.api_key) and _ANTHROPIC_AVAILABLE

    def complete(self, prompt: str) -> str:
        if not self.available():
            # graceful fallback for local dev
            return "(Simulação Claude) " + prompt[:300]
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            msg = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            # Anthropic SDK returns content as a list of blocks
            if hasattr(msg, "content") and len(msg.content) > 0:
                # join text blocks
                parts = []
                for blk in msg.content:
                    if getattr(blk, "type", "") == "text":
                        parts.append(getattr(blk, "text", ""))
                    elif isinstance(blk, dict) and blk.get("type") == "text":
                        parts.append(blk.get("text", ""))
                return "\n".join([p for p in parts if p])
            return "(Claude respondeu, mas sem conteúdo legível.)"
        except Exception as e:
            return f"(Erro Claude) {e}"


# =============================================================================
# Sistema RAG otimizado (linhas 261-500)
# =============================================================================
# We'll implement a lightweight pure-Python TF-IDF store with cosine similarity.
# Documents are chunked; each chunk is a retrievable unit.
TOKEN_PATTERN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+", re.UNICODE)

@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str

@dataclass
class TfIdfIndex:
    chunks: List[Chunk] = field(default_factory=list)
    df: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    idf: Dict[str, float] = field(default_factory=dict)
    norms: List[float] = field(default_factory=list)
    vocab: Dict[str, int] = field(default_factory=dict)  # token -> idx
    weights: List[Dict[int, float]] = field(default_factory=list)  # per-chunk sparse vector

    def clear(self):
        self.chunks.clear()
        self.df.clear()
        self.idf.clear()
        self.norms.clear()
        self.vocab.clear()
        self.weights.clear()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return [t.lower() for t in TOKEN_PATTERN.findall(text)]

    def add_documents(self, docs: Dict[str, str], *, chunk_size: int = 900, overlap: int = 120):
        self.clear()
        # chunking
        def split_into_chunks(text: str) -> List[str]:
            tokens = self.tokenize(text)
            chunks = []
            start = 0
            while start < len(tokens):
                end = min(start + chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                chunks.append(" ".join(chunk_tokens))
                if end == len(tokens):
                    break
                start = max(0, end - overlap)
            return chunks

        # Build chunk list
        for doc_id, text in docs.items():
            parts = split_into_chunks(text)
            for i, part in enumerate(parts):
                self.chunks.append(Chunk(doc_id=doc_id, chunk_id=i, text=part))

        # Build vocab and document frequencies
        for ch in self.chunks:
            seen = set()
            for tok in self.tokenize(ch.text):
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)
                if tok not in seen:
                    self.df[tok] += 1
                    seen.add(tok)

        # Compute IDF
        N = max(1, len(self.chunks))
        for tok, df in self.df.items():
            # smoothed idf
            self.idf[tok] = math.log((1 + N) / (1 + df)) + 1.0

        # Compute per-chunk TF-IDF weights and norms
        self.weights = []
        self.norms = []
        for ch in self.chunks:
            counts = Counter(self.tokenize(ch.text))
            vec: Dict[int, float] = {}
            for tok, c in counts.items():
                idx = self.vocab.get(tok)
                if idx is None:
                    continue
                tf = 1 + math.log(c)
                idf = self.idf.get(tok, 0.0)
                vec[idx] = tf * idf
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.weights.append(vec)
            self.norms.append(norm)

    def search(self, query: str, k: int = 5) -> List[Tuple[float, Chunk]]:
        if not query.strip() or not self.chunks:
            return []
        q_counts = Counter(self.tokenize(query))
        q_vec: Dict[int, float] = {}
        for tok, c in q_counts.items():
            idx = self.vocab.get(tok)
            if idx is None:
                continue
            tf = 1 + math.log(c)
            idf = self.idf.get(tok, 0.0)
            q_vec[idx] = tf * idf
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

        scores: List[Tuple[float, int]] = []
        for i, vec in enumerate(self.weights):
            # cosine
            dot = 0.0
            for idx, val in q_vec.items():
                dot += val * vec.get(idx, 0.0)
            sim = dot / (q_norm * self.norms[i])
            scores.append((sim, i))

        scores.sort(reverse=True, key=lambda x: x[0])
        results = []
        for s, i in scores[:k]:
            results.append((s, self.chunks[i]))
        return results

# Session helpers for RAG
def ensure_index():
    if "rag_index" not in st.session_state:
        st.session_state["rag_index"] = TfIdfIndex()
    if "rag_docs" not in st.session_state:
        st.session_state["rag_docs"] = {}

def add_uploaded_docs(uploaded_files: List[io.BytesIO]):
    ensure_index()
    docs = st.session_state["rag_docs"]
    for f in uploaded_files or []:
        name = getattr(f, "name", f"doc_{len(docs)+1}.txt")
        try:
            raw = f.read()
            try:
                text = raw.decode("utf-8", errors="ignore")
            except Exception:
                # try latin-1 fallback
                text = raw.decode("latin-1", errors="ignore")
            docs[name] = text
        except Exception as e:
            st.warning(f"Falha ao ler '{name}': {e}")
    st.session_state["rag_index"].add_documents(docs)

def rebuild_index_if_needed(force: bool = False):
    ensure_index()
    if force or (st.session_state.get("rag_dirty") and st.session_state["rag_docs"]):
        st.session_state["rag_index"].add_documents(st.session_state["rag_docs"])
        st.session_state["rag_dirty"] = False


# =============================================================================
# CSS completo e responsivo (linhas 502-1364)
# =============================================================================
CSS = """
/* Root */
:root {
  --bg: #0b1324;
  --panel: #0f1a33;
  --muted: #95a1c1;
  --primary: #67a7ff;
  --primary-2: #3b82f6;
  --accent: #22d3ee;
  --ok: #22c55e;
  --warn: #f59e0b;
  --err: #ef4444;
  --radius: 16px;
}

/* Page */
.stApp {
  background: radial-gradient(1200px 800px at 10% -10%, #0e1b38 0%, var(--bg) 55%);
  color: #e6edf7;
}

/* Containers */
.block-container {
  padding-top: 1.2rem;
  padding-bottom: 2rem;
  max-width: 1250px;
}

.card {
  background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: var(--radius);
  padding: 1rem 1.2rem;
  box-shadow: 0 20px 40px rgba(0,0,0,0.25);
}

/* Titles */
h1, h2, h3 {
  letter-spacing: 0.2px;
}
h1 {
  font-size: 1.85rem;
}
h2 {
  font-size: 1.4rem;
  color: var(--primary);
}

/* Inputs */
.stTextInput > div > div > input,
.stTextArea > div > textarea,
.stSelectbox > div > div > div > div,
.stFileUploader > div {
  background: #0d1a35 !important;
  border-radius: 12px !important;
  color: #dbe6ff !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
}

.stButton > button {
  border-radius: 999px;
  background: linear-gradient(135deg, var(--primary), var(--accent));
  border: none;
  padding: 0.6rem 1.1rem;
  font-weight: 600;
  color: #0b1224;
  box-shadow: 0 8px 18px rgba(34,211,238,0.25);
}
.stButton > button:hover {
  filter: brightness(1.05);
  transform: translateY(-1px);
}

/* Chat bubbles */
.chat-user, .chat-bot {
  border-radius: 16px;
  padding: 0.7rem 0.9rem;
  margin-bottom: 0.5rem;
  max-width: 100%;
  word-wrap: break-word;
  white-space: pre-wrap;
}
.chat-user {
  background: rgba(34,211,238,0.10);
  border: 1px solid rgba(34,211,238,0.25);
}
.chat-bot {
  background: rgba(59,130,246,0.10);
  border: 1px solid rgba(59,130,246,0.25);
}

/* Badges */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 0.78rem;
  color: #cfe3ff;
  background: rgba(103,167,255,0.12);
  border: 1px solid rgba(103,167,255,0.25);
  border-radius: 999px;
  padding: 0.25rem 0.55rem;
}

/* Responsive two-column layout */
@media (min-width: 1000px) {
  .grid-2 {
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 1rem;
  }
}
@media (max-width: 999px) {
  .grid-2 {
    display: block;
  }
}

/* Footer */
.footer {
  color: var(--muted);
  text-align: center;
  margin-top: 1rem;
  font-size: 0.8rem;
  opacity: 0.85;
}
"""

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# =============================================================================
# JavaScript para mobile (linhas 1366-1456)
# =============================================================================
MOBILE_JS = """
<script>
  // Smooth scroll to bottom on new messages (mobile convenience)
  const target = parent.document.querySelector('.block-container');
  if (target) { target.scrollTop = target.scrollHeight; }
</script>
"""
st.markdown(MOBILE_JS, unsafe_allow_html=True)


# =============================================================================
# Interface UI (linhas 1458-1464)
# =============================================================================
def header():
    left, right = st.columns([0.8, 0.2])
    with left:
        st.markdown("## 🧪 Origin Assistant — RAG + Claude")
        st.caption("Assistente com autenticação PBKDF2, RAG local com TF‑IDF e integração opcional com Claude.")
    with right:
        if current_user():
            st.markdown(f"<div class='badge'>🔐 Logado: <b>{current_user()}</b></div>", unsafe_allow_html=True)
            if st.button("Sair"):
                logout()
                st.rerun()


# =============================================================================
# Função server com todos os handlers (linhas 1466-2086)
# =============================================================================
def handle_upload_and_index():
    st.subheader("📚 Base de Conhecimento (RAG)")
    uploaded = st.file_uploader(
        "Envie arquivos .txt, .md ou .csv/.tsv (texto) para indexação (até ~5 MB cada)",
        type=["txt", "md", "csv", "tsv"],
        accept_multiple_files=True,
        key="uploader_docs",
    )
    if uploaded:
        add_uploaded_docs(uploaded)
        st.success(f"{len(uploaded)} arquivo(s) adicionados.")
    if st.button("Reindexar documentos"):
        rebuild_index_if_needed(force=True)
        st.info("Reindexação concluída.")
    # Show inventory
    ensure_index()
    if st.session_state["rag_docs"]:
        with st.expander("Ver documentos carregados"):
            for k, v in st.session_state["rag_docs"].items():
                st.markdown(f"- `{k}` — {len(v)} caracteres")
    else:
        st.caption("Nenhum documento na base ainda.")

def format_context(results: List[Tuple[float, Chunk]], top_k: int = 4) -> str:
    blocks = []
    for score, ch in results[:top_k]:
        blocks.append(f"[{ch.doc_id}#{ch.chunk_id} | score={score:.3f}]\n{ch.text}")
    return "\n\n".join(blocks)

def chat_section(claude: ClaudeClient):
    st.subheader("💬 Chat")
    prompt = st.text_area("Sua pergunta ou mensagem", height=120, key="chat_input")
    colA, colB, colC = st.columns([0.2, 0.2, 0.6])
    with colA:
        top_k = st.number_input("Top-K RAG", 1, 10, 4, key="topk")
    with colB:
        use_claude = st.toggle("Usar Claude", value=True if claude.available() else False, key="use_claude")
    with colC:
        temperature = st.slider("Temperatura (apenas Claude)", 0.0, 1.0, 0.3, 0.1)

    if "chat" not in st.session_state:
        st.session_state["chat"] = []  # list of dicts: {"role":"user"/"assistant","content": str}

    if st.button("Enviar", type="primary") and prompt.strip():
        # Append user msg
        st.session_state["chat"].append({"role": "user", "content": prompt})

        # Retrieve context
        ensure_index()
        results = st.session_state["rag_index"].search(prompt, k=top_k)
        context = format_context(results, top_k=top_k) if results else ""

        sys_hint = (
            "Você é um assistente especializado em materiais, polímeros e ciência. "
            "Responda de forma direta e cite trechos do contexto com [doc#chunk] quando útil."
        )
        composed = f"{sys_hint}\n\n## Contexto selecionado\n{context}\n\n## Pergunta\n{prompt}"
        reply = claude.complete(composed) if use_claude else "(Modo local) " + composed[:1200]

        st.session_state["chat"].append({"role": "assistant", "content": reply})

    # Chat history
    if st.session_state["chat"]:
        for msg in st.session_state["chat"][-50:]:  # limit render
            cls = "chat-user" if msg["role"] == "user" else "chat-bot"
            st.markdown(f"<div class='{cls}'>{msg['content']}</div>", unsafe_allow_html=True)

    # Utilities
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Limpar chat"):
            st.session_state["chat"] = []
            st.rerun()
    with c2:
        if st.button("Copiar última resposta"):
            if st.session_state["chat"]:
                last = next((m for m in reversed(st.session_state["chat"]) if m["role"]=="assistant"), None)
                if last:
                    st.code(last["content"])
                else:
                    st.info("Sem resposta para copiar.")
    with c3:
        if st.button("Exportar chat (.json)"):
            data = json.dumps(st.session_state["chat"], ensure_ascii=False, indent=2)
            b = io.BytesIO(data.encode("utf-8"))
            st.download_button("Baixar chat.json", b, file_name="chat_export.json", mime="application/json")

def auth_section():
    if not current_user():
        st.info("Entre com sua conta ou registre um usuário (exige admin token nos *Secrets*).")
        c1, c2 = st.columns(2)
        with c1:
            login_form()
        with c2:
            register_form()
        st.stop()

def server():
    # Prepare dependencies and state
    ensure_user_db()
    ensure_index()

    # Claude client wired via secrets
    api_key = read_secret(["anthropic", "api_key"], os.getenv("ANTHROPIC_API_KEY"))
    model = read_secret(["anthropic", "model"], "claude-3-5-sonnet-latest") or "claude-3-5-sonnet-latest"
    claude = ClaudeClient(api_key=api_key, model=model, max_tokens=1024)

    # UI
    header()
    auth_section()

    # Main grid
    st.markdown("<div class='grid-2'>", unsafe_allow_html=True)
    with st.container():
        handle_upload_and_index()
        st.markdown("---")
        chat_section(claude)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='footer'>Origin Assistant · {}</div>".format(now_iso()), unsafe_allow_html=True)


# =============================================================================
# Criação final do app (linha 2089)
# =============================================================================
if __name__ == "__main__":
    server()




