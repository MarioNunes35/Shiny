# app_with_auth.py - Origin Software Assistant com sistema de autenticação
# Compatível com Shiny 1.4.0 para Posit Connect

from shiny import App, ui, render, reactive, Inputs, Outputs, Session
from dotenv import load_dotenv
import os
import hashlib
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

# ---------------- Configuração inicial ----------------
load_dotenv()
API_KEY = os.getenv("ANTHROPIC_API_KEY")
HAS_KEY = bool(API_KEY)

# Paths para o sistema de autenticação
DATA_DIR = Path("data")
AUTH_DIR = DATA_DIR / "auth"
USER_DB_PATH = AUTH_DIR / "users.db"
CACHE_DIR = DATA_DIR / "cache"

# Criar diretórios necessários
for d in (DATA_DIR, AUTH_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------- Sistema de Autenticação ----------------

def hash_password(password: str) -> str:
    """Cria hash seguro da senha"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user_db():
    """Cria tabela de usuários se não existir"""
    with sqlite3.connect(USER_DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                created_at TEXT,
                last_login TEXT,
                active INTEGER DEFAULT 1,
                subscription_expires TEXT,
                is_admin INTEGER DEFAULT 0
            );
        """)
        
        # Criar usuário admin padrão se não existir
        cur.execute("SELECT id FROM users WHERE username = 'admin'")
        if not cur.fetchone():
            admin_pass = hash_password("admin123")
            cur.execute("""
                INSERT INTO users(username, password_hash, email, created_at, active, subscription_expires, is_admin)
                VALUES(?, ?, ?, ?, 1, ?, 1)
            """, ("admin", admin_pass, "admin@origin.com", 
                  datetime.utcnow().isoformat(),
                  (datetime.utcnow() + timedelta(days=365)).isoformat()))
            print("[AUTH] Usuário admin criado com senha padrão: admin123")
        
        con.commit()

def validate_user(username: str, password: str) -> Tuple[bool, str, bool]:
    """
    Valida credenciais do usuário
    Retorna: (sucesso, mensagem_erro, is_admin)
    """
    if not username or not password:
        return False, "Usuário e senha são obrigatórios", False
    
    with sqlite3.connect(USER_DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            SELECT password_hash, active, subscription_expires, is_admin
            FROM users WHERE username = ?
        """, (username,))
        result = cur.fetchone()
        
        if not result:
            return False, "Usuário ou senha incorretos", False
        
        password_hash, active, subscription_expires, is_admin = result
        
        # Verificar senha
        if hash_password(password) != password_hash:
            return False, "Usuário ou senha incorretos", False
        
        # Verificar se usuário está ativo
        if not active:
            return False, "Usuário desativado. Entre em contato com o suporte.", False
        
        # Verificar se assinatura não expirou
        if subscription_expires:
            expiry = datetime.fromisoformat(subscription_expires)
            if datetime.utcnow() > expiry:
                return False, "Assinatura expirada. Renove seu acesso.", False
        
        # Atualizar último login
        cur.execute(
            "UPDATE users SET last_login = ? WHERE username = ?",
            (datetime.utcnow().isoformat(), username)
        )
        con.commit()
        
        return True, "Login realizado com sucesso!", bool(is_admin)

def add_user(username: str, password: str, email: str = "", months: int = 12) -> Tuple[bool, str]:
    """Adiciona novo usuário (apenas admin)"""
    with sqlite3.connect(USER_DB_PATH) as con:
        cur = con.cursor()
        try:
            expiry = datetime.utcnow() + timedelta(days=30*months)
            cur.execute("""
                INSERT INTO users(username, password_hash, email, created_at, active, subscription_expires, is_admin)
                VALUES(?, ?, ?, ?, 1, ?, 0)
            """, (username, hash_password(password), email, 
                  datetime.utcnow().isoformat(), expiry.isoformat()))
            con.commit()
            return True, f"Usuário '{username}' criado com sucesso!"
        except sqlite3.IntegrityError:
            return False, "Este nome de usuário já existe"

def list_users():
    """Lista todos os usuários"""
    with sqlite3.connect(USER_DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            SELECT username, email, created_at, last_login, active, subscription_expires, is_admin
            FROM users ORDER BY created_at DESC
        """)
        return cur.fetchall()

# Criar banco de dados na inicialização
create_user_db()

# ---------------- Claude Integration ----------------

try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None

client = None
if HAS_KEY and Anthropic is not None:
    try:
        client = Anthropic(api_key=API_KEY)
    except Exception:
        client = None

# ---------------- RAG Configuration ----------------

RAG_FALLBACK = (os.getenv("RAG_FALLBACK", "auto") or "auto").lower()
RAG_MIN_TOPSCORE = float(os.getenv("RAG_MIN_TOPSCORE", "0.18"))
RAG_MIN_CTXCHARS = int(os.getenv("RAG_MIN_CTXCHARS", "300"))

# RAG Dependencies
HAVE_RAG_DEPS = True
try:
    from pypdf import PdfReader
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from joblib import dump, load
    import re
except Exception:
    HAVE_RAG_DEPS = False
    PdfReader = None
    TfidfVectorizer = None
    cosine_similarity = None
    dump = load = None

# S3/R2 Dependencies
HAVE_S3 = True
try:
    import boto3
    from botocore.exceptions import ClientError
except Exception:
    HAVE_S3 = False
    boto3 = None
    ClientError = Exception

print(f"[BOOT] RAG={HAVE_RAG_DEPS} | S3={HAVE_S3} | Claude={(client is not None)} | FB={RAG_FALLBACK}")

# Paths for RAG
CHUNKS_JSON = CACHE_DIR / "chunks.json"
VECTORIZER_JOBLIB = CACHE_DIR / "tfidf_vectorizer.joblib"
MATRIX_JOBLIB = CACHE_DIR / "tfidf_matrix.joblib"

# ---------------- S3/R2 Helpers ----------------
def _s3_conf():
    if not HAVE_S3:
        return None
    endpoint = os.getenv("S3_ENDPOINT_URL")
    bucket = os.getenv("S3_BUCKET")
    key = os.getenv("AWS_ACCESS_KEY_ID")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    prefix = os.getenv("S3_PREFIX", "osa-cache/")
    if prefix and not prefix.endswith('/'):
        prefix = prefix + '/'
    if not (endpoint and bucket and key and secret):
        return None
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=key,
            aws_secret_access_key=secret,
            region_name=os.getenv("AWS_DEFAULT_REGION"),
        )
        return {"client": s3, "bucket": bucket, "prefix": prefix}
    except Exception as e:
        print(f"[S3] erro ao criar cliente: {e}")
        return None

_S3 = _s3_conf()

def s3_pull_cache_if_needed():
    if not _S3:
        return False
    if CHUNKS_JSON.exists() and VECTORIZER_JOBLIB.exists() and MATRIX_JOBLIB.exists():
        return False
    client = _S3["client"]; bucket = _S3["bucket"]
    any_ok = False
    for key, local in [
        ("chunks.json", CHUNKS_JSON),
        ("tfidf_vectorizer.joblib", VECTORIZER_JOBLIB),
        ("tfidf_matrix.joblib", MATRIX_JOBLIB),
    ]:
        try:
            client.download_file(bucket, f"{_S3['prefix']}{key}", str(local))
            any_ok = True
        except Exception:
            pass
    return any_ok

def s3_push_cache():
    if not _S3:
        return False
    if not (CHUNKS_JSON.exists() and VECTORIZER_JOBLIB.exists() and MATRIX_JOBLIB.exists()):
        return False
    client = _S3["client"]; bucket = _S3["bucket"]
    ok_all = True
    for key, local in [
        ("chunks.json", CHUNKS_JSON),
        ("tfidf_vectorizer.joblib", VECTORIZER_JOBLIB),
        ("tfidf_matrix.joblib", MATRIX_JOBLIB),
    ]:
        try:
            client.upload_file(str(local), bucket, f"{_S3['prefix']}{key}")
        except Exception:
            ok_all = False
    return ok_all

# ---------------- RAG Functions ----------------
def extract_text(pdf_path: Path) -> str:
    if not HAVE_RAG_DEPS:
        return ""
    reader = PdfReader(str(pdf_path))
    txt = []
    for pg in reader.pages:
        txt.append(pg.extract_text() or "")
    return "\n".join(txt)

def chunk_text(text: str, max_chars=900, overlap=220):
    if not HAVE_RAG_DEPS:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    i = 0
    while i < len(text):
        j = min(i + max_chars, len(text))
        chunks.append(text[i:j])
        i = j - overlap if j - overlap > i else j
    return [c for c in chunks if c.strip()]

def load_index():
    if _S3:
        s3_pull_cache_if_needed()
    if CHUNKS_JSON.exists() and VECTORIZER_JOBLIB.exists() and MATRIX_JOBLIB.exists():
        chunks = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
        vectorizer = load(VECTORIZER_JOBLIB)
        matrix = load(MATRIX_JOBLIB)
        return chunks, vectorizer, matrix
    return [], None, None

def save_index(chunks, vectorizer, matrix):
    CHUNKS_JSON.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    dump(vectorizer, VECTORIZER_JOBLIB)
    dump(matrix, MATRIX_JOBLIB)
    if _S3:
        s3_push_cache()

def add_pdfs_to_index(file_paths: list):
    if not HAVE_RAG_DEPS:
        return 0
    chunks, _, _ = load_index()
    new_chunks = []
    for p in file_paths:
        try:
            doc_id = hashlib.sha256(p.read_bytes()).hexdigest()
            text = extract_text(p)
            for idx, ch in enumerate(chunk_text(text)):
                new_chunks.append({
                    "doc_id": doc_id,
                    "source": p.name,
                    "chunk_id": f"{doc_id[:8]}-{idx}",
                    "text": ch
                })
        except Exception:
            pass
    if not new_chunks:
        return len(chunks)
    all_chunks = chunks + new_chunks
    corpus = [c["text"] for c in all_chunks]
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=120_000)
    matrix = vectorizer.fit_transform(corpus)
    save_index(all_chunks, vectorizer, matrix)
    return len(all_chunks)

def retrieve(query: str, k=4):
    chunks, vectorizer, matrix = load_index()
    if not chunks or vectorizer is None:
        return [], 0.0
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    idx = sims.argsort()[::-1][:k]
    hits = [{
        "score": float(sims[i]),
        "text": chunks[i]["text"],
        "source": chunks[i]["source"],
        "chunk_id": chunks[i]["chunk_id"]
    } for i in idx]
    top = float(sims[idx[0]]) if len(idx) else 0.0
    return hits, top

def build_context(query: str):
    hits, top = retrieve(query, k=4)
    ctx = "\n\n".join(
        [f"[{i+1}] ({h['source']} • {h['chunk_id']} • score={h['score']:.3f})\n{h['text']}" for i, h in enumerate(hits)]
    )
    cites = "\n".join([f"- {h['source']} ({h['chunk_id']})" for h in hits])
    stats = {"top": top, "chars": len(ctx), "nhits": len(hits)}
    return ctx, cites, stats

def rag_should_fallback(stats: dict) -> bool:
    if RAG_FALLBACK == "off":
        return False
    return (stats.get("top", 0.0) < RAG_MIN_TOPSCORE) or (stats.get("chars", 0) < RAG_MIN_CTXCHARS)

def anthropic_messages_from_history(history):
    msgs = []
    for m in history:
        if m["role"] in ("user","assistant"):
            msgs.append({"role": m["role"], "content":[{"type":"text","text": m["content"]}]})
    return msgs

def _extract_text_from_resp(resp):
    parts = []
    try:
        for block in getattr(resp, "content", []):
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
    except Exception:
        parts = [str(resp)]
    return "\n".join(parts) if parts else str(resp)

def chat_reply_with_context(history, model):
    if client is None:
        return "Claude indisponível. Configure ANTHROPIC_API_KEY e o pacote 'anthropic'."

    question = next((m["content"] for m in reversed(history) if m["role"]=="user"), "")

    # RAG com contexto
    if HAVE_RAG_DEPS:
        ctx, cites, stats = build_context(question)
    else:
        ctx, cites, stats = ("", "", {"top": 0.0, "chars": 0, "nhits": 0})

    use_rag = bool(ctx) and not rag_should_fallback(stats)

    if use_rag:
        system = (
            "Você é o Origin Software Assistant. Use apenas o CONTEXTO abaixo; "
            "se algo não estiver no contexto, responda somente com o que está suportado.\n\n"
            f"=== CONTEXTO ===\n{ctx}\n=== FIM DO CONTEXTO ==="
        )
        resp = client.messages.create(
            model=model, max_tokens=900, temperature=0.2,
            system=system, messages=anthropic_messages_from_history(history)
        )
        answer = _extract_text_from_resp(resp)
        if cites:
            answer += "\n\n---\n**Fontes:**\n" + cites
        return answer

    # Fallback (sem contexto)
    system = "Você é o Origin Software Assistant. Responda com clareza e objetividade."
    resp = client.messages.create(
        model=model, max_tokens=900, temperature=0.2,
        system=system, messages=anthropic_messages_from_history(history)
    )
    answer = _extract_text_from_resp(resp)
    answer += "\n\n_(Respondi por conhecimento geral; PDFs não tinham informação suficiente.)_"
    return answer

# Try to restore cache on startup
try:
    s3_pull_cache_if_needed()
except Exception:
    pass

# ---------------- CSS Customizado ----------------

FULL_CSS = """
:root{
  --bg:#F7F7F8; --panel:#FFFFFF;
  --bubble-user:#E5F2FF; --bubble-assistant:#F7F7F8;
  --border:#E2E2E3; --text:#0F172A; --muted:#6B7280;
  --accent:#10A37F; --accent-hover:#0E8B6F;
  --error:#EF4444; --success:#10B981;
  --shadow: 0 1px 3px rgba(0,0,0,0.08);
}
[data-theme='dark']{
  --bg:#202123; --panel:#2D2E30;
  --bubble-user:#343642; --bubble-assistant:#444654;
  --border:#444654; --text:#ECECF1; --muted:#9CA3AF;
  --accent:#19C37D; --accent-hover:#15A366;
  --shadow: 0 2px 6px rgba(0,0,0,0.3);
}

* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { height: 100%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
body { background: var(--bg); color: var(--text); }

/* Login Page */
.login-container {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.login-card {
  background: var(--panel);
  border-radius: 20px;
  box-shadow: 0 20px 60px rgba(0,0,0,0.3);
  padding: 40px;
  width: 100%;
  max-width: 400px;
  margin: 20px;
}

.login-header {
  text-align: center;
  margin-bottom: 30px;
}

.login-logo {
  font-size: 48px;
  margin-bottom: 16px;
}

.login-title {
  font-size: 24px;
  font-weight: 700;
  margin-bottom: 8px;
  background: linear-gradient(135deg, var(--accent), var(--accent-hover));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.login-subtitle {
  color: var(--muted);
  font-size: 14px;
}

/* Main App */
.app-header {
  background: var(--panel);
  border-bottom: 1px solid var(--border);
  padding: 16px 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: var(--shadow);
}

.app-header h1 {
  font-size: 20px;
  margin: 0;
}

/* Form Elements */
.form-group {
  margin-bottom: 20px;
}

.form-label {
  display: block;
  margin-bottom: 8px;
  font-size: 14px;
  font-weight: 500;
  color: var(--text);
}

input[type="text"],
input[type="password"],
input[type="email"],
input[type="number"],
select,
textarea {
  width: 100%;
  padding: 12px 16px;
  border: 2px solid var(--border);
  border-radius: 10px;
  font-size: 15px;
  background: var(--bg);
  color: var(--text);
  transition: all 0.2s;
}

input:focus,
select:focus,
textarea:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(25,195,125,0.1);
}

/* Buttons */
.btn {
  padding: 12px 24px;
  border-radius: 10px;
  border: none;
  font-weight: 600;
  font-size: 15px;
  cursor: pointer;
  transition: all 0.2s;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  text-align: center;
}

.btn-primary {
  background: linear-gradient(135deg, var(--accent), var(--accent-hover));
  color: white;
  width: 100%;
}

.btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(25,195,125,0.3);
}

.btn-secondary {
  background: var(--bg);
  color: var(--text);
  border: 2px solid var(--border);
  width: 100%;
}

.btn-secondary:hover {
  background: var(--panel);
  border-color: var(--accent);
}

.btn-logout {
  background: rgba(239,68,68,0.1);
  color: var(--error);
  border: 1px solid var(--error);
  padding: 8px 16px;
  width: auto;
}

/* Messages */
.alert {
  padding: 12px 16px;
  border-radius: 8px;
  margin-bottom: 20px;
  font-size: 14px;
}

.alert-error {
  background: rgba(239,68,68,0.1);
  color: var(--error);
  border: 1px solid rgba(239,68,68,0.2);
}

.alert-success {
  background: rgba(16,185,129,0.1);
  color: var(--success);
  border: 1px solid rgba(16,185,129,0.2);
}

.alert-info {
  background: rgba(59,130,246,0.1);
  color: #3b82f6;
  border: 1px solid rgba(59,130,246,0.2);
}

/* Admin Panel */
.admin-panel {
  background: var(--panel);
  border-radius: 12px;
  padding: 24px;
  margin: 20px;
  box-shadow: var(--shadow);
}

.admin-header {
  margin-bottom: 20px;
  padding-bottom: 16px;
  border-bottom: 2px solid var(--border);
}

.user-card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 12px;
}

.user-info {
  margin-bottom: 8px;
}

.user-name {
  font-weight: 600;
  font-size: 16px;
  margin-bottom: 4px;
}

.user-meta {
  font-size: 13px;
  color: var(--muted);
}

.status-badge {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 500;
}

.status-active {
  background: rgba(16,185,129,0.1);
  color: var(--success);
}

.status-inactive {
  background: rgba(239,68,68,0.1);
  color: var(--error);
}

/* Content Area */
.content-area {
  padding: 20px;
  background: var(--bg);
  min-height: calc(100vh - 80px);
}

/* Divider */
.divider {
  height: 1px;
  background: var(--border);
  margin: 20px 0;
}
"""

# ---------------- Interface do App ----------------

app_ui = ui.page_fluid(
    ui.tags.style(FULL_CSS),
    ui.tags.script("""
        // Theme handler
        document.documentElement.setAttribute('data-theme', 'dark');
    """),
    
    # Container principal com condicional
    ui.output_ui("main_content")
)

def server(input: Inputs, output: Outputs, session: Session):
    # Estado da sessão
    authenticated = reactive.Value(False)
    current_user = reactive.Value("")
    is_admin = reactive.Value(False)
    login_message = reactive.Value("")
    demo_shown = reactive.Value(False)
    
    # Estado do chat
    history = reactive.Value([])
    typing = reactive.Value(False)
    
    def push(role, content):
        history.set(history() + [{"role": role, "content": content}])
    
    @output
    @render.ui
    def main_content():
        """Renderiza login ou app principal baseado no estado de autenticação"""
        if not authenticated():
            # TELA DE LOGIN
            return ui.div({"class": "login-container"},
                ui.div({"class": "login-card"},
                    ui.div({"class": "login-header"},
                        ui.div({"class": "login-logo"}, "🔐"),
                        ui.h1({"class": "login-title"}, "Origin Software Assistant"),
                        ui.p({"class": "login-subtitle"}, "Entre com suas credenciais para acessar")
                    ),
                    
                    # Mensagem de feedback
                    ui.output_ui("login_feedback"),
                    
                    # Formulário
                    ui.div({"class": "form-group"},
                        ui.span({"class": "form-label"}, "👤 Usuário"),
                        ui.input_text("username", None, 
                            placeholder="Digite seu usuário",
                            width="100%"
                        )
                    ),
                    
                    ui.div({"class": "form-group"},
                        ui.span({"class": "form-label"}, "🔑 Senha"),
                        ui.input_password("password", None,
                            placeholder="Digite sua senha",
                            width="100%"
                        )
                    ),
                    
                    ui.input_action_button("login_btn", "🚀 Entrar",
                        class_="btn btn-primary"
                    ),
                    
                    ui.div({"class": "divider"}),
                    
                    ui.input_action_button("demo_btn", "👁️ Ver credenciais demo",
                        class_="btn btn-secondary"
                    ),
                    
                    # Info demo
                    ui.output_ui("demo_info")
                )
            )
        else:
            # APP PRINCIPAL (após login)
            return ui.TagList(
                # Header
                ui.div({"class": "app-header"},
                    ui.div(
                        ui.h1("🚀 Origin Software Assistant"),
                        ui.p({"style": "margin: 0; color: var(--muted); font-size: 14px;"}, 
                            f"Bem-vindo, {current_user()}{'  (Admin)' if is_admin() else ''}")
                    ),
                    ui.input_action_button("logout_btn", "🚪 Logout",
                        class_="btn btn-logout"
                    )
                ),
                
                # Admin Panel (condicional)
                ui.output_ui("admin_panel"),
                
                # Content Area - Chat RAG Integrado
                ui.div({"class": "content-area", "style": "padding: 0; display: flex; flex-direction: column; height: calc(100vh - 80px);"},
                    
                    # Knowledge Base Section
                    ui.div({"class": "kb-section", "style": "background: var(--panel); padding: 16px 24px; border-bottom: 1px solid var(--border);"},
                        ui.div({"class": "kb-card", "style": "background: var(--bg); border: 1px solid var(--border); border-radius: 12px; padding: 16px;"},
                            ui.h3("📚 Base de Conhecimento", {"style": "font-size: 16px; font-weight: 600; margin-bottom: 12px;"}),
                            ui.input_file("docs", "Adicionar PDFs", 
                                multiple=True, 
                                accept=[".pdf"],
                                width="auto"
                            ),
                            ui.div({"class": "kb-info", "style": "font-size: 13px; color: var(--muted); margin-top: 8px; padding: 8px; background: var(--panel); border-radius: 6px;"},
                                ui.output_text("kb_status")
                            )
                        )
                    ),
                    
                    # Chat Area
                    ui.div({"class": "chat-wrapper", "style": "flex: 1; overflow-y: auto; padding: 24px; background: var(--bg);"},
                        ui.div({"class": "chat-container", "style": "max-width: 900px; margin: 0 auto;"},
                            ui.output_ui("chat_thread")
                        )
                    ),
                    
                    # Composer
                    ui.div({"class": "composer-wrapper", "style": "background: var(--panel); border-top: 1px solid var(--border); padding: 20px 24px; box-shadow: 0 -2px 10px rgba(0,0,0,0.1);"},
                        ui.div({"class": "composer", "style": "max-width: 900px; margin: 0 auto; display: flex; gap: 16px; align-items: flex-end;"},
                            ui.div({"class": "input-group", "style": "flex: 1;"},
                                ui.input_text_area("prompt", None, 
                                    placeholder="Digite sua mensagem... (Shift+Enter para nova linha)",
                                    rows=3,
                                    width="100%"
                                ),
                                ui.div({"class": "controls-row", "style": "display: flex; gap: 12px; margin-top: 12px; align-items: center;"},
                                    ui.input_select("model", None, 
                                        {
                                            "claude-3-haiku-20240307": "⚡ Claude 3 Haiku (rápido)",
                                            "claude-3-5-sonnet-20240620": "✨ Claude 3.5 Sonnet (avançado)"
                                        }, 
                                        selected="claude-3-haiku-20240307",
                                        width="auto"
                                    ),
                                    ui.input_action_button("clear", "🗑️ Limpar", 
                                        class_="btn btn-secondary",
                                        style="width: auto; padding: 8px 16px;"
                                    )
                                )
                            ),
                            ui.input_action_button("send", "Enviar →", 
                                class_="btn btn-primary",
                                style="min-width: 100px; height: 48px;"
                            )
                        )
                    )
                )
            )
    
    @output
    @render.ui
    def login_feedback():
        """Mostra mensagens de feedback do login"""
        if login_message():
            msg = login_message()
            if "sucesso" in msg.lower():
                return ui.div({"class": "alert alert-success"}, f"✅ {msg}")
            else:
                return ui.div({"class": "alert alert-error"}, f"❌ {msg}")
        return ui.TagList()
    
    @output
    @render.ui
    def demo_info():
        """Mostra informações da conta demo"""
        if demo_shown():
            return ui.div({"class": "alert alert-info"},
                ui.strong("Conta Demo:"),
                ui.br(),
                "Usuário: ", ui.code("admin"),
                ui.br(),
                "Senha: ", ui.code("admin123")
            )
        return ui.TagList()
    
    @output
    @render.ui
    def admin_panel():
        """Painel administrativo para admins"""
        if not is_admin():
            return ui.TagList()
        
        users = list_users()
        
        return ui.div({"class": "admin-panel"},
            ui.div({"class": "admin-header"},
                ui.h2("🛠️ Painel Administrativo"),
                ui.p({"style": "margin: 0; color: var(--muted);"}, 
                    f"{len(users)} usuários cadastrados")
            ),
            
            # Adicionar usuário
            ui.div({"style": "margin-bottom: 24px;"},
                ui.h3("➕ Adicionar Novo Usuário"),
                ui.row(
                    ui.column(6,
                        ui.input_text("new_username", "Usuário", 
                            placeholder="Nome de usuário")
                    ),
                    ui.column(6,
                        ui.input_password("new_password", "Senha",
                            placeholder="Senha segura")
                    )
                ),
                ui.row(
                    ui.column(6,
                        ui.input_text("new_email", "Email (opcional)",
                            placeholder="email@exemplo.com")
                    ),
                    ui.column(6,
                        ui.input_numeric("new_months", "Meses de acesso",
                            value=12, min=1, max=36)
                    )
                ),
                ui.br(),
                ui.input_action_button("add_user_btn", "Criar Usuário",
                    class_="btn btn-primary",
                    width="200px"
                ),
                ui.output_text("add_user_feedback")
            ),
            
            ui.div({"class": "divider"}),
            
            # Lista de usuários
            ui.h3("👥 Usuários Cadastrados"),
            ui.TagList(*[
                ui.div({"class": "user-card"},
                    ui.div({"class": "user-info"},
                        ui.div({"class": "user-name"}, 
                            f"{'👑 ' if user[6] else ''}{user[0]}"
                        ),
                        ui.div({"class": "user-meta"},
                            f"Email: {user[1] or 'N/A'} | ",
                            f"Criado: {user[2][:10] if user[2] else 'N/A'} | ",
                            f"Último login: {user[3][:10] if user[3] else 'Nunca'}"
                        )
                    ),
                    ui.span({"class": f"status-badge {'status-active' if user[4] else 'status-inactive'}"},
                        "Ativo" if user[4] else "Inativo"
                    )
                ) for user in users
            ])
        )
    
    # Chat Outputs
    @output
    @render.text
    def kb_status():
        """Status da base de conhecimento"""
        if not HAVE_RAG_DEPS:
            return "⚠️ RAG indisponível: instale pypdf, scikit-learn e joblib"
        chunks, _, _ = load_index()
        n_docs = len({c["source"] for c in chunks}) if chunks else 0
        n_chunks = len(chunks)
        
        status_parts = [f"📄 {n_docs} documento(s)", f"🧩 {n_chunks} fragmento(s)"]
        
        if _S3:
            status_parts.append("☁️ Sincronização S3 ativa")
        
        if RAG_FALLBACK != "off":
            status_parts.append(f"🔄 Fallback: {RAG_MIN_TOPSCORE:.2f}/{RAG_MIN_CTXCHARS}")
            
        return " • ".join(status_parts)
    
    @output
    @render.ui
    def chat_thread():
        """Renderiza thread do chat"""
        items = []
        for m in history():
            is_assistant = m["role"] == "assistant"
            
            avatar_style = "width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 14px; flex-shrink: 0;"
            if is_assistant:
                avatar_style += "background: linear-gradient(135deg, var(--accent), var(--accent-hover)); color: white;"
                avatar_text = "OA"
                role_text = "Origin Assistant"
            else:
                avatar_style += "background: linear-gradient(135deg, #667EEA, #764BA2); color: white;"
                avatar_text = "V"
                role_text = "Você"
            
            message_style = "display: flex; gap: 16px; margin-bottom: 24px;"
            if not is_assistant:
                message_style += "flex-direction: row-reverse;"
            
            bubble_style = "background: var(--bubble-assistant); border: 1px solid var(--border); border-radius: 18px; padding: 12px 16px; box-shadow: var(--shadow);"
            if not is_assistant:
                bubble_style = "background: var(--bubble-user); border: 1px solid var(--border); border-radius: 18px; padding: 12px 16px; box-shadow: var(--shadow);"
            
            items.append(
                ui.div({"style": message_style},
                    ui.div({"style": avatar_style}, avatar_text),
                    ui.div({"style": "flex: 1; max-width: 75%;"},
                        ui.div({"style": f"font-size: 12px; font-weight: 600; color: var(--muted); margin-bottom: 4px; {'text-align: right;' if not is_assistant else ''}"},
                            role_text
                        ),
                        ui.div({"style": bubble_style},
                            ui.div({"style": "font-size: 15px; line-height: 1.6; color: var(--text);"},
                                ui.markdown(m["content"])
                            )
                        )
                    )
                )
            )
        
        # Typing indicator
        if typing():
            items.append(
                ui.div({"style": "display: flex; gap: 16px; margin-bottom: 24px;"},
                    ui.div({"style": "width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 14px; background: linear-gradient(135deg, var(--accent), var(--accent-hover)); color: white;"}, "OA"),
                    ui.div({"style": "flex: 1; max-width: 75%;"},
                        ui.div({"style": "font-size: 12px; font-weight: 600; color: var(--muted); margin-bottom: 4px;"},
                            "Origin Assistant"
                        ),
                        ui.div({"style": "background: var(--bubble-assistant); border: 1px solid var(--border); border-radius: 18px; padding: 12px 16px;"},
                            ui.div({"style": "display: flex; gap: 4px;"},
                                ui.div({"style": "width: 8px; height: 8px; border-radius: 50%; background: var(--muted); animation: typing 1.4s infinite;"}),
                                ui.div({"style": "width: 8px; height: 8px; border-radius: 50%; background: var(--muted); animation: typing 1.4s infinite; animation-delay: 0.2s;"}),
                                ui.div({"style": "width: 8px; height: 8px; border-radius: 50%; background: var(--muted); animation: typing 1.4s infinite; animation-delay: 0.4s;"})
                            )
                        )
                    )
                )
            )
        
        return ui.TagList(*items) if items else ui.div(
            {"style": "text-align: center; color: var(--muted); padding: 40px;"},
            "💬 Inicie uma conversa enviando uma mensagem"
        )
    
    # Event Handlers - Login
    @reactive.Effect
    @reactive.event(input.demo_btn)
    def show_demo():
        """Mostra credenciais demo"""
        demo_shown.set(True)
    
    @reactive.Effect
    @reactive.event(input.login_btn)
    def handle_login():
        """Processa login"""
        username = input.username()
        password = input.password()
        
        success, message, admin = validate_user(username, password)
        login_message.set(message)
        
        if success:
            authenticated.set(True)
            current_user.set(username)
            is_admin.set(admin)
            ui.notification_show(f"Bem-vindo, {username}!", type="message", duration=3)
    
    @reactive.Effect
    @reactive.event(input.logout_btn)
    def handle_logout():
        """Processa logout"""
        authenticated.set(False)
        current_user.set("")
        is_admin.set(False)
        login_message.set("")
        demo_shown.set(False)
        history.set([])  # Limpar histórico do chat
        ui.notification_show("Logout realizado com sucesso", type="message", duration=2)
    
    # Event Handlers - Admin
    @output
    @render.text
    def add_user_feedback():
        """Feedback da criação de usuário"""
        return ""
    
    @reactive.Effect
    @reactive.event(input.add_user_btn)
    def handle_add_user():
        """Adiciona novo usuário"""
        if not is_admin():
            return
        
        username = input.new_username()
        password = input.new_password()
        email = input.new_email()
        months = input.new_months()
        
        if not username or not password:
            ui.notification_show("Usuário e senha são obrigatórios", type="warning", duration=3)
            return
        
        success, message = add_user(username, password, email, months)
        
        if success:
            ui.notification_show(message, type="message", duration=3)
            # Limpar campos
            ui.update_text("new_username", value="")
            ui.update_password("new_password", value="")
            ui.update_text("new_email", value="")
        else:
            ui.notification_show(message, type="warning", duration=3)
    
    # Event Handlers - Chat
    @reactive.Effect
    @reactive.event(input.clear)
    def _clear():
        """Limpa o chat"""
        history.set([])
        ui.update_text_area("prompt", value="")
        ui.notification_show("Chat limpo com sucesso", type="message", duration=2)
    
    @reactive.Effect
    @reactive.event(input.docs)
    def _ingest_pdfs():
        """Processa PDFs enviados"""
        if not HAVE_RAG_DEPS:
            ui.notification_show(
                "⚠️ Instale pypdf, scikit-learn e joblib para ativar o RAG", 
                type="error",
                duration=5
            )
            return
        
        files = input.docs() or []
        if not files:
            return
        
        paths = []
        for f in files:
            src = Path(f["datapath"])
            dst = DATA_DIR / f["name"]
            dst.write_bytes(src.read_bytes())
            paths.append(dst)
        
        total = add_pdfs_to_index(paths)
        
        if _S3:
            ok = s3_push_cache()
            if ok:
                ui.notification_show(
                    f"✅ {len(files)} PDF(s) processados • Total: {total} fragmentos • Sincronizado com S3",
                    type="success",
                    duration=4
                )
            else:
                ui.notification_show(
                    f"✅ {len(files)} PDF(s) processados • Total: {total} fragmentos",
                    type="success",
                    duration=4
                )
        else:
            ui.notification_show(
                f"✅ {len(files)} PDF(s) processados • Total: {total} fragmentos",
                type="success",
                duration=4
            )
    
    @reactive.Effect
    @reactive.event(input.send)
    def _send():
        """Envia mensagem no chat"""
        if not authenticated():
            return
            
        q = (input.prompt() or "").strip()
        if not q:
            ui.notification_show("⚠️ Digite uma mensagem", type="warning", duration=2)
            return
        
        push("user", q)
        ui.update_text_area("prompt", value="")
        
        if client is None:
            push("assistant", 
                "⚠️ Claude indisponível. Configure ANTHROPIC_API_KEY e instale o pacote 'anthropic'."
            )
            return
        
        typing.set(True)
        model = input.model() or "claude-3-haiku-20240307"
        
        try:
            reply = chat_reply_with_context(history(), model)
        except Exception as e:
            reply = f"❌ Erro ao processar: {str(e)}"
        finally:
            typing.set(False)
        
        push("assistant", reply)

app = App(app_ui, server)







