# app_mobile_responsive.py - Origin Software Assistant com design responsivo
# Interface mobile-first inspirada no Claude Code UI

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

# Paths para o sistema - USANDO CAMINHO PERSISTENTE
if os.path.exists("/home/shiny"):
    BASE_DIR = Path("/home/shiny/.origin_assistant")
else:
    BASE_DIR = Path.home() / ".origin_assistant"

DATA_DIR = BASE_DIR / "data"
AUTH_DIR = DATA_DIR / "auth"
USER_DB_PATH = AUTH_DIR / "users.db"
CACHE_DIR = DATA_DIR / "cache"

# Criar diretórios necessários
for d in (BASE_DIR, DATA_DIR, AUTH_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

print(f"[BOOT] Usando diretório base: {BASE_DIR}")
print(f"[BOOT] Banco de dados em: {USER_DB_PATH}")

# ---------------- Sistema de Autenticação ----------------

def hash_password(password: str) -> str:
    """Cria hash seguro da senha"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user_db():
    """Cria tabela de usuários se não existir"""
    try:
        con = sqlite3.connect(str(USER_DB_PATH))
        cur = con.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
        
        cur.execute("SELECT id FROM users WHERE username = ?", ("admin",))
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
        con.close()
        print(f"[AUTH] Banco de dados inicializado em {USER_DB_PATH}")
        
    except Exception as e:
        print(f"[AUTH ERROR] Erro ao criar banco: {e}")

def validate_user(username: str, password: str) -> Tuple[bool, str, bool]:
    """Valida credenciais do usuário"""
    if not username or not password:
        return False, "Usuário e senha são obrigatórios", False
    
    try:
        con = sqlite3.connect(str(USER_DB_PATH))
        cur = con.cursor()
        
        cur.execute("""
            SELECT password_hash, active, subscription_expires, is_admin
            FROM users WHERE username = ?
        """, (username,))
        result = cur.fetchone()
        
        if not result:
            con.close()
            return False, "Usuário ou senha incorretos", False
        
        password_hash, active, subscription_expires, is_admin = result
        
        if hash_password(password) != password_hash:
            con.close()
            return False, "Usuário ou senha incorretos", False
        
        if not active:
            con.close()
            return False, "Usuário desativado. Entre em contato com o suporte.", False
        
        if subscription_expires:
            expiry = datetime.fromisoformat(subscription_expires)
            if datetime.utcnow() > expiry:
                con.close()
                return False, "Assinatura expirada. Renove seu acesso.", False
        
        cur.execute(
            "UPDATE users SET last_login = ? WHERE username = ?",
            (datetime.utcnow().isoformat(), username)
        )
        con.commit()
        con.close()
        
        return True, "Login realizado com sucesso!", bool(is_admin)
        
    except Exception as e:
        print(f"[AUTH ERROR] Erro ao validar usuário: {e}")
        return False, "Erro ao processar login", False

def add_user(username: str, password: str, email: str = "", months: int = 12) -> Tuple[bool, str]:
    """Adiciona novo usuário"""
    try:
        con = sqlite3.connect(str(USER_DB_PATH))
        cur = con.cursor()
        
        expiry = datetime.utcnow() + timedelta(days=30*months)
        cur.execute("""
            INSERT INTO users(username, password_hash, email, created_at, active, subscription_expires, is_admin)
            VALUES(?, ?, ?, ?, 1, ?, 0)
        """, (username, hash_password(password), email, 
              datetime.utcnow().isoformat(), expiry.isoformat()))
        
        con.commit()
        con.close()
        return True, f"Usuário '{username}' criado com sucesso!"
        
    except sqlite3.IntegrityError:
        return False, "Este nome de usuário já existe"
    except Exception as e:
        print(f"[AUTH ERROR] Erro ao adicionar usuário: {e}")
        return False, f"Erro ao criar usuário: {str(e)}"

def list_users():
    """Lista todos os usuários"""
    try:
        con = sqlite3.connect(str(USER_DB_PATH))
        cur = con.cursor()
        cur.execute("""
            SELECT username, email, created_at, last_login, active, subscription_expires, is_admin
            FROM users ORDER BY created_at DESC
        """)
        users = cur.fetchall()
        con.close()
        return users
    except Exception as e:
        print(f"[AUTH ERROR] Erro ao listar usuários: {e}")
        return []

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

print(f"[BOOT] RAG={HAVE_RAG_DEPS} | Claude={(client is not None)}")

CHUNKS_JSON = CACHE_DIR / "chunks.json"
VECTORIZER_JOBLIB = CACHE_DIR / "tfidf_vectorizer.joblib"
MATRIX_JOBLIB = CACHE_DIR / "tfidf_matrix.joblib"

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

    if HAVE_RAG_DEPS:
        ctx, cites, stats = build_context(question)
    else:
        ctx, cites, stats = ("", "", {"top": 0.0, "chars": 0, "nhits": 0})

    use_rag = bool(ctx) and not rag_should_fallback(stats)

    if use_rag:
        system = (
            "Você é o Origin Software Assistant, um especialista EXCLUSIVO no software OriginPro para análise de dados e criação de gráficos científicos. "
            "IMPORTANTE: Você APENAS responde sobre o OriginPro. Todas as suas respostas devem ser no contexto do OriginPro.\n\n"
            "Regras críticas:\n"
            "- Se perguntarem sobre plotagem, gráficos, análise de dados, ou qualquer funcionalidade técnica, responda SEMPRE no contexto do OriginPro\n"
            "- Use termos específicos do OriginPro como: worksheet, workbook, graph window, layer, plot types, analysis tools\n"
            "- Mencione menus e ferramentas específicas do Origin quando relevante\n"
            "- Se a pergunta não for sobre o OriginPro, educadamente redirecione para o OriginPro\n\n"
            f"=== CONTEXTO DA DOCUMENTAÇÃO ===\n{ctx}\n=== FIM DO CONTEXTO ==="
        )
    else:
        system = (
            "Você é o Origin Software Assistant, um especialista EXCLUSIVO no software OriginPro para análise de dados e criação de gráficos científicos.\n\n"
            "IMPORTANTE: Você APENAS responde sobre o OriginPro. Todas as suas respostas devem ser no contexto do OriginPro.\n\n"
            "Regras críticas:\n"
            "- Se perguntarem sobre plotagem, gráficos, análise de dados, responda SEMPRE explicando como fazer no OriginPro\n"
            "- Use termos específicos do OriginPro: worksheet, workbook, graph window, layer, plot types (Line, Scatter, Column, etc.)\n"
            "- Mencione menus do Origin: Plot menu, Analysis menu, Statistics menu, etc.\n"
            "- Para plotagem de gráficos, sempre explique o processo no Origin:\n"
            "  1. Importar dados no worksheet\n"
            "  2. Selecionar colunas relevantes\n"
            "  3. Escolher tipo de gráfico no menu Plot\n"
            "  4. Customizar usando as ferramentas do Origin\n"
            "- Se a pergunta não for relacionada ao OriginPro, educadamente informe que você é especializado apenas no OriginPro"
        )

    resp = client.messages.create(
        model=model, 
        max_tokens=900, 
        temperature=0.2,
        system=system, 
        messages=anthropic_messages_from_history(history)
    )
    
    answer = _extract_text_from_resp(resp)
    
    if use_rag and cites:
        answer += "\n\n---\n**Fontes:**\n" + cites
    elif not use_rag:
        answer += "\n\n_(Resposta baseada no conhecimento geral do OriginPro. Adicione documentação PDF para respostas mais específicas.)_"
    
    return answer

# ---------------- CSS Mobile Responsive ----------------

MOBILE_RESPONSIVE_CSS = """
/* Reset e Base */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body {
    height: 100%;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0e0e0e;
    color: #e0e0e0;
    line-height: 1.6;
    overflow-x: hidden;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-track {
    background: #1a1a1a;
}

::-webkit-scrollbar-thumb {
    background: #3a3a3a;
    border-radius: 3px;
}

/* Viewport Meta Detection */
@media (max-width: 768px) {
    html {
        font-size: 14px;
    }
}

/* Login Page - Mobile First */
.login-container {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #0e0e0e;
    padding: 20px;
}

.login-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 24px;
    width: 100%;
    max-width: 400px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.8);
}

@media (max-width: 480px) {
    .login-card {
        padding: 20px;
        margin: 10px;
    }
}

.login-header {
    text-align: center;
    margin-bottom: 24px;
}

.login-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 8px;
    color: #fff;
}

@media (max-width: 480px) {
    .login-title {
        font-size: 18px;
    }
}

.login-subtitle {
    color: #888;
    font-size: 13px;
}

/* Mobile Navigation */
.mobile-header {
    display: none;
    background: #141414;
    border-bottom: 1px solid #2a2a2a;
    padding: 12px 16px;
    position: sticky;
    top: 0;
    z-index: 100;
    justify-content: space-between;
    align-items: center;
}

@media (max-width: 768px) {
    .mobile-header {
        display: flex;
    }
}

.mobile-title {
    font-size: 16px;
    font-weight: 600;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 8px;
}

.mobile-menu-btn {
    background: transparent;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    color: #e0e0e0;
    padding: 8px 12px;
    font-size: 12px;
    cursor: pointer;
}

/* Sidebar - Mobile Responsive */
.sidebar {
    width: 260px;
    background: #141414;
    border-right: 1px solid #2a2a2a;
    display: flex;
    flex-direction: column;
    height: 100vh;
    position: fixed;
    left: 0;
    top: 0;
    z-index: 200;
    transition: transform 0.3s ease;
}

@media (max-width: 768px) {
    .sidebar {
        transform: translateX(-100%);
        width: 280px;
        box-shadow: 2px 0 10px rgba(0,0,0,0.5);
    }
    
    .sidebar.open {
        transform: translateX(0);
    }
}

.sidebar-overlay {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.6);
    z-index: 150;
}

@media (max-width: 768px) {
    .sidebar-overlay.show {
        display: block;
    }
}

.sidebar-header {
    padding: 16px;
    border-bottom: 1px solid #2a2a2a;
}

.sidebar-title {
    font-size: 16px;
    font-weight: 600;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 8px;
}

.sidebar-user {
    font-size: 12px;
    color: #666;
    margin-top: 4px;
}

.sessions-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
}

.session-item {
    padding: 12px;
    margin-bottom: 4px;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    color: #888;
    cursor: pointer;
    font-size: 14px;
    transition: all 0.2s;
    word-wrap: break-word;
}

.session-item:hover {
    background: #1a1a1a;
    color: #e0e0e0;
}

.session-item.active {
    background: #1e1e1e;
    border-color: #3a3a3a;
    color: #fff;
}

.new-session-btn {
    margin: 8px;
    padding: 12px;
    background: #1e1e1e;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    color: #e0e0e0;
    cursor: pointer;
    text-align: center;
    transition: all 0.2s;
    font-size: 14px;
}

.new-session-btn:hover {
    background: #2a2a2a;
    border-color: #4a4a4a;
}

/* Main Content - Mobile Responsive */
.main-content {
    margin-left: 260px;
    height: 100vh;
    display: flex;
    flex-direction: column;
    background: #0e0e0e;
}

@media (max-width: 768px) {
    .main-content {
        margin-left: 0;
        padding-top: 0;
    }
}

.header-bar {
    background: #141414;
    border-bottom: 1px solid #2a2a2a;
    padding: 12px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

@media (max-width: 768px) {
    .header-bar {
        display: none;
    }
}

.header-title {
    font-size: 14px;
    color: #888;
    display: flex;
    align-items: center;
    gap: 8px;
}

.header-actions {
    display: flex;
    gap: 8px;
}

/* Knowledge Base - Mobile Responsive */
.kb-section {
    padding: 12px 16px;
    background: #141414;
    border-bottom: 1px solid #2a2a2a;
}

@media (max-width: 768px) {
    .kb-section {
        padding: 8px 12px;
    }
}

.kb-row {
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
}

@media (max-width: 768px) {
    .kb-row {
        flex-direction: column;
        gap: 8px;
        align-items: stretch;
    }
}

.kb-upload {
    flex: 1;
    min-width: 200px;
}

.kb-info {
    font-size: 12px;
    color: #888;
    white-space: nowrap;
}

@media (max-width: 768px) {
    .kb-info {
        font-size: 11px;
        white-space: normal;
        text-align: center;
    }
}

/* Chat Area - Mobile Responsive */
.chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    padding-bottom: 120px; /* Espaço para composer fixo */
}

@media (max-width: 768px) {
    .chat-area {
        padding: 12px;
        padding-bottom: 140px;
    }
}

.chat-container {
    max-width: 800px;
    margin: 0 auto;
}

/* Messages - Mobile Responsive */
.message {
    margin-bottom: 20px;
    display: flex;
    gap: 10px;
}

@media (max-width: 768px) {
    .message {
        gap: 8px;
        margin-bottom: 16px;
    }
}

.message-avatar {
    width: 28px;
    height: 28px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 600;
    flex-shrink: 0;
}

@media (max-width: 768px) {
    .message-avatar {
        width: 24px;
        height: 24px;
        font-size: 10px;
    }
}

.message.user .message-avatar {
    background: #2a3f5f;
    color: #61afef;
}

.message.assistant .message-avatar {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.message-content {
    flex: 1;
    min-width: 0; /* Permite quebra de texto */
}

.message-role {
    font-size: 11px;
    font-weight: 600;
    color: #888;
    margin-bottom: 4px;
    text-transform: uppercase;
}

.message-text {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 12px 14px;
    color: #e0e0e0;
    font-size: 14px;
    line-height: 1.5;
    word-wrap: break-word;
    overflow-wrap: break-word;
}

@media (max-width: 768px) {
    .message-text {
        font-size: 13px;
        padding: 10px 12px;
    }
}

.message.user .message-text {
    background: #1e2330;
    border-color: #2a3f5f;
}

/* Composer - Mobile Fixed */
.composer {
    background: #141414;
    border-top: 1px solid #2a2a2a;
    padding: 12px 16px;
    position: fixed;
    bottom: 0;
    left: 260px;
    right: 0;
    z-index: 50;
}

@media (max-width: 768px) {
    .composer {
        left: 0;
        padding: 10px 12px;
        padding-bottom: calc(10px + env(safe-area-inset-bottom));
    }
}

.composer-inner {
    max-width: 800px;
    margin: 0 auto;
}

.input-wrapper {
    display: flex;
    gap: 8px;
    align-items: flex-end;
}

@media (max-width: 480px) {
    .input-wrapper {
        flex-direction: column;
        gap: 8px;
        align-items: stretch;
    }
}

.input-container {
    flex: 1;
    min-width: 0;
}

.composer-controls {
    display: flex;
    gap: 8px;
    margin-top: 8px;
    flex-wrap: wrap;
}

@media (max-width: 480px) {
    .composer-controls {
        justify-content: space-between;
    }
}

/* Textarea */
textarea {
    width: 100%;
    background: #1a1a1a;
    color: #e0e0e0;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 10px;
    font-size: 14px;
    resize: none;
    font-family: inherit;
    min-height: 44px;
}

@media (max-width: 768px) {
    textarea {
        font-size: 16px; /* Previne zoom no iOS */
        padding: 12px;
    }
}

textarea:focus {
    outline: none;
    border-color: #3a3a3a;
    box-shadow: 0 0 0 1px #3a3a3a;
}

/* Buttons - Mobile Responsive */
.btn {
    padding: 10px 16px;
    background: #2a2a2a;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    transition: all 0.2s;
    white-space: nowrap;
    min-height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
}

@media (max-width: 768px) {
    .btn {
        font-size: 14px;
        padding: 12px 16px;
    }
}

.btn:hover {
    background: #3a3a3a;
    border-color: #4a4a4a;
}

.btn:active {
    transform: translateY(1px);
}

.btn-primary {
    background: #667eea;
    border-color: #667eea;
    color: white;
}

.btn-primary:hover {
    background: #5a67d8;
    border-color: #5a67d8;
}

.btn-logout {
    background: transparent;
    border: 1px solid #3a3a3a;
    padding: 6px 12px;
    font-size: 12px;
    min-height: auto;
}

.btn-logout:hover {
    border-color: #e06c75;
    color: #e06c75;
}

/* Forms - Mobile Responsive */
input[type="text"],
input[type="password"],
input[type="email"],
input[type="number"] {
    width: 100%;
    padding: 12px;
    background: #0e0e0e;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    color: #e0e0e0;
    font-size: 14px;
    min-height: 44px;
}

@media (max-width: 768px) {
    input[type="text"],
    input[type="password"],
    input[type="email"],
    input[type="number"] {
        font-size: 16px; /* Previne zoom no iOS */
    }
}

input:focus {
    outline: none;
    border-color: #3a3a3a;
}

select {
    width: 100%;
    padding: 10px;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    color: #e0e0e0;
    font-size: 13px;
    min-height: 40px;
}

@media (max-width: 768px) {
    select {
        font-size: 14px;
        min-height: 44px;
    }
}

.form-group {
    margin-bottom: 16px;
}

.form-label {
    display: block;
    margin-bottom: 6px;
    font-size: 13px;
    color: #888;
}

/* File Input - Mobile Friendly */
input[type="file"] {
    width: 100%;
    padding: 8px;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    color: #e0e0e0;
    font-size: 13px;
}

@media (max-width: 768px) {
    input[type="file"] {
        font-size: 14px;
        padding: 10px;
    }
}

/* Admin Panel - Mobile Responsive */
.admin-panel {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 16px;
    margin: 16px;
    display: none; /* Hidden by default */
}

@media (max-width: 768px) {
    .admin-panel {
        margin: 12px;
        padding: 12px;
    }
}

.admin-panel.show {
    display: block;
}

.admin-form-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 8px;
    margin-bottom: 12px;
}

@media (max-width: 768px) {
    .admin-form-row {
        grid-template-columns: 1fr;
    }
}

.user-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 8px;
    font-size: 13px;
    word-wrap: break-word;
}

@media (max-width: 768px) {
    .user-card {
        font-size: 12px;
        padding: 10px;
    }
}

/* Alerts - Mobile Responsive */
.alert {
    padding: 10px 12px;
    border-radius: 6px;
    margin-bottom: 12px;
    font-size: 13px;
    word-wrap: break-word;
}

.alert-error {
    background: rgba(224, 108, 117, 0.1);
    color: #e06c75;
    border: 1px solid rgba(224, 108, 117, 0.3);
}

.alert-success {
    background: rgba(152, 195, 121, 0.1);
    color: #98c379;
    border: 1px solid rgba(152, 195, 121, 0.3);
}

/* Loading dots - Mobile */
.typing-dots {
    display: flex;
    gap: 4px;
    padding: 12px;
    justify-content: flex-start;
}

.typing-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #667eea;
    animation: typing 1.4s infinite;
}

@media (max-width: 768px) {
    .typing-dot {
        width: 5px;
        height: 5px;
    }
}

@keyframes typing {
    0%, 60%, 100% { opacity: 0.2; }
    30% { opacity: 1; }
}

.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

/* Welcome Message - Mobile */
.welcome-message {
    text-align: center;
    padding: 30px 20px;
    color: #666;
}

@media (max-width: 768px) {
    .welcome-message {
        padding: 20px 15px;
    }
}

.welcome-title {
    font-size: 20px;
    margin-bottom: 16px;
    color: #fff;
}

@media (max-width: 768px) {
    .welcome-title {
        font-size: 18px;
    }
}

.welcome-features {
    max-width: 400px;
    margin: 16px auto;
    text-align: left;
    font-size: 14px;
    line-height: 1.8;
}

@media (max-width: 768px) {
    .welcome-features {
        font-size: 13px;
        margin: 12px auto;
    }
}

/* Utility Classes */
.hidden-mobile {
    display: block;
}

@media (max-width: 768px) {
    .hidden-mobile {
        display: none;
    }
}

.show-mobile {
    display: none;
}

@media (max-width: 768px) {
    .show-mobile {
        display: block;
    }
}

/* Safe area for iOS */
@supports (padding: max(0px)) {
    .composer {
        padding-bottom: max(12px, env(safe-area-inset-bottom));
    }
    
    @media (max-width: 768px) {
        .composer {
            padding-bottom: max(10px, env(safe-area-inset-bottom));
        }
    }
}

/* Touch targets */
@media (hover: none) {
    .btn, .session-item, .new-session-btn {
        min-height: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
}

/* Landscape mode adjustments */
@media (max-width: 768px) and (orientation: landscape) {
    .login-card {
        max-height: 90vh;
        overflow-y: auto;
    }
    
    .chat-area {
        padding-bottom: 100px;
    }
    
    .composer {
        padding: 8px 12px;
    }
}
"""

# ---------------- Interface Mobile-First ----------------

app_ui = ui.page_fluid(
    ui.tags.style(MOBILE_RESPONSIVE_CSS),
    ui.output_ui("main_content")
)

def server(input: Inputs, output: Outputs, session: Session):
    # Estado da sessão
    authenticated = reactive.Value(False)
    current_user = reactive.Value("")
    is_admin = reactive.Value(False)
    login_message = reactive.Value("")
    
    # Estado do chat
    history = reactive.Value([])
    typing = reactive.Value(False)
    
    # Estado mobile
    sidebar_open = reactive.Value(False)
    admin_panel_open = reactive.Value(False)
    
    def push(role, content):
        history.set(history() + [{"role": role, "content": content}])
    
    @output
    @render.ui
    def main_content():
        """Renderiza login ou app principal com design mobile-first"""
        if not authenticated():
            # LOGIN PAGE - MOBILE OPTIMIZED
            return ui.div({"class": "login-container"},
                ui.div({"class": "login-card"},
                    ui.div({"class": "login-header"},
                        ui.h1({"class": "login-title"}, "🚀 Origin Assistant"),
                        ui.p({"class": "login-subtitle"}, "Especialista em OriginPro")
                    ),
                    
                    ui.output_ui("login_feedback"),
                    
                    ui.div({"class": "form-group"},
                        ui.span({"class": "form-label"}, "Usuário"),
                        ui.input_text("username", None, placeholder="Digite seu usuário")
                    ),
                    
                    ui.div({"class": "form-group"},
                        ui.span({"class": "form-label"}, "Senha"),
                        ui.input_password("password", None, placeholder="Digite sua senha")
                    ),
                    
                    ui.input_action_button("login_btn", "Entrar", 
                        class_="btn btn-primary", 
                        style="width: 100%; margin-bottom: 16px;"
                    ),
                    
                    ui.hr({"style": "margin: 16px 0; border-color: #2a2a2a;"}),
                    
                    ui.div({"style": "text-align: center; color: #666; font-size: 12px;"},
                        "Demo: admin / admin123"
                    )
                )
            )
        else:
            # APP PRINCIPAL - MOBILE FIRST DESIGN
            return ui.TagList(
                # Mobile Header (only visible on mobile)
                ui.div({"class": "mobile-header"},
                    ui.div({"class": "mobile-title"},
                        "🚀 Origin Assistant"
                    ),
                    ui.div({"style": "display: flex; gap: 8px;"},
                        ui.input_action_button("toggle_sidebar", "☰", 
                            class_="mobile-menu-btn"
                        ),
                        ui.input_action_button("logout_btn_mobile", "Sair",
                            class_="mobile-menu-btn"
                        )
                    )
                ),
                
                # Sidebar Overlay (mobile)
                ui.div({"class": "sidebar-overlay", "id": "sidebar-overlay"}),
                
                # Sidebar - Responsiva
                ui.div({"class": "sidebar", "id": "main-sidebar"},
                    ui.div({"class": "sidebar-header"},
                        ui.div({"class": "sidebar-title"},
                            "🚀 Origin Assistant"
                        ),
                        ui.div({"class": "sidebar-user"},
                            f"Usuário: {current_user()}"
                        )
                    ),
                    
                    ui.div({"class": "sessions-list"},
                        ui.div({"class": "session-item active"},
                            "💬 Conversa atual"
                        )
                    ),
                    
                    ui.input_action_button("clear_chat", "➕ Nova Conversa",
                        class_="new-session-btn"
                    ),
                    
                    # Actions no bottom da sidebar
                    ui.div(
                        {"style": "margin-top: auto; padding: 8px; border-top: 1px solid #2a2a2a;"},
                        ui.input_action_button(
                            "show_admin", 
                            "👤 Admin",
                            class_="btn", 
                            style="width: 100%; margin-bottom: 8px;"
                        ) if is_admin() else ui.TagList(),
                        ui.input_action_button(
                            "logout_btn", 
                            "🚪 Logout",
                            class_="btn btn-logout hidden-mobile", 
                            style="width: 100%;"
                        )
                    )
                ),
                
                # Main Content Area
                ui.div({"class": "main-content"},
                    # Header Bar (hidden on mobile)
                    ui.div({"class": "header-bar hidden-mobile"},
                        ui.div({"class": "header-title"},
                            "📊 Especialista em OriginPro"
                        ),
                        ui.div(
                            {"class": "header-actions"},
                            ui.input_action_button(
                                "show_admin_desktop", 
                                "Admin",
                                class_="btn btn-logout"
                            ) if is_admin() else ui.TagList(),
                            ui.input_action_button(
                                "logout_btn_desktop", 
                                "Logout",
                                class_="btn btn-logout"
                            )
                        )
                    ),
                    
                    # Knowledge Base Section - Mobile Optimized
                    ui.div({"class": "kb-section"},
                        ui.div({"class": "kb-row"},
                            ui.div({"class": "kb-upload"},
                                ui.input_file("docs", "📚 Documentação OriginPro (PDFs)", 
                                    multiple=True, 
                                    accept=[".pdf"]
                                )
                            ),
                            ui.div({"class": "kb-info"},
                                ui.output_text("kb_status")
                            )
                        )
                    ),
                    
                    # Admin Panel (condicional e mobile-friendly)
                    ui.output_ui("admin_panel"),
                    
                    # Chat Area
                    ui.div({"class": "chat-area"},
                        ui.div({"class": "chat-container"},
                            ui.output_ui("chat_thread")
                        )
                    ),
                    
                    # Composer - Fixed at bottom
                    ui.div({"class": "composer"},
                        ui.div({"class": "composer-inner"},
                            ui.div({"class": "input-wrapper"},
                                ui.div({"class": "input-container"},
                                    ui.input_text_area("prompt", None, 
                                        placeholder="Pergunte sobre OriginPro: gráficos, análise de dados...",
                                        rows=2
                                    ),
                                    ui.div({"class": "composer-controls"},
                                        ui.input_select("model", None, 
                                            {
                                                "claude-3-haiku-20240307": "⚡ Haiku (rápido)",
                                                "claude-3-5-sonnet-20240620": "✨ Sonnet (avançado)"
                                            }, 
                                            selected="claude-3-haiku-20240307",
                                            style="flex: 1; min-width: 150px;"
                                        ),
                                        ui.input_action_button("send", "Enviar", 
                                            class_="btn btn-primary",
                                            style="min-width: 80px;"
                                        )
                                    )
                                )
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
                return ui.div({"class": "alert alert-error"}, f"⚠️ {msg}")
        return ui.TagList()
    
    @output
    @render.ui
    def admin_panel():
        """Painel administrativo mobile-friendly"""
        if not is_admin():
            return ui.TagList()
        
        users = list_users()
        
        panel_class = "admin-panel"
        if admin_panel_open():
            panel_class += " show"
        
        return ui.div({"class": panel_class},
            ui.h3("👥 Gerenciar Usuários", style="margin-bottom: 16px;"),
            
            # Adicionar usuário - Mobile optimized
            ui.div({"class": "admin-form-row"},
                ui.input_text("new_username", None, placeholder="Usuário"),
                ui.input_password("new_password", None, placeholder="Senha"),
                ui.input_text("new_email", None, placeholder="Email (opcional)"),
                ui.input_numeric("new_months", None, value=12, min=1, max=36, placeholder="Meses")
            ),
            ui.input_action_button("add_user_btn", "Adicionar Usuário", 
                class_="btn", style="width: 100%; margin-bottom: 16px;"
            ),
            
            ui.hr(style="margin: 16px 0; border-color: #2a2a2a;"),
            
            # Lista de usuários
            ui.h4("Usuários Cadastrados", style="margin-bottom: 12px; font-size: 16px;"),
            ui.TagList([
                ui.div({"class": "user-card"},
                    ui.div(style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 4px;",
                        ui.strong(user[0] + (" Admin" if user[6] else "")),
                        ui.span(
                            {"style": "color: #98c379;" if user[4] else "color: #e06c75;"},
                            "Ativo" if user[4] else "Inativo"
                        )
                    ),
                    ui.div(style="font-size: 11px; color: #666; margin-top: 4px;",
                        f"Email: {user[1] or 'sem email'} | Exp: {user[5][:10] if user[5] else 'N/A'}"
                    )
                ) for user in users
            ])
        )
    
    @output
    @render.text
    def kb_status():
        """Status da base de conhecimento"""
        if not HAVE_RAG_DEPS:
            return "⚠️ RAG indisponível"
        chunks, _, _ = load_index()
        n_docs = len({c["source"] for c in chunks}) if chunks else 0
        n_chunks = len(chunks)
        return f"📄 {n_docs} docs • 🧩 {n_chunks} chunks"
    
    @output
    @render.ui
    def chat_thread():
        """Renderiza thread do chat com design mobile-friendly"""
        items = []
        
        # Mensagem inicial se chat vazio
        if not history() and not typing():
            items.append(
                ui.div({"class": "welcome-message"},
                    ui.h3({"class": "welcome-title"}, "Bem-vindo ao Origin Assistant!"),
                    ui.p("Faça perguntas sobre:"),
                    ui.div({"class": "welcome-features"},
                        "• Como criar gráficos no OriginPro", ui.br(),
                        "• Análise de dados e estatísticas", ui.br(),
                        "• Importação e manipulação de dados", ui.br(),
                        "• Ferramentas de fitting e análise", ui.br(),
                        "• Customização de gráficos científicos"
                    )
                )
            )
        
        # Mensagens do chat
        for m in history():
            is_user = m["role"] == "user"
            
            items.append(
                ui.div({"class": f"message {'user' if is_user else 'assistant'}"},
                    ui.div({"class": "message-avatar"},
                        "U" if is_user else "OA"
                    ),
                    ui.div({"class": "message-content"},
                        ui.div({"class": "message-role"},
                            "Você" if is_user else "Origin Assistant"
                        ),
                        ui.div({"class": "message-text"},
                            ui.markdown(m["content"])
                        )
                    )
                )
            )
        
        # Typing indicator
        if typing():
            items.append(
                ui.div({"class": "message assistant"},
                    ui.div({"class": "message-avatar"}, "OA"),
                    ui.div({"class": "message-content"},
                        ui.div({"class": "message-role"}, "Origin Assistant"),
                        ui.div({"class": "message-text"},
                            ui.div({"class": "typing-dots"},
                                ui.div({"class": "typing-dot"}),
                                ui.div({"class": "typing-dot"}),
                                ui.div({"class": "typing-dot"})
                            )
                        )
                    )
                )
            )
        
        return ui.TagList(*items)
    
    # Event Handlers - Login
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
            print(f"[LOGIN] Usuário {username} autenticado com sucesso")
    
    # Event Handlers - Logout (múltiplos botões)
    @reactive.Effect
    @reactive.event(input.logout_btn, input.logout_btn_mobile, input.logout_btn_desktop)
    def handle_logout():
        """Processa logout"""
        authenticated.set(False)
        current_user.set("")
        is_admin.set(False)
        history.set([])
        sidebar_open.set(False)
        admin_panel_open.set(False)
        ui.notification_show("Logout realizado", type="message", duration=2)
    
    # Event Handlers - Mobile Navigation
    @reactive.Effect
    @reactive.event(input.toggle_sidebar)
    def toggle_sidebar():
        """Toggle da sidebar mobile"""
        sidebar_open.set(not sidebar_open())
        # Adicionar/remover classes via JavaScript seria ideal aqui
        # Por limitação do Shiny, faremos isso via CSS media queries
    
    # Event Handlers - Admin
    @reactive.Effect
    @reactive.event(input.show_admin, input.show_admin_desktop)
    def toggle_admin():
        """Toggle do painel admin"""
        admin_panel_open.set(not admin_panel_open())
    
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
            ui.notification_show("Preencha usuário e senha", type="warning", duration=3)
            return
        
        success, message = add_user(username, password, email, months)
        ui.notification_show(message, type="message" if success else "warning", duration=3)
        
        if success:
            # Limpar campos - CORREÇÃO AQUI
            ui.update_text("new_username", value="")
            # REMOVIDO ui.update_password que não existe
            ui.update_text("new_email", value="")
            ui.update_numeric("new_months", value=12)
    
    # Event Handlers - Chat
    @reactive.Effect
    @reactive.event(input.clear_chat)
    def _clear():
        """Limpa o chat"""
        history.set([])
        ui.update_text_area("prompt", value="")
        sidebar_open.set(False)  # Fecha sidebar no mobile
    
    @reactive.Effect
    @reactive.event(input.docs)
    def _ingest_pdfs():
        """Processa PDFs enviados"""
        if not HAVE_RAG_DEPS:
            ui.notification_show("⚠️ Instale pypdf e sklearn para RAG", type="error", duration=5)
            return
        
        files = input.docs() or []
        if not files:
            return
        
        paths = []
        for f in files:
            src = Path(f["datapath"])
            dst = CACHE_DIR / f["name"]
            dst.write_bytes(src.read_bytes())
            paths.append(dst)
        
        total = add_pdfs_to_index(paths)
        ui.notification_show(f"✅ {len(files)} PDFs processados • Total: {total} chunks", type="success", duration=4)
    
    @reactive.Effect
    @reactive.event(input.send)
    def _send():
        """Envia mensagem no chat"""
        if not authenticated():
            return
            
        q = (input.prompt() or "").strip()
        if not q:
            return
        
        push("user", q)
        ui.update_text_area("prompt", value="")
        sidebar_open.set(False)  # Fecha sidebar no mobile
        
        if client is None:
            push("assistant", "⚠️ Configure ANTHROPIC_API_KEY para usar o Claude.")
            return
        
        typing.set(True)
        model = input.model() or "claude-3-haiku-20240307"
        
        try:
            reply = chat_reply_with_context(history(), model)
        except Exception as e:
            reply = f"❌ Erro: {str(e)}"
        finally:
            typing.set(False)
        
        push("assistant", reply)

app = App(app_ui, server)







