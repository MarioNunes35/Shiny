# app_origin_fixed.py - Origin Software Assistant com persistência corrigida
# Interface inspirada no Claude Code UI

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
# No Posit Connect, use /home/shiny para persistência
if os.path.exists("/home/shiny"):
    # Ambiente Posit Connect
    BASE_DIR = Path("/home/shiny/.origin_assistant")
else:
    # Ambiente local
    BASE_DIR = Path.home() / ".origin_assistant"

# Criar estrutura de diretórios
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
        
        # Criar tabela
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
        
        # Verificar se admin existe
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
        
        # Verificar senha
        if hash_password(password) != password_hash:
            con.close()
            return False, "Usuário ou senha incorretos", False
        
        # Verificar se usuário está ativo
        if not active:
            con.close()
            return False, "Usuário desativado. Entre em contato com o suporte.", False
        
        # Verificar expiração
        if subscription_expires:
            expiry = datetime.fromisoformat(subscription_expires)
            if datetime.utcnow() > expiry:
                con.close()
                return False, "Assinatura expirada. Renove seu acesso.", False
        
        # Atualizar último login
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

print(f"[BOOT] RAG={HAVE_RAG_DEPS} | Claude={(client is not None)}")

# Paths for RAG
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

    # RAG com contexto
    if HAVE_RAG_DEPS:
        ctx, cites, stats = build_context(question)
    else:
        ctx, cites, stats = ("", "", {"top": 0.0, "chars": 0, "nhits": 0})

    use_rag = bool(ctx) and not rag_should_fallback(stats)

    # SYSTEM PROMPT FOCADO NO ORIGINPRO
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

# ---------------- CSS Estilo Claude Code UI ----------------

CLAUDE_CODE_UI_CSS = """
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
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #1a1a1a;
}

::-webkit-scrollbar-thumb {
    background: #3a3a3a;
    border-radius: 4px;
}

/* Login Page */
.login-container {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #0e0e0e;
}

.login-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 40px;
    width: 100%;
    max-width: 400px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.8);
}

.login-header {
    text-align: center;
    margin-bottom: 32px;
}

.login-title {
    font-size: 24px;
    font-weight: 600;
    margin-bottom: 8px;
    color: #fff;
}

.login-subtitle {
    color: #888;
    font-size: 14px;
}

/* Sidebar */
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

.sessions-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
}

.session-item {
    padding: 10px 12px;
    margin-bottom: 4px;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    color: #888;
    cursor: pointer;
    font-size: 14px;
    transition: all 0.2s;
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
    padding: 10px;
    background: #1e1e1e;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    color: #e0e0e0;
    cursor: pointer;
    text-align: center;
    transition: all 0.2s;
}

.new-session-btn:hover {
    background: #2a2a2a;
    border-color: #4a4a4a;
}

/* Main Content */
.main-content {
    margin-left: 260px;
    height: 100vh;
    display: flex;
    flex-direction: column;
    background: #0e0e0e;
}

.header-bar {
    background: #141414;
    border-bottom: 1px solid #2a2a2a;
    padding: 12px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
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

/* Chat Area */
.chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
}

.chat-container {
    max-width: 800px;
    margin: 0 auto;
}

/* Messages */
.message {
    margin-bottom: 24px;
    display: flex;
    gap: 12px;
}

.message-avatar {
    width: 32px;
    height: 32px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 600;
    flex-shrink: 0;
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
}

.message-role {
    font-size: 12px;
    font-weight: 600;
    color: #888;
    margin-bottom: 4px;
    text-transform: uppercase;
}

.message-text {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 12px 16px;
    color: #e0e0e0;
    font-size: 14px;
    line-height: 1.6;
}

.message.user .message-text {
    background: #1e2330;
    border-color: #2a3f5f;
}

/* Composer */
.composer {
    background: #141414;
    border-top: 1px solid #2a2a2a;
    padding: 16px 20px;
}

.composer-inner {
    max-width: 800px;
    margin: 0 auto;
}

.input-wrapper {
    display: flex;
    gap: 12px;
    align-items: flex-end;
}

.input-container {
    flex: 1;
}

textarea {
    width: 100%;
    background: #1a1a1a;
    color: #e0e0e0;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 12px;
    font-size: 14px;
    resize: none;
    font-family: inherit;
}

textarea:focus {
    outline: none;
    border-color: #3a3a3a;
    box-shadow: 0 0 0 1px #3a3a3a;
}

/* Buttons */
.btn {
    padding: 10px 20px;
    background: #2a2a2a;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    transition: all 0.2s;
}

.btn:hover {
    background: #3a3a3a;
    border-color: #4a4a4a;
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
}

.btn-logout:hover {
    border-color: #e06c75;
    color: #e06c75;
}

/* Forms */
input[type="text"],
input[type="password"],
input[type="email"] {
    width: 100%;
    padding: 10px;
    background: #0e0e0e;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    color: #e0e0e0;
    font-size: 14px;
}

input:focus {
    outline: none;
    border-color: #3a3a3a;
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

/* Knowledge Base */
.kb-section {
    padding: 16px 20px;
    background: #141414;
    border-bottom: 1px solid #2a2a2a;
}

.kb-info {
    font-size: 13px;
    color: #888;
    margin-top: 8px;
}

/* Alerts */
.alert {
    padding: 10px 14px;
    border-radius: 6px;
    margin-bottom: 16px;
    font-size: 13px;
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

/* Admin Panel */
.admin-panel {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 20px;
    margin: 20px;
}

.user-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 8px;
}

/* Loading dots */
.typing-dots {
    display: flex;
    gap: 4px;
    padding: 12px;
}

.typing-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #667eea;
    animation: typing 1.4s infinite;
}

@keyframes typing {
    0%, 60%, 100% { opacity: 0.2; }
    30% { opacity: 1; }
}

.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
"""

# ---------------- Interface do App ----------------

app_ui = ui.page_fluid(
    ui.tags.style(CLAUDE_CODE_UI_CSS),
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
    
    def push(role, content):
        history.set(history() + [{"role": role, "content": content}])
    
    @output
    @render.ui
    def main_content():
        """Renderiza login ou app principal"""
        if not authenticated():
            # LOGIN PAGE
            return ui.div({"class": "login-container"},
                ui.div({"class": "login-card"},
                    ui.div({"class": "login-header"},
                        ui.h1({"class": "login-title"}, "🚀 Origin Software Assistant"),
                        ui.p({"class": "login-subtitle"}, "Sistema especializado em OriginPro")
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
                    
                    ui.input_action_button("login_btn", "Entrar", class_="btn btn-primary", style="width: 100%;"),
                    
                    ui.hr({"style": "margin: 20px 0; border-color: #2a2a2a;"}),
                    
                    ui.div({"style": "text-align: center; color: #666; font-size: 13px;"},
                        "Credenciais demo: admin / admin123"
                    )
                )
            )
        else:
            # APP PRINCIPAL COM SIDEBAR
            return ui.TagList(
                # Sidebar estilo Claude Code UI
                ui.div({"class": "sidebar"},
                    ui.div({"class": "sidebar-header"},
                        ui.div({"class": "sidebar-title"},
                            "🚀 Origin Assistant"
                        ),
                        ui.div({"style": "font-size: 12px; color: #666; margin-top: 4px;"},
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
                    )
                ),
                
                # Main Content Area
                ui.div({"class": "main-content"},
                    # Header Bar
                    ui.div({"class": "header-bar"},
                        ui.div({"class": "header-title"},
                            "📊 Especialista em OriginPro"
                        ),
                        ui.div({"class": "header-actions"},
                            ui.input_action_button("logout_btn", "Logout",
                                class_="btn btn-logout"
                            ) if not is_admin() else ui.TagList(
                                ui.input_action_button("show_admin", "Admin",
                                    class_="btn btn-logout"
                                ),
                                ui.input_action_button("logout_btn", "Logout",
                                    class_="btn btn-logout"
                                )
                            )
                        )
                    ),
                    
                    # Knowledge Base Section
                    ui.div({"class": "kb-section"},
                        ui.row(
                            ui.column(8,
                                ui.input_file("docs", "📚 Adicionar documentação do OriginPro (PDFs)", 
                                    multiple=True, 
                                    accept=[".pdf"]
                                )
                            ),
                            ui.column(4,
                                ui.div({"class": "kb-info"},
                                    ui.output_text("kb_status")
                                )
                            )
                        )
                    ),
                    
                    # Admin Panel (condicional)
                    ui.output_ui("admin_panel"),
                    
                    # Chat Area
                    ui.div({"class": "chat-area"},
                        ui.div({"class": "chat-container"},
                            ui.output_ui("chat_thread")
                        )
                    ),
                    
                    # Composer
                    ui.div({"class": "composer"},
                        ui.div({"class": "composer-inner"},
                            ui.div({"class": "input-wrapper"},
                                ui.div({"class": "input-container"},
                                    ui.input_text_area("prompt", None, 
                                        placeholder="Pergunte sobre o OriginPro: plotagem, análise de dados, ferramentas estatísticas...",
                                        rows=3
                                    ),
                                    ui.div({"style": "display: flex; gap: 10px; margin-top: 10px;"},
                                        ui.input_select("model", None, 
                                            {
                                                "claude-3-haiku-20240307": "⚡ Haiku (rápido)",
                                                "claude-3-5-sonnet-20240620": "✨ Sonnet (avançado)"
                                            }, 
                                            selected="claude-3-haiku-20240307"
                                        )
                                    )
                                ),
                                ui.input_action_button("send", "Enviar", 
                                    class_="btn btn-primary"
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
        """Painel administrativo para admins"""
        if not is_admin():
            return ui.TagList()
        
        users = list_users()
        
        return ui.div({"class": "admin-panel"},
            ui.h3("👥 Gerenciar Usuários"),
            ui.hr(),
            
            # Adicionar usuário
            ui.row(
                ui.column(3,
                    ui.input_text("new_username", "Usuário", placeholder="nome")
                ),
                ui.column(3,
                    ui.input_password("new_password", "Senha", placeholder="senha")
                ),
                ui.column(3,
                    ui.input_text("new_email", "Email", placeholder="email")
                ),
                ui.column(3,
                    ui.input_numeric("new_months", "Meses", value=12, min=1, max=36)
                )
            ),
            ui.input_action_button("add_user_btn", "Adicionar Usuário", class_="btn"),
            ui.output_text("add_user_feedback"),
            
            ui.hr(),
            
            # Lista de usuários
            ui.h4("Usuários Cadastrados"),
            ui.TagList(*[
                ui.div({"class": "user-card"},
                    ui.strong(user[0] + (" 👑" if user[6] else "")),
                    ui.span(f" • {user[1] or 'sem email'}"),
                    ui.span(f" • Ativo: {'Sim' if user[4] else 'Não'}"),
                    ui.span(f" • Expira: {user[5][:10] if user[5] else 'N/A'}")
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
        """Renderiza thread do chat"""
        items = []
        
        # Mensagem inicial se chat vazio
        if not history() and not typing():
            items.append(
                ui.div({"style": "text-align: center; padding: 40px; color: #666;"},
                    ui.h3("Bem-vindo ao Origin Software Assistant!"),
                    ui.p("Faça perguntas sobre:"),
                    ui.div({"style": "margin-top: 20px; text-align: left; max-width: 400px; margin: 20px auto;"},
                        "• Como criar gráficos no OriginPro",
                        ui.br(),
                        "• Análise de dados e estatísticas",
                        ui.br(),
                        "• Importação e manipulação de dados",
                        ui.br(),
                        "• Ferramentas de fitting e análise",
                        ui.br(),
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
    
    @reactive.Effect
    @reactive.event(input.logout_btn)
    def handle_logout():
        """Processa logout"""
        authenticated.set(False)
        current_user.set("")
        is_admin.set(False)
        history.set([])
        ui.notification_show("Logout realizado", type="message", duration=2)
    
    # Event Handlers - Admin
    @output
    @render.text
    def add_user_feedback():
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
            ui.notification_show("Preencha usuário e senha", type="warning", duration=3)
            return
        
        success, message = add_user(username, password, email, months)
        ui.notification_show(message, type="message" if success else "warning", duration=3)
        
        if success:
            # Limpar campos
            ui.update_text("new_username", value="")
            ui.update_password("new_password", value="")
            ui.update_text("new_email", value="")
    
    # Event Handlers - Chat
    @reactive.Effect
    @reactive.event(input.clear_chat)
    def _clear():
        """Limpa o chat"""
        history.set([])
        ui.update_text_area("prompt", value="")
    
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







