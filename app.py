# app.py - Origin Software Assistant com Design Mobile Responsive e Melhorias de Segurança
# Sistema completo com autenticação segura, RAG otimizado e interface mobile-first

from shiny import App, ui, render, reactive, Inputs, Outputs, Session
from dotenv import load_dotenv
import os
import hashlib
import secrets
import hmac
import sqlite3
import json
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import re

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
LOGS_DIR = DATA_DIR / "logs"

# Criar diretórios necessários
for d in (BASE_DIR, DATA_DIR, AUTH_DIR, CACHE_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------- Sistema de Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"[BOOT] Usando diretório base: {BASE_DIR}")
logger.info(f"[BOOT] Banco de dados em: {USER_DB_PATH}")

# ---------------- Sistema de Autenticação Seguro ----------------

def hash_password_secure(password: str, salt: str = None) -> Tuple[str, str]:
    """Cria hash seguro com salt usando PBKDF2"""
    if salt is None:
        salt = secrets.token_hex(32)
    
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100_000  # 100,000 iterações para maior segurança
    )
    
    return key.hex(), salt

def verify_password_secure(password: str, password_hash: str, salt: str) -> bool:
    """Verifica senha contra hash armazenado de forma segura"""
    new_hash, _ = hash_password_secure(password, salt)
    return hmac.compare_digest(new_hash, password_hash)

def create_user_db():
    """Cria tabela de usuários com campos de segurança aprimorados"""
    try:
        con = sqlite3.connect(str(USER_DB_PATH))
        cur = con.cursor()
        
        # Criar tabela com campos adicionais de segurança
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                email TEXT,
                created_at TEXT,
                last_login TEXT,
                active INTEGER DEFAULT 1,
                subscription_expires TEXT,
                is_admin INTEGER DEFAULT 0,
                session_token TEXT,
                token_expires TEXT,
                failed_attempts INTEGER DEFAULT 0,
                locked_until TEXT
            );
        """)
        
        # Verificar se usuário admin existe
        cur.execute("SELECT id FROM users WHERE username = ?", ("admin",))
        if not cur.fetchone():
            admin_pass_hash, admin_salt = hash_password_secure("admin123")
            cur.execute("""
                INSERT INTO users(username, password_hash, salt, email, created_at, active, subscription_expires, is_admin)
                VALUES(?, ?, ?, ?, ?, 1, ?, 1)
            """, ("admin", admin_pass_hash, admin_salt, "admin@origin.com", 
                  datetime.utcnow().isoformat(),
                  (datetime.utcnow() + timedelta(days=365)).isoformat()))
            logger.info("[AUTH] Usuário admin criado com senha padrão: admin123")
        
        con.commit()
        con.close()
        logger.info(f"[AUTH] Banco de dados inicializado com segurança aprimorada")
        
    except Exception as e:
        logger.error(f"[AUTH ERROR] Erro ao criar banco: {e}")

def validate_user(username: str, password: str) -> Tuple[bool, str, bool]:
    """Valida credenciais do usuário com proteção contra força bruta"""
    if not username or not password:
        return False, "Usuário e senha são obrigatórios", False
    
    try:
        con = sqlite3.connect(str(USER_DB_PATH))
        cur = con.cursor()
        
        # Verificar tentativas falhadas e bloqueio
        cur.execute("""
            SELECT password_hash, salt, active, subscription_expires, is_admin, 
                   failed_attempts, locked_until
            FROM users WHERE username = ?
        """, (username,))
        result = cur.fetchone()
        
        if not result:
            con.close()
            return False, "Usuário ou senha incorretos", False
        
        password_hash, salt, active, subscription_expires, is_admin, failed_attempts, locked_until = result
        
        # Verificar se conta está bloqueada
        if locked_until:
            lock_time = datetime.fromisoformat(locked_until)
            if datetime.utcnow() < lock_time:
                con.close()
                return False, f"Conta bloqueada por muitas tentativas. Tente novamente em alguns minutos.", False
            else:
                # Desbloquear conta
                cur.execute("UPDATE users SET locked_until = NULL, failed_attempts = 0 WHERE username = ?", (username,))
        
        # Verificar senha
        if not verify_password_secure(password, password_hash, salt):
            # Incrementar tentativas falhadas
            failed_attempts += 1
            if failed_attempts >= 5:
                # Bloquear conta por 15 minutos
                locked_until = (datetime.utcnow() + timedelta(minutes=15)).isoformat()
                cur.execute("""
                    UPDATE users SET failed_attempts = ?, locked_until = ? WHERE username = ?
                """, (failed_attempts, locked_until, username))
            else:
                cur.execute("UPDATE users SET failed_attempts = ? WHERE username = ?", (failed_attempts, username))
            
            con.commit()
            con.close()
            return False, "Usuário ou senha incorretos", False
        
        # Resetar tentativas falhadas em login bem-sucedido
        cur.execute("UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE username = ?", (username,))
        
        if not active:
            con.close()
            return False, "Usuário desativado. Entre em contato com o suporte.", False
        
        if subscription_expires:
            expiry = datetime.fromisoformat(subscription_expires)
            if datetime.utcnow() > expiry:
                con.close()
                return False, "Assinatura expirada. Renove seu acesso.", False
        
        # Gerar token de sessão
        session_token = secrets.token_hex(32)
        token_expires = (datetime.utcnow() + timedelta(hours=24)).isoformat()
        
        cur.execute("""
            UPDATE users SET last_login = ?, session_token = ?, token_expires = ? 
            WHERE username = ?
        """, (datetime.utcnow().isoformat(), session_token, token_expires, username))
        
        con.commit()
        con.close()
        
        logger.info(f"[AUTH] Login bem-sucedido para usuário: {username}")
        return True, "Login realizado com sucesso!", bool(is_admin)
        
    except Exception as e:
        logger.error(f"[AUTH ERROR] Erro ao validar usuário: {e}")
        return False, "Erro ao processar login", False

def add_user(username: str, password: str, email: str = "", months: int = 12) -> Tuple[bool, str]:
    """Adiciona novo usuário com hash seguro"""
    try:
        con = sqlite3.connect(str(USER_DB_PATH))
        cur = con.cursor()
        
        password_hash, salt = hash_password_secure(password)
        expiry = datetime.utcnow() + timedelta(days=30*months)
        
        cur.execute("""
            INSERT INTO users(username, password_hash, salt, email, created_at, active, subscription_expires, is_admin)
            VALUES(?, ?, ?, ?, ?, 1, ?, 0)
        """, (username, password_hash, salt, email, 
              datetime.utcnow().isoformat(), expiry.isoformat()))
        
        con.commit()
        con.close()
        
        logger.info(f"[ADMIN] Novo usuário criado: {username}")
        return True, f"Usuário '{username}' criado com sucesso!"
        
    except sqlite3.IntegrityError:
        return False, "Este nome de usuário já existe"
    except Exception as e:
        logger.error(f"[AUTH ERROR] Erro ao adicionar usuário: {e}")
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
        logger.error(f"[AUTH ERROR] Erro ao listar usuários: {e}")
        return []

def log_user_action(username: str, action: str, details: str = ""):
    """Log de ações do usuário para auditoria"""
    logger.info(f"USER_ACTION | user={username} | action={action} | details={details}")

def log_api_usage(username: str, model: str, tokens_estimate: int):
    """Log de uso da API do Claude"""
    logger.info(f"API_USAGE | user={username} | model={model} | tokens={tokens_estimate}")

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
        logger.info("[CLAUDE] Cliente Anthropic inicializado com sucesso")
    except Exception as e:
        logger.error(f"[CLAUDE ERROR] Falha ao inicializar cliente: {e}")
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
    import numpy as np
except Exception:
    HAVE_RAG_DEPS = False
    PdfReader = None
    TfidfVectorizer = None
    cosine_similarity = None
    dump = load = None
    np = None

logger.info(f"[BOOT] RAG={HAVE_RAG_DEPS} | Claude={(client is not None)}")

CHUNKS_JSON = CACHE_DIR / "chunks.json"
VECTORIZER_JOBLIB = CACHE_DIR / "tfidf_vectorizer.joblib"
MATRIX_JOBLIB = CACHE_DIR / "tfidf_matrix.joblib"

# ---------------- RAG Functions Otimizadas ----------------

def extract_text(pdf_path: Path) -> str:
    """Extrai texto de PDF"""
    if not HAVE_RAG_DEPS:
        return ""
    try:
        reader = PdfReader(str(pdf_path))
        txt = []
        for pg in reader.pages:
            txt.append(pg.extract_text() or "")
        return "\n".join(txt)
    except Exception as e:
        logger.error(f"[RAG ERROR] Erro ao extrair texto do PDF: {e}")
        return ""

def chunk_text_smart(text: str, max_chars=900, overlap=220) -> List[str]:
    """Chunking inteligente preservando contexto"""
    if not HAVE_RAG_DEPS:
        return []
    
    # Limpar espaços extras
    text = re.sub(r"\s+", " ", text).strip()
    
    # Dividir por parágrafos primeiro
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para_size = len(para)
        
        if current_size + para_size > max_chars:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return [c for c in chunks if c.strip()]

def load_index():
    """Carrega índice do cache"""
    if CHUNKS_JSON.exists() and VECTORIZER_JOBLIB.exists() and MATRIX_JOBLIB.exists():
        try:
            chunks = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
            vectorizer = load(VECTORIZER_JOBLIB)
            matrix = load(MATRIX_JOBLIB)
            return chunks, vectorizer, matrix
        except Exception as e:
            logger.error(f"[RAG ERROR] Erro ao carregar índice: {e}")
    return [], None, None

def save_index(chunks, vectorizer, matrix):
    """Salva índice no cache"""
    try:
        CHUNKS_JSON.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        dump(vectorizer, VECTORIZER_JOBLIB)
        dump(matrix, MATRIX_JOBLIB)
        logger.info(f"[RAG] Índice salvo com {len(chunks)} chunks")
    except Exception as e:
        logger.error(f"[RAG ERROR] Erro ao salvar índice: {e}")

def add_pdfs_to_index(file_paths: list):
    """Adiciona PDFs ao índice RAG"""
    if not HAVE_RAG_DEPS:
        return 0
    
    chunks, _, _ = load_index()
    new_chunks = []
    
    for p in file_paths:
        try:
            doc_id = hashlib.sha256(p.read_bytes()).hexdigest()
            text = extract_text(p)
            
            if not text:
                logger.warning(f"[RAG] Nenhum texto extraído de {p.name}")
                continue
            
            for idx, ch in enumerate(chunk_text_smart(text)):
                new_chunks.append({
                    "doc_id": doc_id,
                    "source": p.name,
                    "chunk_id": f"{doc_id[:8]}-{idx}",
                    "text": ch
                })
            
            logger.info(f"[RAG] Processado: {p.name} - {len(new_chunks)} chunks")
            
        except Exception as e:
            logger.error(f"[RAG ERROR] Erro ao processar {p.name}: {e}")
    
    if not new_chunks:
        return len(chunks)
    
    all_chunks = chunks + new_chunks
    corpus = [c["text"] for c in all_chunks]
    
    # Vetorização otimizada
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=120_000,
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(corpus)
    
    save_index(all_chunks, vectorizer, matrix)
    return len(all_chunks)

def hybrid_retrieve(query: str, k=4) -> Tuple[List[Dict], float]:
    """Busca híbrida com boost para termos do OriginPro"""
    chunks, vectorizer, matrix = load_index()
    
    if not chunks or vectorizer is None:
        return [], 0.0
    
    try:
        # Busca TF-IDF
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, matrix)[0]
        
        # Keywords do OriginPro para boost
        origin_keywords = [
            'worksheet', 'workbook', 'graph', 'plot', 'analysis',
            'fitting', 'statistics', 'import', 'export', 'layer',
            'column', 'row', 'axis', 'legend', 'annotation',
            'origin', 'originpro', 'labtalk', 'originc'
        ]
        
        query_lower = query.lower()
        
        # Aplicar boost baseado em keywords
        for i, chunk in enumerate(chunks):
            boost = 1.0
            text_lower = chunk["text"].lower()
            
            # Boost por keywords presentes
            for keyword in origin_keywords:
                if keyword in query_lower and keyword in text_lower:
                    boost *= 1.3
            
            # Boost por menções ao Origin
            origin_mentions = text_lower.count('origin') + text_lower.count('originpro')
            boost *= (1 + origin_mentions * 0.05)
            
            sims[i] *= boost
        
        # Selecionar top-k resultados
        idx = sims.argsort()[::-1][:k]
        
        hits = [{
            "score": float(sims[i]),
            "text": chunks[i]["text"],
            "source": chunks[i]["source"],
            "chunk_id": chunks[i]["chunk_id"]
        } for i in idx if sims[i] > 0.01]  # Filtrar scores muito baixos
        
        top_score = float(sims[idx[0]]) if len(idx) else 0.0
        
        return hits, top_score
        
    except Exception as e:
        logger.error(f"[RAG ERROR] Erro na busca: {e}")
        return [], 0.0

def build_context(query: str):
    """Constrói contexto otimizado para a query"""
    hits, top = hybrid_retrieve(query, k=4)
    
    if not hits:
        return "", "", {"top": 0.0, "chars": 0, "nhits": 0}
    
    ctx = "\n\n".join([
        f"[{i+1}] Fonte: {h['source']} | Relevância: {h['score']:.2f}\n{h['text']}"
        for i, h in enumerate(hits)
    ])
    
    cites = "\n".join([f"- {h['source']} (chunk: {h['chunk_id']})" for h in hits])
    stats = {"top": top, "chars": len(ctx), "nhits": len(hits)}
    
    return ctx, cites, stats

def rag_should_fallback(stats: dict) -> bool:
    """Determina se deve fazer fallback para conhecimento geral"""
    if RAG_FALLBACK == "off":
        return False
    return (stats.get("top", 0.0) < RAG_MIN_TOPSCORE) or (stats.get("chars", 0) < RAG_MIN_CTXCHARS)

def anthropic_messages_from_history(history):
    """Converte histórico para formato da API Anthropic"""
    msgs = []
    for m in history:
        if m["role"] in ("user", "assistant"):
            msgs.append({"role": m["role"], "content": [{"type": "text", "text": m["content"]}]})
    return msgs

def _extract_text_from_resp(resp):
    """Extrai texto da resposta da API"""
    parts = []
    try:
        for block in getattr(resp, "content", []):
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
    except Exception:
        parts = [str(resp)]
    return "\n".join(parts) if parts else str(resp)

def chat_reply_with_context(history, model, username="user"):
    """Gera resposta com contexto RAG otimizado"""
    if client is None:
        return "⚠️ Claude indisponível. Configure ANTHROPIC_API_KEY."

    question = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    
    if not question:
        return "Por favor, faça uma pergunta sobre o OriginPro."

    # Buscar contexto relevante
    if HAVE_RAG_DEPS:
        ctx, cites, stats = build_context(question)
    else:
        ctx, cites, stats = ("", "", {"top": 0.0, "chars": 0, "nhits": 0})

    use_rag = bool(ctx) and not rag_should_fallback(stats)
    
    # System prompt otimizado
    if use_rag:
        system = f"""Você é o Origin Software Assistant, um especialista EXCLUSIVO no software OriginPro para análise de dados e criação de gráficos científicos.

REGRAS FUNDAMENTAIS:
1. SEMPRE responda no contexto do OriginPro
2. Use terminologia específica do software (worksheet, workbook, graph window, layer, etc.)
3. Mencione menus e ferramentas específicas do Origin
4. Forneça passos práticos e detalhados
5. Se a pergunta não for sobre o OriginPro, redirecione educadamente

CONTEXTO DA DOCUMENTAÇÃO:
{ctx}

Use o contexto acima para fornecer respostas precisas e específicas."""
    else:
        system = """Você é o Origin Software Assistant, especialista EXCLUSIVO no OriginPro.

IMPORTANTE: Todas as respostas devem ser sobre o OriginPro.

Para qualquer pergunta sobre análise de dados, gráficos ou processamento:
1. Explique como fazer no OriginPro
2. Use menus específicos: Plot menu, Analysis menu, Statistics menu
3. Descreva o processo passo a passo:
   - Importar dados no worksheet
   - Selecionar colunas apropriadas
   - Escolher ferramenta no menu correspondente
   - Customizar usando as opções do Origin

Se a pergunta não for sobre o OriginPro, informe educadamente que você é especializado apenas neste software."""

    try:
        # Fazer chamada à API
        resp = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.3,
            system=system,
            messages=anthropic_messages_from_history(history)
        )
        
        answer = _extract_text_from_resp(resp)
        
        # Estimar tokens para logging
        tokens_estimate = len(question.split()) + len(answer.split())
        log_api_usage(username, model, tokens_estimate)
        
        # Adicionar citações se houver
        if use_rag and cites:
            answer += f"\n\n📚 **Fontes consultadas:**\n{cites}"
        elif not use_rag and HAVE_RAG_DEPS:
            answer += "\n\n💡 _Dica: Adicione documentação PDF do OriginPro para respostas mais específicas._"
        
        return answer
        
    except Exception as e:
        logger.error(f"[CLAUDE ERROR] Erro na API: {e}")
        return f"❌ Erro ao processar: {str(e)}"

# ---------------- CSS Mobile Responsive Completo ----------------

MOBILE_RESPONSIVE_CSS = """
/* ===== RESET E BASE ===== */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body {
    height: 100%;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', sans-serif;
    background: #0e0e0e;
    color: #e0e0e0;
    line-height: 1.6;
    overflow-x: hidden;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: #1a1a1a;
}

::-webkit-scrollbar-thumb {
    background: #3a3a3a;
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: #4a4a4a;
}

/* ===== VIEWPORT RESPONSIVO ===== */
@media (max-width: 768px) {
    html {
        font-size: 14px;
    }
}

/* ===== LOGIN PAGE ===== */
.login-container {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #0e0e0e 0%, #1a1a1a 100%);
    padding: 20px;
}

.login-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 32px;
    width: 100%;
    max-width: 420px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.8);
}

@media (max-width: 480px) {
    .login-card {
        padding: 24px;
        margin: 10px;
    }
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
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
}

.login-subtitle {
    color: #888;
    font-size: 14px;
}

/* ===== MOBILE NAVIGATION ===== */
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
    margin: 12px;
    padding: 12px;
    background: #1e1e1e;
    border: 1px solid #3a3a3a;
    border-radius: 8px;
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

/* ===== MAIN CONTENT ===== */
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

/* ===== STATUS BAR ===== */
.status-bar {
    display: flex;
    gap: 16px;
    padding: 8px 16px;
    background: #0e0e0e;
    border-bottom: 1px solid #2a2a2a;
    font-size: 11px;
    justify-content: space-between;
    flex-wrap: wrap;
}

.status-item {
    display: flex;
    align-items: center;
    gap: 4px;
}

/* ===== KNOWLEDGE BASE ===== */
.kb-section {
    padding: 16px;
    background: #141414;
    border-bottom: 1px solid #2a2a2a;
}

@media (max-width: 768px) {
    .kb-section {
        padding: 12px;
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

/* ===== QUICK ACTIONS ===== */
.quick-actions {
    padding: 16px;
    background: #141414;
    border-radius: 8px;
    margin: 16px;
}

@media (max-width: 768px) {
    .quick-actions {
        margin: 12px;
        padding: 12px;
    }
}

.quick-action-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 8px;
}

.quick-btn {
    padding: 12px 8px;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    color: #888;
    font-size: 12px;
    cursor: pointer;
    text-align: center;
    transition: all 0.2s;
    min-height: 44px;
}

.quick-btn:hover {
    background: #2a2a2a;
    color: #e0e0e0;
    border-color: #3a3a3a;
}

/* ===== CHAT AREA ===== */
.chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    padding-bottom: 140px;
}

@media (max-width: 768px) {
    .chat-area {
        padding: 12px;
        padding-bottom: 160px;
    }
}

.chat-container {
    max-width: 800px;
    margin: 0 auto;
}

/* ===== MESSAGES ===== */
.message {
    margin-bottom: 24px;
    display: flex;
    gap: 12px;
    animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 768px) {
    .message {
        gap: 8px;
        margin-bottom: 16px;
    }
}

.message-avatar {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 600;
    flex-shrink: 0;
}

@media (max-width: 768px) {
    .message-avatar {
        width: 28px;
        height: 28px;
        font-size: 11px;
    }
}

.message.user .message-avatar {
    background: linear-gradient(135deg, #2a3f5f 0%, #3a4f6f 100%);
    color: #61afef;
}

.message.assistant .message-avatar {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.message-content {
    flex: 1;
    min-width: 0;
}

.message-role {
    font-size: 12px;
    font-weight: 600;
    color: #888;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.message-text {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 14px 16px;
    color: #e0e0e0;
    font-size: 14px;
    line-height: 1.6;
    word-wrap: break-word;
    overflow-wrap: break-word;
}

@media (max-width: 768px) {
    .message-text {
        font-size: 13px;
        padding: 12px 14px;
    }
}

.message.user .message-text {
    background: #1e2330;
    border-color: #2a3f5f;
}

.message-text h1, .message-text h2, .message-text h3 {
    margin-top: 16px;
    margin-bottom: 8px;
    color: #fff;
}

.message-text p {
    margin-bottom: 8px;
}

.message-text ul, .message-text ol {
    margin-left: 20px;
    margin-bottom: 8px;
}

.message-text code {
    background: #0e0e0e;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 13px;
}

.message-text pre {
    background: #0e0e0e;
    padding: 12px;
    border-radius: 6px;
    overflow-x: auto;
    margin: 8px 0;
}

/* ===== COMPOSER ===== */
.composer {
    background: #141414;
    border-top: 1px solid #2a2a2a;
    padding: 16px;
    position: fixed;
    bottom: 0;
    left: 260px;
    right: 0;
    z-index: 50;
}

@media (max-width: 768px) {
    .composer {
        left: 0;
        padding: 12px;
        padding-bottom: calc(12px + env(safe-area-inset-bottom));
    }
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

/* ===== FORM ELEMENTS ===== */
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
    min-height: 48px;
    max-height: 150px;
}

@media (max-width: 768px) {
    textarea {
        font-size: 16px;
        padding: 12px;
    }
}

textarea:focus {
    outline: none;
    border-color: #3a3a3a;
    box-shadow: 0 0 0 1px #3a3a3a;
}

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
        font-size: 16px;
    }
}

input:focus {
    outline: none;
    border-color: #3a3a3a;
    box-shadow: 0 0 0 1px #3a3a3a;
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
    cursor: pointer;
}

@media (max-width: 768px) {
    select {
        font-size: 14px;
        min-height: 44px;
    }
}

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

.form-group {
    margin-bottom: 20px;
}

.form-label {
    display: block;
    margin-bottom: 8px;
    font-size: 13px;
    color: #888;
    font-weight: 500;
}

/* ===== BUTTONS ===== */
.btn {
    padding: 10px 20px;
    background: #2a2a2a;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
    transition: all 0.2s;
    white-space: nowrap;
    min-height: 44px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}

.btn:hover {
    background: #3a3a3a;
    border-color: #4a4a4a;
    transform: translateY(-1px);
}

.btn:active {
    transform: translateY(0);
}

.btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
    color: white;
    font-weight: 600;
}

.btn-primary:hover {
    background: linear-gradient(135deg, #5a67d8 0%, #6a3f92 100%);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
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

.btn-small {
    padding: 6px 12px;
    font-size: 12px;
    min-height: 32px;
}

/* ===== ADMIN PANEL ===== */
.admin-panel {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 20px;
    margin: 16px;
    display: none;
}

@media (max-width: 768px) {
    .admin-panel {
        margin: 12px;
        padding: 16px;
    }
}

.admin-panel.show {
    display: block;
    animation: slideDown 0.3s ease;
}

@keyframes slideDown {
    from {
        opacity: 0;
        transform: translateY(-10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.admin-form-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
}

@media (max-width: 768px) {
    .admin-form-row {
        grid-template-columns: 1fr;
    }
}

.user-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 8px;
    font-size: 13px;
    transition: all 0.2s;
}

.user-card:hover {
    border-color: #3a3a3a;
    transform: translateX(2px);
}

/* ===== ALERTS ===== */
.alert {
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 16px;
    font-size: 14px;
    animation: slideIn 0.3s ease;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateX(-10px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
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

.alert-info {
    background: rgba(97, 175, 239, 0.1);
    color: #61afef;
    border: 1px solid rgba(97, 175, 239, 0.3);
}

/* ===== LOADING ANIMATION ===== */
.typing-dots {
    display: flex;
    gap: 4px;
    padding: 12px;
}

.typing-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    animation: typing 1.4s infinite;
}

@keyframes typing {
    0%, 60%, 100% { 
        opacity: 0.2; 
        transform: scale(0.8);
    }
    30% { 
        opacity: 1;
        transform: scale(1.2);
    }
}

.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

/* ===== WELCOME MESSAGE ===== */
.welcome-message {
    text-align: center;
    padding: 40px 20px;
    color: #666;
}

@media (max-width: 768px) {
    .welcome-message {
        padding: 30px 15px;
    }
}

.welcome-title {
    font-size: 24px;
    margin-bottom: 20px;
    color: #fff;
    font-weight: 600;
}

@media (max-width: 768px) {
    .welcome-title {
        font-size: 20px;
    }
}

.welcome-features {
    max-width: 500px;
    margin: 20px auto;
    text-align: left;
    font-size: 14px;
    line-height: 2;
    color: #888;
}

/* ===== EXPORT OPTIONS ===== */
.export-options {
    display: flex;
    gap: 8px;
    padding: 8px;
    justify-content: flex-end;
    flex-wrap: wrap;
}

/* ===== TOOLTIPS ===== */
.tooltip {
    position: relative;
}

.tooltip::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    background: #2a2a2a;
    color: #e0e0e0;
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 12px;
    white-space: nowrap;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s;
    z-index: 1000;
}

.tooltip:hover::after {
    opacity: 1;
}

/* ===== UTILITY CLASSES ===== */
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

.text-center { text-align: center; }
.text-right { text-align: right; }
.mt-1 { margin-top: 8px; }
.mt-2 { margin-top: 16px; }
.mb-1 { margin-bottom: 8px; }
.mb-2 { margin-bottom: 16px; }

/* ===== SAFE AREAS iOS ===== */
@supports (padding: max(0px)) {
    .composer {
        padding-bottom: max(16px, env(safe-area-inset-bottom));
    }
    
    @media (max-width: 768px) {
        .composer {
            padding-bottom: max(12px, env(safe-area-inset-bottom));
        }
    }
}

/* ===== TOUCH TARGETS ===== */
@media (hover: none) {
    .btn, .session-item, .new-session-btn, .quick-btn {
        min-height: 44px;
    }
}

/* ===== LANDSCAPE MODE ===== */
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

/* ===== DARK MODE SUPPORT ===== */
@media (prefers-color-scheme: light) {
    /* Suporte futuro para tema claro */
}

/* ===== PRINT STYLES ===== */
@media print {
    .sidebar, .composer, .header-bar, .mobile-header {
        display: none;
    }
    
    .main-content {
        margin-left: 0;
    }
    
    .message {
        break-inside: avoid;
    }
}
"""

# ---------------- JavaScript para Mobile ----------------

MOBILE_JS = """
<script>
// Controle da sidebar mobile
function toggleSidebar() {
    const sidebar = document.getElementById('main-sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    
    if (sidebar && overlay) {
        sidebar.classList.toggle('open');
        overlay.classList.toggle('show');
        
        // Prevenir scroll do body quando sidebar aberta
        if (sidebar.classList.contains('open')) {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = '';
        }
    }
}

document.addEventListener('DOMContentLoaded', function() {
    // Fechar sidebar ao clicar no overlay
    const overlay = document.getElementById('sidebar-overlay');
    if (overlay) {
        overlay.addEventListener('click', toggleSidebar);
    }
    
    // Auto-resize do textarea
    const textarea = document.querySelector('textarea[id*="prompt"]');
    if (textarea) {
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 150) + 'px';
        });
    }
    
    // Scroll automático para última mensagem
    function scrollToBottom() {
        const chatArea = document.querySelector('.chat-area');
        if (chatArea) {
            setTimeout(() => {
                chatArea.scrollTop = chatArea.scrollHeight;
            }, 100);
        }
    }
    
    // Observer para novas mensagens
    const observer = new MutationObserver(scrollToBottom);
    const chatContainer = document.querySelector('.chat-container');
    if (chatContainer) {
        observer.observe(chatContainer, { childList: true, subtree: true });
    }
    
    // Prevenir zoom no iOS quando focando inputs
    if (/iPhone|iPad|iPod/.test(navigator.userAgent)) {
        document.addEventListener('touchstart', function(e) {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                e.target.style.fontSize = '16px';
            }
        });
    }
    
    // Detectar teclado virtual e ajustar layout
    let viewportHeight = window.innerHeight;
    window.addEventListener('resize', function() {
        const currentHeight = window.innerHeight;
        const keyboardHeight = viewportHeight - currentHeight;
        
        const composer = document.querySelector('.composer');
        if (composer) {
            if (keyboardHeight > 100) {
                // Teclado aberto
                composer.style.position = 'absolute';
            } else {
                // Teclado fechado
                composer.style.position = 'fixed';
            }
        }
    });
    
    // Vibração sutil ao enviar mensagem (mobile)
    window.vibrateFeedback = function() {
        if ('vibrate' in navigator) {
            navigator.vibrate(10);
        }
    };
});

// Toggle do painel admin
function toggleAdminPanel() {
    const panel = document.querySelector('.admin-panel');
    if (panel) {
        panel.classList.toggle('show');
    }
}

// Toggle sidebar via botão
Shiny.addCustomMessageHandler('toggle-sidebar', function(message) {
    toggleSidebar();
});

// Scroll to bottom
Shiny.addCustomMessageHandler('scroll-to-bottom', function(message) {
    setTimeout(() => {
        const chatArea = document.querySelector('.chat-area');
        if (chatArea) {
            chatArea.scrollTop = chatArea.scrollHeight;
        }
    }, 100);
});
</script>
"""

# ---------------- Interface Principal ----------------

app_ui = ui.page_fluid(
    ui.tags.style(MOBILE_RESPONSIVE_CSS),
    ui.tags.script(MOBILE_JS),
    ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no"),
    ui.tags.meta(name="apple-mobile-web-app-capable", content="yes"),
    ui.tags.meta(name="apple-mobile-web-app-status-bar-style", content="black-translucent"),
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
        """Adiciona mensagem ao histórico"""
        history.set(history() + [{"role": role, "content": content}])
        # Trigger scroll to bottom via JavaScript
        session.send_custom_message('scroll-to-bottom', {})
    
    @output
    @render.ui
    def main_content():
        """Renderiza interface principal"""
        if not authenticated():
            # LOGIN PAGE
            return ui.div({"class": "login-container"},
                ui.div({"class": "login-card"},
                    ui.div({"class": "login-header"},
                        ui.h1({"class": "login-title"}, 
                            "🚀 Origin Assistant"
                        ),
                        ui.p({"class": "login-subtitle"}, 
                            "Especialista em OriginPro para Análise de Dados"
                        )
                    ),
                    
                    ui.output_ui("login_feedback"),
                    
                    ui.div({"class": "form-group"},
                        ui.span({"class": "form-label"}, "Usuário"),
                        ui.input_text("username", None, 
                            placeholder="Digite seu usuário",
                            autocomplete="username"
                        )
                    ),
                    
                    ui.div({"class": "form-group"},
                        ui.span({"class": "form-label"}, "Senha"),
                        ui.input_password("password", None, 
                            placeholder="Digite sua senha",
                            autocomplete="current-password"
                        )
                    ),
                    
                    ui.input_action_button("login_btn", "Entrar", 
                        class_="btn btn-primary", 
                        style="width: 100%; margin-bottom: 16px;"
                    ),
                    
                    ui.hr({"style": "margin: 20px 0; border-color: #2a2a2a;"}),
                    
                    ui.div({"style": "text-align: center; color: #666; font-size: 12px;"},
                        "🔒 Acesso seguro com PBKDF2",
                        ui.br(),
                        "Demo: admin / admin123"
                    )
                )
            )
        else:
            # APP PRINCIPAL
            return ui.TagList(
                # Mobile Header
                ui.div({"class": "mobile-header"},
                    ui.div({"class": "mobile-title"},
                        "🚀 Origin Assistant"
                    ),
                    ui.div({"style": "display: flex; gap: 8px;"},
                        ui.input_action_button("toggle_sidebar", "☰", 
                            class_="mobile-menu-btn",
                            onclick="toggleSidebar()"
                        ),
                        ui.input_action_button("logout_btn_mobile", "Sair",
                            class_="mobile-menu-btn"
                        )
                    )
                ),
                
                # Sidebar Overlay
                ui.div({"class": "sidebar-overlay", "id": "sidebar-overlay"}),
                
                # Sidebar
                ui.div({"class": "sidebar", "id": "main-sidebar"},
                    ui.div({"class": "sidebar-header"},
                        ui.div({"class": "sidebar-title"},
                            "🚀 Origin Assistant"
                        ),
                        ui.div({"class": "sidebar-user"},
                            f"👤 {current_user()}"
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
                        {"style": "margin-top: auto; padding: 12px; border-top: 1px solid #2a2a2a;"},
                        ui.input_action_button(
                            "show_admin", 
                            "⚙️ Admin",
                            class_="btn btn-small", 
                            style="width: 100%; margin-bottom: 8px;"
                        ) if is_admin() else ui.TagList(),
                        ui.input_action_button(
                            "logout_btn", 
                            "🚪 Logout",
                            class_="btn btn-logout hidden-mobile", 
                            style="width: 100%;"
                        )
                    )
}

.mobile-menu-btn:hover {
    background: #1a1a1a;
    border-color: #4a4a4a;
}

/* ===== SIDEBAR ===== */
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
        box-shadow: 2px 0 20px rgba(0,0,0,0.8);
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
    backdrop-filter: blur(2px);
}

@media (max-width: 768px) {
    .sidebar-overlay.show {
        display: block;
    }
}

.sidebar-header {
    padding: 20px;
    border-bottom: 1px solid #2a2a2a;
    background: #0e0e0e;
}

.sidebar-title {
    font-size: 18px;
    font-weight: 600;
    color: #fff;
    display: flex;
    align-items: center;
    gap: 8px;
}

.sidebar-user {
    font-size: 12px;
    color: #666;
    margin-top: 8px;
}

.sessions-list {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
}

.session-item {
    padding: 12px 16px;
    margin-bottom: 4px;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    color: #888;
    cursor: pointer;
    font-size: 14px;
    transition: all 0.2s;







