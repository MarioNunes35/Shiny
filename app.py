# app.py – Shiny (Python) chat + RAG de PDFs com persistência S3 (Cloudflare R2) e FALLBACK automático para Claude
# Design melhorado com espaçamento otimizado e visual mais moderno
#
# ENV necessárias no Posit Connect (Settings → Environment):
#   ANTHROPIC_API_KEY
#   S3_ENDPOINT_URL=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
#   S3_BUCKET=origin-assistant-cache
#   AWS_ACCESS_KEY_ID=<Access Key ID>
#   AWS_SECRET_ACCESS_KEY=<Secret Access Key>
#   AWS_DEFAULT_REGION=auto        # opcional
#   S3_PREFIX=osa-cache/           # opcional (terminar com '/')
#
#   RAG_FALLBACK=auto              # auto | off
#   RAG_MIN_TOPSCORE=0.18          # limiar do score da 1ª evidência (0–1)
#   RAG_MIN_CTXCHARS=300           # mínimo de caracteres do contexto
from shiny import App, ui, render, reactive
from dotenv import load_dotenv
import os, re, json, hashlib
from pathlib import Path

# ---------------- Env / Claude ----------------
load_dotenv()
API_KEY = os.getenv("ANTHROPIC_API_KEY")
HAS_KEY = bool(API_KEY)

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

# ---------------- Thresholds / fallback ----------------
RAG_FALLBACK = (os.getenv("RAG_FALLBACK", "auto") or "auto").lower()
RAG_MIN_TOPSCORE = float(os.getenv("RAG_MIN_TOPSCORE", "0.18"))
RAG_MIN_CTXCHARS = int(os.getenv("RAG_MIN_CTXCHARS", "300"))

def rag_should_fallback(stats: dict) -> bool:
    if RAG_FALLBACK == "off":
        return False
    return (stats.get("top", 0.0) < RAG_MIN_TOPSCORE) or (stats.get("chars", 0) < RAG_MIN_CTXCHARS)

# ---------------- RAG deps ----------------
HAVE_RAG_DEPS = True
try:
    from pypdf import PdfReader
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from joblib import dump, load
except Exception:
    HAVE_RAG_DEPS = False
    PdfReader = None
    TfidfVectorizer = None
    cosine_similarity = None
    dump = load = None

# ---------------- S3 / R2 deps ----------------
HAVE_S3 = True
try:
    import boto3
    from botocore.exceptions import ClientError
except Exception:
    HAVE_S3 = False
    boto3 = None
    ClientError = Exception

print(f"[BOOT] RAG={HAVE_RAG_DEPS} | S3={HAVE_S3} | Claude={(client is not None)} | FB={RAG_FALLBACK}/{RAG_MIN_TOPSCORE}/{RAG_MIN_CTXCHARS}")

# ---------------- Caminhos do índice ----------------
DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"
CHUNKS_JSON = CACHE_DIR / "chunks.json"
VECTORIZER_JOBLIB = CACHE_DIR / "tfidf_vectorizer.joblib"
MATRIX_JOBLIB = CACHE_DIR / "tfidf_matrix.joblib"

for d in (DATA_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------- Helpers S3/R2 ----------------
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
    region = os.getenv("AWS_DEFAULT_REGION")
    if region and region.lower() == "auto":
        region = None
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=key,
            aws_secret_access_key=secret,
            region_name=region,
        )
        return {"client": s3, "bucket": bucket, "prefix": prefix}
    except Exception as e:
        print(f"[S3] erro ao criar cliente: {e}")
        return None

_S3 = _s3_conf()

def _key(name: str) -> str:
    p = _S3["prefix"] if (_S3 and _S3.get("prefix")) else ""
    return f"{p}{name}"

def s3_pull_cache_if_needed():
    if not _S3:
        return False
    if CHUNKS_JSON.exists() and VECTORIZER_JOBLIB.exists() and MATRIX_JOBLIB.exists():
        print("[S3] cache local presente.")
        return False
    client = _S3["client"]; bucket = _S3["bucket"]
    any_ok = False
    for key, local in [
        ("chunks.json", CHUNKS_JSON),
        ("tfidf_vectorizer.joblib", VECTORIZER_JOBLIB),
        ("tfidf_matrix.joblib", MATRIX_JOBLIB),
    ]:
        try:
            client.download_file(bucket, _key(key), str(local))
            print(f"[S3] baixado s3://{bucket}/{_key(key)} -> {local}")
            any_ok = True
        except ClientError as e:
            code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            if code in ("404","NoSuchKey","NotFound"):
                print(f"[S3] ausente: s3://{bucket}/{_key(key)}")
            else:
                print(f"[S3] erro ao baixar {key}: {e}")
        except Exception as e:
            print(f"[S3] erro genérico ao baixar {key}: {e}")
    return any_ok

def s3_push_cache():
    if not _S3:
        return False
    if not (CHUNKS_JSON.exists() and VECTORIZER_JOBLIB.exists() and MATRIX_JOBLIB.exists()):
        print("[S3] cache local incompleto; não enviado.")
        return False
    client = _S3["client"]; bucket = _S3["bucket"]
    ok_all = True
    for key, local in [
        ("chunks.json", CHUNKS_JSON),
        ("tfidf_vectorizer.joblib", VECTORIZER_JOBLIB),
        ("tfidf_matrix.joblib", MATRIX_JOBLIB),
    ]:
        try:
            extra = {"ContentType": "application/json"} if local.suffix == ".json" else {"ContentType": "application/octet-stream"}
            client.upload_file(str(local), bucket, _key(key), ExtraArgs=extra)
            print(f"[S3] enviado {local} -> s3://{bucket}/{_key(key)}")
        except Exception as e:
            ok_all = False
            print(f"[S3] erro ao enviar {local}: {e}")
    return ok_all

try:
    s3_pull_cache_if_needed()
except Exception as e:
    print(f"[S3] falha ao restaurar cache no boot: {e}")

# ---------------- RAG helpers ----------------
def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    txt = []
    for pg in reader.pages:
        txt.append(pg.extract_text() or "")
    return "\n".join(txt)

def chunk_text(text: str, max_chars=900, overlap=220):
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

def add_pdfs_to_index(file_paths: list[Path]):
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
        except Exception as e:
            print(f"[RAG] erro ao processar {p.name}: {e}")
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

# ---------------- Chat helpers ----------------
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

# ---------------- CSS / UI ----------------
CSS = """
:root{
  --bg:#F7F7F8; --panel:#FFFFFF;
  --bubble-user:#E5F2FF; --bubble-assistant:#F7F7F8;
  --border:#E2E2E3; --text:#0F172A; --muted:#6B7280;
  --accent:#10A37F; --accent-hover:#0E8B6F;
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
html,body{height:100%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;}
body{
  background: var(--bg);
  color: var(--text);
  transition: background 0.3s ease, color 0.3s ease;
}

/* Layout */
.app-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

/* Header */
.header{
  background: var(--panel);
  border-bottom: 1px solid var(--border);
  padding: 16px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: var(--shadow);
}
.header .logo-section {
  display: flex;
  align-items: center;
  gap: 12px;
}
.header .logo {
  font-size: 28px;
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
}
.header h1 {
  font-size: 20px;
  font-weight: 600;
  margin: 0;
  background: linear-gradient(135deg, var(--accent), var(--accent-hover));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.header .subtitle {
  font-size: 12px;
  color: var(--muted);
  margin-top: 2px;
}
.header .controls {
  display: flex;
  gap: 12px;
  align-items: center;
}

/* Status badge */
.status-badge {
  padding: 6px 12px;
  border-radius: 20px;
  font-size: 13px;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 6px;
  background: var(--bubble-assistant);
  border: 1px solid var(--border);
}
.status-badge.ready { color: #10b981; }
.status-badge.warning { color: #f59e0b; }
.status-badge.error { color: #ef4444; }

/* Knowledge base */
.kb-section {
  padding: 16px 24px;
  background: var(--panel);
  border-bottom: 1px solid var(--border);
}
.kb-card {
  background: var(--bubble-assistant);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
  box-shadow: var(--shadow);
}
.kb-card h3 {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.kb-info {
  font-size: 13px;
  color: var(--muted);
  margin-top: 8px;
  padding: 8px;
  background: var(--bg);
  border-radius: 6px;
}

/* Chat area */
.chat-wrapper {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  scroll-behavior: smooth;
}
.chat-container {
  max-width: 900px;
  margin: 0 auto;
}

/* Messages */
.message {
  display: flex;
  gap: 16px;
  margin-bottom: 24px;
  animation: fadeIn 0.3s ease;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.message.user {
  flex-direction: row-reverse;
}

.avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 14px;
  flex-shrink: 0;
  box-shadow: var(--shadow);
}
.message.assistant .avatar {
  background: linear-gradient(135deg, var(--accent), var(--accent-hover));
  color: white;
}
.message.user .avatar {
  background: linear-gradient(135deg, #667EEA, #764BA2);
  color: white;
}

.message-content {
  flex: 1;
  max-width: 75%;
}
.message.user .message-content {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
}

.bubble {
  background: var(--bubble-assistant);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 12px 16px;
  box-shadow: var(--shadow);
}
.message.user .bubble {
  background: var(--bubble-user);
}

.role {
  font-size: 12px;
  font-weight: 600;
  color: var(--muted);
  margin-bottom: 4px;
}
.message.user .role {
  text-align: right;
}

/* ESPAÇAMENTO MELHORADO */
.content {
  font-size: 15px;
  line-height: 1.6;
  color: var(--text);
}
.content p {
  margin: 0.6rem 0;
}
.content p:first-child {
  margin-top: 0;
}
.content p:last-child {
  margin-bottom: 0;
}
.content ul, .content ol {
  margin: 0.8rem 0;
  padding-left: 1.5rem;
}
.content li {
  margin: 0.4rem 0;
  line-height: 1.6;
}
.content h1, .content h2, .content h3 {
  margin: 1rem 0 0.5rem;
  font-weight: 600;
  line-height: 1.3;
}
.content h1 { font-size: 1.5rem; }
.content h2 { font-size: 1.3rem; }
.content h3 { font-size: 1.1rem; }
.content code {
  background: var(--bg);
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 0.9em;
  font-family: 'Monaco', 'Menlo', monospace;
}
.content pre {
  background: var(--bg);
  padding: 12px;
  border-radius: 8px;
  overflow-x: auto;
  margin: 0.8rem 0;
}
.content blockquote {
  border-left: 3px solid var(--accent);
  padding-left: 16px;
  margin: 0.8rem 0;
  color: var(--muted);
}
.content hr {
  border: none;
  border-top: 1px solid var(--border);
  margin: 1rem 0;
}
.content a {
  color: var(--accent);
  text-decoration: none;
  border-bottom: 1px solid transparent;
  transition: border-color 0.2s;
}
.content a:hover {
  border-bottom-color: var(--accent);
}
.content strong {
  font-weight: 600;
}

/* Typing indicator */
.typing-indicator {
  display: flex;
  gap: 4px;
  padding: 8px;
}
.typing-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--muted);
  animation: typing 1.4s infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes typing {
  0%, 60%, 100% { opacity: 0.3; }
  30% { opacity: 1; }
}

/* Composer */
.composer-wrapper {
  background: var(--panel);
  border-top: 1px solid var(--border);
  padding: 20px 24px;
  box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
}
.composer {
  max-width: 900px;
  margin: 0 auto;
  display: flex;
  gap: 16px;
  align-items: flex-end;
}
.input-group {
  flex: 1;
}

/* Textarea */
textarea.form-control {
  width: 100%;
  background: var(--bg);
  color: var(--text);
  border: 2px solid var(--border);
  border-radius: 12px;
  padding: 14px 16px;
  min-height: 80px;
  max-height: 200px;
  resize: vertical;
  font-size: 15px;
  line-height: 1.5;
  transition: all 0.2s ease;
  font-family: inherit;
}
textarea.form-control:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(25,195,125,0.1);
}
textarea.form-control::placeholder {
  color: var(--muted);
  opacity: 0.7;
}

/* Controls */
.controls-row {
  display: flex;
  gap: 12px;
  margin-top: 12px;
  align-items: center;
}

select.form-select {
  flex: 1;
  background: var(--bg);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
}
select.form-select:hover {
  border-color: var(--accent);
}
select.form-select:focus {
  outline: none;
  border-color: var(--accent);
}

/* Buttons */
.btn {
  padding: 8px 20px;
  border-radius: 8px;
  border: none;
  font-weight: 500;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s ease;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.btn-primary {
  background: linear-gradient(135deg, var(--accent), var(--accent-hover));
  color: white;
  min-width: 100px;
  height: 48px;
  font-size: 15px;
}
.btn-primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(25,195,125,0.3);
}
.btn-primary:active {
  transform: translateY(0);
}
.btn-secondary {
  background: var(--bubble-assistant);
  color: var(--text);
  border: 1px solid var(--border);
}
.btn-secondary:hover {
  background: var(--bg);
}

/* Responsive */
@media (max-width: 768px) {
  .header { padding: 12px 16px; }
  .header h1 { font-size: 18px; }
  .chat-wrapper { padding: 16px; }
  .message-content { max-width: 85%; }
  .composer-wrapper { padding: 16px; }
  .composer { flex-direction: column; align-items: stretch; }
  .btn-primary { width: 100%; }
}
"""

app_ui = ui.page_fluid(
    ui.tags.style(CSS),
    ui.tags.script("""
      Shiny.addCustomMessageHandler('set_theme', (theme) => {
        document.documentElement.setAttribute('data-theme', theme);
        try { localStorage.setItem('osa-theme', theme); } catch(e){}
      });
      (function(){
        let saved = 'dark';
        try {
          saved = localStorage.getItem('osa-theme') ||
                  (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
        } catch(e) {}
        document.documentElement.setAttribute('data-theme', saved);
        Shiny.setInputValue('theme', saved, {priority:'event'});
      })();
    """),
    
    ui.div({"class": "app-container"},
        # Header
        ui.div({"class": "header"},
            ui.div({"class": "logo-section"},
                ui.span({"class": "logo"}, "🚀"),
                ui.div(
                    ui.h1("Origin Software Assistant"),
                    ui.div({"class": "subtitle"}, "Chat inteligente com RAG • Powered by Claude")
                )
            ),
            ui.div({"class": "controls"},
                ui.output_ui("status_badge"),
                ui.input_select("theme", None, 
                    {"dark": "🌙 Escuro", "light": "☀️ Claro"}, 
                    selected="dark",
                    width="auto"
                )
            )
        ),
        
        # Knowledge Base
        ui.div({"class": "kb-section"},
            ui.div({"class": "kb-card"},
                ui.h3("📚 Base de Conhecimento"),
                ui.input_file("docs", "Adicionar PDFs", 
                    multiple=True, 
                    accept=[".pdf"],
                    width="auto"
                ),
                ui.div({"class": "kb-info"},
                    ui.output_text("kb_status")
                )
            )
        ),
        
        # Chat
        ui.div({"class": "chat-wrapper"},
            ui.div({"class": "chat-container"},
                ui.output_ui("chat_thread")
            )
        ),
        
        # Composer
        ui.div({"class": "composer-wrapper"},
            ui.div({"class": "composer"},
                ui.div({"class": "input-group"},
                    ui.input_text_area("prompt", None, 
                        placeholder="Digite sua mensagem... (Shift+Enter para nova linha)",
                        rows=3,
                        width="100%"
                    ),
                    ui.div({"class": "controls-row"},
                        ui.input_select("model", None, 
                            {
                                "claude-3-haiku-20240307": "⚡ Claude 3 Haiku (rápido)",
                                "claude-3-5-sonnet-20240620": "✨ Claude 3.5 Sonnet (avançado)"
                            }, 
                            selected="claude-3-haiku-20240307",
                            width="auto"
                        ),
                        ui.input_action_button("clear", "🗑️ Limpar", 
                            class_="btn btn-secondary"
                        )
                    )
                ),
                ui.input_action_button("send", "Enviar →", 
                    class_="btn btn-primary"
                )
            )
        )
    )
)

def server(input, output, session):
    history = reactive.Value([])
    typing = reactive.Value(False)

    def push(role, content):
        history.set(history() + [{"role": role, "content": content}])

    @render.ui
    def status_badge():
        if HAS_KEY and client is not None:
            return ui.div({"class": "status-badge ready"}, 
                "●", " Claude pronto"
            )
        elif HAS_KEY and client is None and Anthropic is None:
            return ui.div({"class": "status-badge warning"}, 
                "●", " Falta 'anthropic'"
            )
        else:
            return ui.div({"class": "status-badge error"}, 
                "●", " Sem API Key"
            )

    @render.text
    def kb_status():
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

    @render.ui
    def chat_thread():
        items = []
        for m in history():
            is_assistant = m["role"] == "assistant"
            
            items.append(
                ui.div(
                    {"class": f"message {'assistant' if is_assistant else 'user'}"},
                    ui.div({"class": "avatar"}, 
                        "OA" if is_assistant else "V"
                    ),
                    ui.div({"class": "message-content"},
                        ui.div({"class": "role"}, 
                            "Origin Assistant" if is_assistant else "Você"
                        ),
                        ui.div({"class": "bubble"},
                            ui.div({"class": "content"}, 
                                ui.markdown(m["content"])
                            )
                        )
                    )
                )
            )
        
        # Typing indicator
        if typing():
            items.append(
                ui.div({"class": "message assistant"},
                    ui.div({"class": "avatar"}, "OA"),
                    ui.div({"class": "message-content"},
                        ui.div({"class": "role"}, "Origin Assistant"),
                        ui.div({"class": "bubble"},
                            ui.div({"class": "typing-indicator"},
                                ui.div({"class": "typing-dot"}),
                                ui.div({"class": "typing-dot"}),
                                ui.div({"class": "typing-dot"})
                            )
                        )
                    )
                )
            )
        
        return ui.TagList(*items) if items else ui.div(
            {"style": "text-align: center; color: var(--muted); padding: 40px;"},
            "💬 Inicie uma conversa enviando uma mensagem"
        )

    @reactive.Effect
    @reactive.event(input.clear)
    def _clear():
        history.set([])
        ui.update_text_area("prompt", value="")
        ui.notification_show("Chat limpo com sucesso", type="message", duration=2)

    # Theme handler
    @reactive.Effect
    async def _theme_apply():
        theme = input.theme() or "dark"
        await session.send_custom_message("set_theme", theme)

    # Keyboard shortcuts and auto-scroll
    ui.tags.script("""
        // Enter to send (Shift+Enter for new line)
        document.addEventListener('keydown', (e) => {
            if(e.target.id === 'prompt' && e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                document.getElementById('send').click();
            }
        });
        
        // Auto-scroll to bottom on new messages
        const scrollToBottom = () => {
            const wrapper = document.querySelector('.chat-wrapper');
            if(wrapper) {
                setTimeout(() => {
                    wrapper.scrollTop = wrapper.scrollHeight;
                }, 100);
            }
        };
        
        // Observer for new messages
        new MutationObserver(scrollToBottom).observe(
            document.body, 
            {childList: true, subtree: true}
        );
        
        // Auto-resize textarea
        const autoResize = (el) => {
            if (!el) return;
            el.style.height = 'auto';
            const newHeight = Math.min(el.scrollHeight, 200);
            el.style.height = newHeight + 'px';
        };
        
        document.addEventListener('input', (e) => {
            if (e.target && e.target.id === 'prompt') {
                autoResize(e.target);
            }
        });
        
        // Initial resize
        setTimeout(() => {
            const promptEl = document.getElementById('prompt');
            if(promptEl) autoResize(promptEl);
        }, 500);
    """)

    @reactive.Effect
    @reactive.event(input.docs)
    def _ingest_pdfs():
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







