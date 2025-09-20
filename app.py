# app.py — Chat UI + Light/Dark + PDF RAG cache (TF-IDF) — Design minimalista Claude
from shiny import App, ui, render, reactive
from dotenv import load_dotenv
import os, traceback, re, json, hashlib
from pathlib import Path

# -------- Env / Anthropic --------
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

# -------- Optional deps for RAG (graceful fallback) --------
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

print("RAG deps disponíveis?", HAVE_RAG_DEPS)

# -------- RAG cache paths --------
DATA_DIR = Path("data")
CACHE_DIR = DATA_DIR / "cache"
CHUNKS_JSON = CACHE_DIR / "chunks.json"
VECTORIZER_JOBLIB = CACHE_DIR / "tfidf_vectorizer.joblib"
MATRIX_JOBLIB = CACHE_DIR / "tfidf_matrix.joblib"

for d in (DATA_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# -------- RAG helpers --------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

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

def add_pdfs_to_index(file_paths: list[Path]):
    if not HAVE_RAG_DEPS:
        return 0
    chunks, _, _ = load_index()
    new_chunks = []
    for p in file_paths:
        try:
            doc_id = sha256_file(p)
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
        return []
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    top_idx = sims.argsort()[::-1][:k]
    results = []
    for i in top_idx:
        c = chunks[i]
        results.append({
            "score": float(sims[i]),
            "text": c["text"],
            "source": c["source"],
            "chunk_id": c["chunk_id"]
        })
    return results

def build_context(query: str):
    hits = retrieve(query, k=4)
    ctx = "\n\n".join(
        [f"[{i+1}] ({h['source']} • {h['chunk_id']} • score={h['score']:.3f})\n{h['text']}" for i, h in enumerate(hits)]
    )
    cites = "\n".join([f"- {h['source']} ({h['chunk_id']})" for h in hits])
    return ctx, cites

# -------- Chat helpers --------
def anthropic_messages_from_history(history):
    msgs = []
    for m in history:
        role = m["role"]
        if role not in ("user","assistant"):
            continue
        msgs.append({"role": role, "content":[{"type":"text","text": m["content"]}]})
    return msgs

def chat_reply_with_context(history, model):
    if client is None:
        return "Configuração necessária: defina ANTHROPIC_API_KEY e instale 'anthropic'."
    
    # Última pergunta do usuário
    question = next((m["content"] for m in reversed(history) if m["role"]=="user"), "")
    ctx, cites = build_context(question) if HAVE_RAG_DEPS else ("", "")
    
    # Se há contexto dos PDFs, usa o sistema RAG
    if ctx.strip():
        system = (
            "Você é o Origin Software Assistant. Use o CONTEXTO dos documentos para responder. "
            "Se a informação estiver disponível no contexto, responda baseado nele. "
            "Se não estiver no contexto, informe que a informação não foi encontrada nos documentos "
            "e ofereça uma resposta baseada no seu conhecimento geral.\n\n"
            f"=== CONTEXTO DOS DOCUMENTOS ===\n{ctx}\n=== FIM DO CONTEXTO ==="
        )
    else:
        # Se não há contexto dos PDFs, responde normalmente com conhecimento do Claude
        system = (
            "Você é o Origin Software Assistant. Responda de forma útil e precisa usando "
            "seu conhecimento. Seja claro, direto e forneça informações práticas."
        )
    
    resp = client.messages.create(
        model=model, max_tokens=1200, temperature=0.3,
        system=system, messages=anthropic_messages_from_history(history)
    )
    
    parts = []
    try:
        for block in getattr(resp, "content", []):
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
    except Exception:
        parts = [str(resp)]
    
    answer = "\n".join(parts) if parts else str(resp)
    
    # Adiciona fontes apenas se usou documentos
    if cites and ctx.strip():
        answer += "\n\n---\n**📚 Fontes dos documentos:**\n" + cites
    
    return answer

# ----------------- CSS Minimalista - Cores do Claude -----------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

:root {
  --bg: #f8fafc;
  --surface: #ffffff;
  --border: #e5e7eb;
  --text-primary: #1f2937;
  --text-secondary: #6b7280;
  --text-muted: #9ca3af;
  --orange: #ff7849;
  --orange-hover: #ff6332;
  --orange-light: #fff5f2;
  --gray-50: #f9fafb;
  --gray-100: #f3f4f6;
  --gray-900: #111827;
  --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
}

[data-theme='dark'] {
  --bg: #0f1419;
  --surface: #1a1f2e;
  --border: #2d3748;
  --text-primary: #e2e8f0;
  --text-secondary: #a0aec0;
  --text-muted: #718096;
  --gray-50: #1a202c;
  --gray-100: #2d3748;
  --gray-900: #f7fafc;
  --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.3);
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.4);
}

* {
  box-sizing: border-box;
}

html, body {
  height: 100%;
  margin: 0;
  padding: 0;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  background: var(--bg);
  color: var(--text-primary);
  font-size: 14px;
  line-height: 1.5;
}

.header {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 16px 24px;
  position: sticky;
  top: 0;
  z-index: 10;
}

.header-content {
  max-width: 800px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.header h1 {
  font-size: 18px;
  font-weight: 600;
  margin: 0;
  color: var(--text-primary);
}

.header-controls {
  display: flex;
  align-items: center;
  gap: 12px;
}

.status-badge {
  font-size: 12px;
  padding: 4px 8px;
  border-radius: 6px;
  font-weight: 500;
}

.status-ok {
  background: #dcfce7;
  color: #16a34a;
}

.status-warn {
  background: #fef3c7;
  color: #d97706;
}

.status-error {
  background: #fee2e2;
  color: #dc2626;
}

[data-theme='dark'] .status-ok {
  background: #064e3b;
  color: #34d399;
}

[data-theme='dark'] .status-warn {
  background: #451a03;
  color: #fbbf24;
}

[data-theme='dark'] .status-error {
  background: #450a0a;
  color: #f87171;
}

.main-content {
  max-width: 800px;
  margin: 0 auto;
  padding: 0 24px;
}

.kb-section {
  margin: 24px 0;
}

.kb-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px;
  box-shadow: var(--shadow-sm);
}

.kb-header {
  font-weight: 600;
  margin-bottom: 16px;
  color: var(--text-primary);
}

.file-input-wrapper {
  position: relative;
  display: inline-block;
}

.file-input-wrapper input[type="file"] {
  position: absolute;
  opacity: 0;
  width: 100%;
  height: 100%;
  cursor: pointer;
}

.file-input-label {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  background: var(--gray-50);
  border: 1px dashed var(--border);
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
  color: var(--text-secondary);
  font-size: 14px;
}

.file-input-label:hover {
  background: var(--gray-100);
  border-color: var(--orange);
}

.kb-status {
  margin-top: 12px;
  font-size: 13px;
  color: var(--text-muted);
}

.chat-container {
  margin: 24px 0 180px;
  min-height: 400px;
}

.message {
  display: flex;
  gap: 12px;
  margin-bottom: 24px;
  align-items: flex-start;
}

.message.user {
  flex-direction: row-reverse;
}

.avatar {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 12px;
  flex-shrink: 0;
}

.avatar.assistant {
  background: var(--orange);
  color: white;
}

.avatar.user {
  background: var(--gray-100);
  color: var(--text-secondary);
}

.message-content {
  max-width: 70%;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 16px;
  box-shadow: var(--shadow-sm);
}

.message.user .message-content {
  background: var(--orange-light);
  border-color: transparent;
}

[data-theme='dark'] .message.user .message-content {
  background: #2d1b14;
}

.role-label {
  font-size: 12px;
  font-weight: 500;
  color: var(--text-muted);
  margin-bottom: 8px;
}

.message-text {
  color: var(--text-primary);
  line-height: 1.6;
}

.typing-indicator {
  display: flex;
  gap: 4px;
  align-items: center;
  color: var(--text-muted);
}

.typing-dots {
  display: flex;
  gap: 2px;
}

.typing-dot {
  width: 4px;
  height: 4px;
  background: var(--text-muted);
  border-radius: 50%;
  animation: typing 1.4s infinite ease-in-out;
}

.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes typing {
  0%, 80%, 100% { opacity: 0.3; }
  40% { opacity: 1; }
}

.input-panel {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  background: var(--surface);
  border-top: 1px solid var(--border);
  padding: 20px 24px 24px;
  backdrop-filter: blur(10px);
  box-shadow: 0 -4px 12px rgba(0, 0, 0, 0.05);
}

.input-container {
  max-width: 800px;
  margin: 0 auto;
}

.input-row {
  display: flex;
  gap: 16px;
  align-items: flex-end;
  margin-bottom: 16px;
}

.textarea-wrapper {
  flex: 1;
  position: relative;
}

#prompt {
  width: 100%;
  min-height: 80px;
  max-height: 200px;
  padding: 16px 20px;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--surface);
  color: var(--text-primary);
  font-family: inherit;
  font-size: 15px;
  line-height: 1.5;
  resize: none;
  outline: none;
  transition: all 0.2s;
}

#prompt:focus {
  border-color: var(--orange);
  box-shadow: 0 0 0 3px rgba(255, 120, 73, 0.1);
}

#prompt::placeholder {
  color: var(--text-muted);
}

#send {
  background: var(--orange);
  color: white;
  border: none;
  border-radius: 12px;
  padding: 12px 20px;
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.2s;
  min-width: 80px;
}

#send:hover:not(:disabled) {
  background: var(--orange-hover);
}

#send:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.controls-row {
  display: flex;
  gap: 12px;
  align-items: center;
}

#model {
  flex: 1;
  padding: 10px 14px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--surface);
  color: var(--text-primary);
  font-size: 14px;
  outline: none;
  transition: all 0.2s;
}

#model:focus {
  border-color: var(--orange);
  box-shadow: 0 0 0 2px rgba(255, 120, 73, 0.1);
}

#theme {
  padding: 8px 12px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--surface);
  color: var(--text-primary);
  font-size: 14px;
  outline: none;
  transition: all 0.2s;
}

#theme:focus {
  border-color: var(--orange);
}

#clear {
  background: var(--surface);
  color: var(--text-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 18px;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s;
}

#clear:hover {
  background: var(--gray-50);
  color: var(--text-primary);
  border-color: var(--text-muted);
}

/* Hide default Shiny styling */
.shiny-input-container label,
.form-group label {
  display: none;
}

.form-control, .form-select {
  border: 1px solid var(--border) !important;
  background: var(--bg) !important;
  color: var(--text-primary) !important;
}

.card {
  background: transparent !important;
  border: none !important;
}

.card-header {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  margin-bottom: 16px !important;
}
"""

# ------------------ UI ------------------
app_ui = ui.page_fluid(
    ui.tags.style(CSS),
    # Theme bootstrap + handler
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
    
    # Header
    ui.div(
        {"class": "header"},
        ui.div(
            {"class": "header-content"},
            ui.h1("Origin Assistant"),
            ui.div(
                {"class": "header-controls"},
                ui.output_ui("status_badge"),
                ui.input_select("theme", None,
                               {"dark": "🌙", "light": "☀️"},
                               selected="dark")
            )
        )
    ),
    
    # Main content
    ui.div(
        {"class": "main-content"},
        
        # Knowledge base section
        ui.div(
            {"class": "kb-section"},
            ui.div(
                {"class": "kb-card"},
                ui.div({"class": "kb-header"}, "📚 Base de Conhecimento"),
                ui.div(
                    {"class": "file-input-wrapper"},
                    ui.input_file("docs", "", multiple=True, accept=[".pdf"]),
                    ui.tags.label(
                        {"class": "file-input-label", "for": "docs"},
                        "📄 Adicionar PDFs"
                    )
                ),
                ui.div({"class": "kb-status"}, ui.output_text("kb_status"))
            )
        ),
        
        # Chat area
        ui.div({"class": "chat-container"}, ui.output_ui("chat_thread"))
    ),
    
    # Input panel
    ui.div(
        {"class": "input-panel"},
        ui.div(
            {"class": "input-container"},
            ui.div(
                {"class": "input-row"},
                ui.div(
                    {"class": "textarea-wrapper"},
                    ui.input_text_area("prompt", None, rows=1,
                                       placeholder="Digite sua mensagem...")
                ),
                ui.input_action_button("send", "Enviar", class_="send-button")
            ),
            ui.div(
                {"class": "controls-row"},
                ui.input_select("model", None, {
                    "claude-3-haiku-20240307": "Claude 3 Haiku",
                    "claude-3-5-sonnet-20240620": "Claude 3.5 Sonnet"
                }, selected="claude-3-haiku-20240307"),
                ui.input_action_button("clear", "Limpar")
            )
        )
    )
)

# ------------- Server -------------
def server(input, output, session):
    history = reactive.Value([])
    typing = reactive.Value(False)

    def push(role, content):
        history.set(history() + [{"role": role, "content": content}])

    @render.ui
    def status_badge():
        if HAS_KEY and client is not None:
            return ui.span({"class": "status-badge status-ok"}, "✓ Conectado")
        elif HAS_KEY and client is None and Anthropic is None:
            return ui.span({"class": "status-badge status-warn"}, "⚠ Instalar 'anthropic'")
        else:
            return ui.span({"class": "status-badge status-error"}, "✗ Sem chave")

    @render.text
    def kb_status():
        if not HAVE_RAG_DEPS:
            return "RAG indisponível: instale pypdf, scikit-learn e joblib."
        chunks, _, _ = load_index()
        n_docs = len({c["source"] for c in chunks}) if chunks else 0
        n_chunks = len(chunks)
        return f"{n_docs} PDF(s) • {n_chunks} chunk(s) indexados"

    @render.ui
    def chat_thread():
        items = []
        for m in history():
            is_user = m["role"] == "user"
            items.append(
                ui.div(
                    {"class": f"message {'user' if is_user else 'assistant'}"},
                    ui.div(
                        {"class": f"avatar {'user' if is_user else 'assistant'}"},
                        "U" if is_user else "OA"
                    ),
                    ui.div(
                        {"class": "message-content"},
                        ui.div({"class": "role-label"}, 
                               "Você" if is_user else "Origin Assistant"),
                        ui.div({"class": "message-text"}, ui.markdown(m["content"]))
                    )
                )
            )
        
        if typing():
            items.append(
                ui.div(
                    {"class": "message assistant"},
                    ui.div({"class": "avatar assistant"}, "OA"),
                    ui.div(
                        {"class": "message-content"},
                        ui.div({"class": "role-label"}, "Origin Assistant"),
                        ui.div(
                            {"class": "typing-indicator"},
                            "Digitando",
                            ui.div(
                                {"class": "typing-dots"},
                                ui.div({"class": "typing-dot"}),
                                ui.div({"class": "typing-dot"}),
                                ui.div({"class": "typing-dot"})
                            )
                        )
                    )
                )
            )
        return ui.TagList(*items)

    @reactive.Effect
    @reactive.event(input.clear)
    def _clear():
        history.set([])
        ui.update_text_area("prompt", value="")

    # Apply theme when selection changes
    @reactive.Effect
    async def _theme_apply():
        theme = input.theme() or "dark"
        await session.send_custom_message("set_theme", theme)

    # Auto-scroll script
    ui.tags.script("""
        // Auto-scroll chat
        new MutationObserver(() => {
          const container = document.querySelector('.chat-container');
          if (container) {
            container.scrollTop = container.scrollHeight;
          }
        }).observe(document.body, {childList: true, subtree: true});
        
        // Enter to send (Shift+Enter for new line)
        document.addEventListener('keydown', (e) => {
          if (e.target.id === 'prompt' && e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            document.getElementById('send').click();
          }
        });
        
        // Auto-resize textarea
        function autoResize(textarea) {
          textarea.style.height = 'auto';
          textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
        }
        
        document.addEventListener('input', (e) => {
          if (e.target.id === 'prompt') {
            autoResize(e.target);
          }
        });
    """)

    @reactive.Effect
    @reactive.event(input.docs)
    def _ingest_pdfs():
        if not HAVE_RAG_DEPS:
            ui.notification_show("Instale pypdf, scikit-learn e joblib para ativar o RAG.", type="error")
            return
        files = input.docs() or []
        if not files: return
        paths = []
        for f in files:
            src = Path(f["datapath"])
            dst = DATA_DIR / f["name"]
            dst.write_bytes(src.read_bytes())
            paths.append(dst)
        total = add_pdfs_to_index(paths)
        ui.notification_show(f"PDF(s) adicionados. Total de chunks: {total}", type="message")

    @reactive.Effect
    @reactive.event(input.send)
    def _send():
        q = (input.prompt() or "").strip()
        if not q:
            ui.notification_show("Digite sua mensagem.", type="warning")
            return
        push("user", q)
        ui.update_text_area("prompt", value="")
        if client is None:
            push("assistant", "Claude indisponível. Verifique ANTHROPIC_API_KEY e o pacote 'anthropic'.")
            return
        typing.set(True)
        model = (input.model() or "claude-3-haiku-20240307")
        reply = chat_reply_with_context(history(), model)
        typing.set(False)
        push("assistant", reply)

app = App(app_ui, server)




