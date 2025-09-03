# app.py — Shiny (Python) chat + RAG de PDFs com persistência S3 (Cloudflare R2) e FALLBACK automático para Claude
# Aparência atualizada (modo escuro estilo ChatGPT) + caixa de texto maior com auto-grow
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
    endpoint = os.getenv("S3_ENDPOINT_URL")  # ex.: https://<ACCOUNT_ID>.r2.cloudflarestorage.com
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

    # 1ª via: RAG com contexto
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

    # 2ª via: fallback (sem contexto) — conhecimento geral
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
  /* Light */
  --bg:#F7F7F8; --panel:#FFFFFF;
  --bubble-user:#F2F2F2; --bubble-assistant:#F7F7F8;
  --border:#E2E2E3; --text:#0F172A; --muted:#6B7280;
  --accent:#10A37F;
}
[data-theme='dark']{
  --bg:#343541; --panel:#343541;
  --bubble-user:#444654; --bubble-assistant:#3E3F4A;
  --border:#565869; --text:#ECECF1; --muted:#ACB2BF;
  --accent:#19C37D;
}

html,body{height:100%}
body{
  background:linear-gradient(180deg,var(--bg),var(--panel) 55%,var(--bg));
  color:var(--text);
}
a{color:var(--accent)}

/* ===== FULL BLEED ===== */
:root{ --padX: clamp(14px, 3vw, 32px); }

.header{
  max-width:100%; width:100%;
  margin:18px auto 0; padding:8px var(--padX);
  display:flex; align-items:center; gap:8px; justify-content:space-between
}
.header .left h3{font-weight:700;margin:0}
.header .sub{color:var(--muted);margin:2px 0 0 0}
.header .right{display:flex;gap:8px;align-items:center}
.badge{font-size:.9rem;color:var(--muted)}

.kb{
  max-width:100%; width:100%;
  margin:10px auto; padding:0 var(--padX);
}

.chat-container{
  max-width:100%; width:100%;
  margin:0 auto; padding:8px var(--padX) 260px;
}

.message{
  display:flex;gap:12px;padding:12px 14px;border-radius:16px;margin:10px 0;
  border:1px solid var(--border);background:var(--bubble-assistant)
}
.message.user{background:var(--bubble-user)}
.avatar{
  width:32px;height:32px;border-radius:8px;background:var(--accent);
  display:flex;align-items:center;justify-content:center;font-weight:700;color:white;flex-shrink:0
}
.role{font-weight:600;margin-bottom:4px;color:var(--muted)}

/* --- Texto mais compacto nas respostas --- */
.content{ white-space:pre-wrap; line-height:1.32; font-size:15px; }
.content p{ margin:.22rem 0; }
.content ul, .content ol{ margin:.25rem 0 .25rem 1.15rem; }
.content li{ margin:.10rem 0; }
.content li p{ margin:.18rem 0; }
.content li > ul, .content li > ol{ margin:.18rem 0 .18rem 1.1rem; }
.content h1, .content h2, .content h3{ margin:.5rem 0 .3rem; line-height:1.18; }

.panel-bottom{
  position:sticky;bottom:0;z-index:10;
  backdrop-filter:blur(10px);
  background:linear-gradient(180deg,rgba(0,0,0,0), var(--bg) 30%, var(--bg));
  border-top:1px solid var(--border);
  padding:14px 0 calc(18px + env(safe-area-inset-bottom));
}
.composer{
  max-width:100%; width:100%;
  margin:0 auto; padding:0 var(--padX);
  display:flex;gap:16px;align-items:flex-end;
}
.composer .left{
  flex:1; display:flex; flex-direction:column; gap:12px;
}

textarea.form-control#prompt{
  width:100% !important;
  background:#40414F; color:#ECECF1;
  border:1px solid #6b6f76;
  border-radius:14px; padding:20px 20px;
  min-height:200px; max-height:65vh; resize:vertical;
  font-size:16px; line-height:1.45;
  box-shadow:none;
}
textarea.form-control#prompt::placeholder{color:#c5c7d0;opacity:.95}
textarea.form-control#prompt:focus{
  border-color:#19C37D; outline:none;
  box-shadow:0 0 0 3px rgba(25,195,125,.18);
}
[data-theme='light'] textarea.form-control#prompt{
  background:#fff; color:var(--text); border:1px solid var(--border);
}
[data-theme='light'] textarea.form-control#prompt::placeholder{color:#6B7280}

select.form-select{
  background:var(--panel); color:var(--text); border:1px solid var(--border);
  border-radius:12px; height:48px;
}
.btn-primary{
  background:var(--accent); border-color:var(--accent);
  height:52px; padding:0 22px; border-radius:12px; font-weight:600
}
.btn-primary:hover{filter:brightness(.95)}
.badge-ok{color:#10b981}.badge-warn{color:#f59e0b}.badge-err{color:#ef4444}

"""


app_ui = ui.page_fluid(
    ui.tags.style(CSS),
    # theme handler
    ui.tags.script("""      Shiny.addCustomMessageHandler('set_theme', (theme) => {
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
    ui.div(
        {"class":"header"},
        ui.div({"class":"left"},
               ui.h3("🚀 Origin Software Assistant"),
               ui.p({"class":"sub"}, "Chat + RAG (PDFs) com fallback automático para Claude • Cache S3/R2")),
        ui.div({"class":"right"},
               ui.input_select("theme", None, {"dark":"🌙 Escuro","light":"☀️ Claro"}, selected="dark"),
               ui.tags.span({"class":"badge"}, ui.output_text("status", inline=True)),
        ),
    ),
    ui.div({"class":"kb"},
        ui.card(
            ui.card_header("📚 Base de conhecimento (PDF → índice TF-IDF)"),
            ui.input_file("docs", "Adicionar PDF(s)", multiple=True, accept=[".pdf"]),
            ui.output_text("kb_status")
        )
    ),
    ui.div({"class":"chat-container"}, ui.output_ui("chat_thread")),
    ui.panel_fixed(
        ui.div({"class":"panel-bottom", "style":"padding:0"},
            ui.div({"class":"composer"},
                ui.div({"class":"left"},
                    ui.input_text_area("prompt", None, rows=5,
                        placeholder="Envie uma mensagem… (Shift+Enter = quebra de linha)"),
                    ui.row(
                        ui.column(8,
                            ui.input_select("model", None, {
                                "claude-3-haiku-20240307":"Claude 3 Haiku (econômico)",
                                "claude-3-5-sonnet-20240620":"Claude 3.5 Sonnet (qualidade)"
                            }, selected="claude-3-haiku-20240307")
                        ),
                        ui.column(4, ui.input_action_button("clear","Limpar"))
                    ),
                ),
                ui.input_action_button("send","Enviar", class_="btn btn-primary"),
            )
        ),
        left="0", right="0", bottom="0"
    )
)

def server(input, output, session):
    history = reactive.Value([])
    typing = reactive.Value(False)

    def push(role, content):
        history.set(history() + [{"role": role, "content": content}])

    @render.text
    def status():
        if HAS_KEY and client is not None:
            return "✅ Claude pronto"
        elif HAS_KEY and client is None and Anthropic is None:
            return "⚠️ Falta instalar 'anthropic'"
        else:
            return "❌ Sem ANTHROPIC_API_KEY"

    @render.text
    def kb_status():
        if not HAVE_RAG_DEPS:
            return "RAG indisponível: instale pypdf, scikit-learn e joblib."
        chunks, _, _ = load_index()
        n_docs = len({c["source"] for c in chunks}) if chunks else 0
        n_chunks = len(chunks)
        tip = " • S3 ativo" if _S3 else ""
        fb = f" • FB:{RAG_FALLBACK}({RAG_MIN_TOPSCORE}/{RAG_MIN_CTXCHARS})"
        return f"📄 {n_docs} PDF(s) • 🧩 {n_chunks} chunk(s){tip}{fb}"

    @render.ui
    def chat_thread():
        items = []
        for m in history():
            cls = "assistant" if m["role"] == "assistant" else "user"
            avatar = "OA" if m["role"] == "assistant" else "Você"
            items.append(
                ui.div(
                    {"class": f"message {cls}"},
                    ui.div({"class":"avatar"}, avatar[0]),
                    ui.div(
                        ui.div({"class":"role"}, "Origin Assistant" if m["role"]=="assistant" else "Você"),
                        ui.div({"class":"content"}, ui.markdown(m["content"])),
                    )
                )
            )
        if typing():
            items.append(
                ui.div({"class":"message assistant"},
                       ui.div({"class":"avatar"}, "OA"),
                       ui.div(ui.div({"class":"role"},"Origin Assistant"),
                              ui.div({"class":"content"},"Digitando… ⏳")))
            )
        return ui.TagList(*items)

    @reactive.Effect
    @reactive.event(input.clear)
    def _clear():
        history.set([])
        ui.update_text_area("prompt", value="")

    # Tema (Shiny 1.4 exige await)
    @reactive.Effect
    async def _theme_apply():
        theme = input.theme() or "dark"
        await session.send_custom_message("set_theme", theme)

    # atalhos + autoscroll
    ui.tags.script("""        document.addEventListener('keydown', (e)=>{
          if(e.target.id==='prompt' && e.key==='Enter' && !e.shiftKey){
            e.preventDefault();
            document.getElementById('send').click();
          }
        });
        new MutationObserver(()=>{
          const el=document.querySelector('.chat-container');
          if(el) el.scrollTop = el.scrollHeight;
        }).observe(document.body,{childList:true,subtree:true});
    """    )

    # auto-grow suave do textarea
    ui.tags.script("""      const grow = el => {
        if (!el) return;
        el.style.height = 'auto';
        const h = Math.min(el.scrollHeight, window.innerHeight * 0.4);
        el.style.height = h + 'px';
      };
      const promptEl = () => document.getElementById('prompt');
      document.addEventListener('input', (e) => {
        if (e.target && e.target.id === 'prompt') grow(e.target);
      });
      const obs = new MutationObserver(() => grow(promptEl()));
      obs.observe(document.body, {childList:true, subtree:true});
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
        ok = s3_push_cache() if _S3 else False
        msg = f"PDF(s) adicionados. Total de chunks: {total}" + (" • sincronizado com S3" if ok else "")
        ui.notification_show(msg, type="message")

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







