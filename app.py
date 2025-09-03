# app-16.py — Shiny (Python) chat UI + RAG de PDFs (TF-IDF) com persistência em S3 (Cloudflare R2)
# Compatível com Shiny 1.4 (usa `await session.send_custom_message`)
#
# ENV esperadas no Posit Connect:
#   ANTHROPIC_API_KEY
#   S3_ENDPOINT_URL=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
#   S3_BUCKET=origin-assistant-cache
#   AWS_ACCESS_KEY_ID=<Access Key ID>
#   AWS_SECRET_ACCESS_KEY=<Secret Access Key>
#   AWS_DEFAULT_REGION=auto            (opcional; pode omitir)
#   S3_PREFIX=osa-cache/               (opcional; terminar com '/')
#
from shiny import App, ui, render, reactive
from dotenv import load_dotenv
import os, re, json, hashlib
from pathlib import Path

# ---------------- Env / Anthropíc ----------------
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

print(f"[BOOT] RAG={HAVE_RAG_DEPS} | S3={HAVE_S3} | Anthropic={(client is not None)}")

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
    # normaliza prefixo
    if prefix and not prefix.endswith('/'):
        prefix = prefix + '/'
    if not (endpoint and bucket and key and secret):
        return None
    # region: se vier vazio/auto, omite para deixar boto3 escolher
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
    # se já existe local, não baixa
    if CHUNKS_JSON.exists() and VECTORIZER_JOBLIB.exists() and MATRIX_JOBLIB.exists():
        print("[S3] cache local encontrado; não foi necessário baixar.")
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
        print("[S3] cache local incompleto; não foi enviado.")
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

# tenta restaurar do S3 na inicialização
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
    # primeiro tenta baixar do S3 (se habilitado)
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
    # sincroniza com S3
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
        return []
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    top_idx = sims.argsort()[::-1][:k]
    return [{
        "score": float(sims[i]),
        "text": chunks[i]["text"],
        "source": chunks[i]["source"],
        "chunk_id": chunks[i]["chunk_id"]
    } for i in top_idx]

def build_context(query: str):
    hits = retrieve(query, k=4)
    ctx = "\n\n".join(
        [f"[{i+1}] ({h['source']} • {h['chunk_id']} • score={h['score']:.3f})\n{h['text']}" for i, h in enumerate(hits)]
    )
    cites = "\n".join([f"- {h['source']} ({h['chunk_id']})" for h in hits])
    return ctx, cites

# ---------------- Chat helpers ----------------
def anthropic_messages_from_history(history):
    msgs = []
    for m in history:
        if m["role"] in ("user","assistant"):
            msgs.append({"role": m["role"], "content":[{"type":"text","text": m["content"]}]})
    return msgs

def chat_reply_with_context(history, model):
    if client is None:
        return "Claude indisponível. Configure ANTHROPIC_API_KEY e o pacote 'anthropic'."
    question = next((m["content"] for m in reversed(history) if m["role"]=="user"), "")
    ctx, cites = build_context(question) if HAVE_RAG_DEPS else ("", "")

    system = (
        "Você é o Origin Software Assistant. Use o CONTEXTO quando ele estiver presente; "
        "se a resposta não estiver no contexto, diga claramente que o documento não contém a informação.\n\n"
        f"=== CONTEXTO ===\n{ctx}\n=== FIM DO CONTEXTO ==="
    )
    resp = client.messages.create(
        model=model, max_tokens=900, temperature=0.2,
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
    if cites:
        answer += "\n\n---\n**Fontes:**\n" + cites
    return answer

# ---------------- CSS / UI ----------------
CSS = """
:root{
  --bg:#f8fafc; --panel:#ffffff; --bubble-user:#f3f4f6; --bubble-assistant:#eef2ff;
  --border:#e5e7eb; --text:#0f172a; --muted:#475569; --accent:#7c3aed;
}
[data-theme='dark']{
  --bg:#0b1220; --panel:#0f172a; --bubble-user:#111827; --bubble-assistant:#0b1320;
  --border:#1f2937; --text:#e5e7eb; --muted:#9ca3af; --accent:#8b5cf6;
}
html,body{height:100%}
body{background:linear-gradient(180deg,var(--bg),var(--panel) 55%,var(--bg));color:var(--text);}
a{color:var(--accent)}
.header{max-width:980px;margin:18px auto 0;padding:8px 16px;display:flex;align-items:center;gap:8px;justify-content:space-between}
.header .left h3{font-weight:700;margin:0}
.header .sub{color:var(--muted);margin:2px 0 0 0}
.header .right{display:flex;gap:8px;align-items:center}
.badge{font-size:.9rem;color:var(--muted)}
.kb{max-width:980px;margin:10px auto;padding:0 16px}
.chat-container{max-width:980px;margin:0 auto;padding:8px 16px 120px}
.message{display:flex;gap:12px;padding:14px 16px;border-radius:16px;margin:10px 0;border:1px solid var(--border);background:var(--bubble-assistant)}
.message.user{background:var(--bubble-user)}
.avatar{width:32px;height:32px;border-radius:8px;background:var(--accent);display:flex;align-items:center;justify-content:center;font-weight:700;color:white;flex-shrink:0}
.role{font-weight:600;margin-bottom:4px;color:var(--muted)}
.content{white-space:pre-wrap;line-height:1.5}
.panel-bottom{backdrop-filter:blur(10px); background:color-mix(in oklab, var(--bg) 70%, var(--panel)); border-top:1px solid var(--border); padding:12px}
.composer{max-width:980px;margin:0 auto;display:flex;gap:10px;align-items:flex-end}
.composer .left{flex:1}
textarea.form-control{background:var(--panel);color:var(--text);border:1px solid var(--border);}
select.form-select{background:var(--panel);color:var(--text);border:1px solid var(--border);}
.btn-primary{background:var(--accent);border-color:var(--accent)}
.badge-ok{color:#10b981}.badge-warn{color:#f59e0b}.badge-err{color:#ef4444}
"""

app_ui = ui.page_fluid(
    ui.tags.style(CSS),
    # theme handler
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
    ui.div(
        {"class":"header"},
        ui.div({"class":"left"},
               ui.h3("🚀 Origin Software Assistant"),
               ui.p({"class":"sub"}, "Chat + RAG de PDFs (cache local + S3/R2) • Claude via ANTHROPIC_API_KEY")),
        ui.div({"class":"right"},
               ui.input_select("theme", None,
                               {"dark":"🌙 Escuro","light":"☀️ Claro"},
                               selected="dark"),
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
                    ui.input_text_area("prompt", None, rows=3,
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
            return "✅ Chave detectada"
        elif HAS_KEY and client is None and Anthropic is None:
            return "⚠️ Falta instalar 'anthropic'"
        else:
            return "❌ Sem chave"

    @render.text
    def kb_status():
        if not HAVE_RAG_DEPS:
            return "RAG indisponível: instale pypdf, scikit-learn e joblib."
        chunks, _, _ = load_index()
        n_docs = len({c["source"] for c in chunks}) if chunks else 0
        n_chunks = len(chunks)
        tip = " • S3 ativo" if _S3 else ""
        return f"📄 {n_docs} PDF(s) • 🧩 {n_chunks} chunk(s){tip}"

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

    # atalhos de teclado + autoscroll
    ui.tags.script("""
        document.addEventListener('keydown', (e)=>{
          if(e.target.id==='prompt' && e.key==='Enter' && !e.shiftKey){
            e.preventDefault();
            document.getElementById('send').click();
          }
        });
        new MutationObserver(()=>{
          const el=document.querySelector('.chat-container');
          if(el) el.scrollTop = el.scrollHeight;
        }).observe(document.body,{childList:true,subtree:true});
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

