# app.py — ChatGPT-style UI with Light/Dark theme toggle (fixed panel_fixed style)
from shiny import App, ui, render, reactive
from dotenv import load_dotenv
import os
import traceback

# --- Env (Connect Cloud: Settings → Environment; local: .env) ---
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

# ----------------- CSS (Light/Dark via data-theme attr) -----------------
CSS = """
:root{
  --bg:#f8fafc;
  --panel:#ffffff;
  --bubble-user:#f3f4f6;
  --bubble-assistant:#eef2ff;
  --border:#e5e7eb;
  --text:#0f172a;
  --muted:#475569;
  --accent:#7c3aed;
}
[data-theme='dark']{
  --bg:#0b1220;
  --panel:#0f172a;
  --bubble-user:#111827;
  --bubble-assistant:#0b1320;
  --border:#1f2937;
  --text:#e5e7eb;
  --muted:#9ca3af;
  --accent:#8b5cf6;
}
html,body{height:100%}
body{background:linear-gradient(180deg,var(--bg),var(--panel) 55%,var(--bg));color:var(--text);}
a{color:var(--accent)}
.header{max-width:980px;margin:18px auto 0;padding:8px 16px;display:flex;align-items:center;gap:8px;justify-content:space-between}
.header .left h3{font-weight:700;margin:0}
.header .sub{color:var(--muted);margin:2px 0 0 0}
.header .right{display:flex;gap:8px;align-items:center}
.badge{font-size:.9rem;color:var(--muted)}
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
.badge-ok{color:#10b981}
.badge-warn{color:#f59e0b}
.badge-err{color:#ef4444}
"""

# ------------------ UI ------------------
app_ui = ui.page_fluid(
    ui.tags.style(CSS),
    # Theme bootstrap + handler (persiste no localStorage)
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
               ui.p({"class":"sub"}, "Chat ao estilo ChatGPT • Claude opcional via ANTHROPIC_API_KEY")),
        ui.div({"class":"right"},
               ui.input_select("theme", None,
                               {"dark":"🌙 Escuro","light":"☀️ Claro"},
                               selected="dark"),
               ui.tags.span({"class":"badge"}, ui.output_text("status", inline=True)),
        ),
    ),
    ui.div({"class":"chat-container"}, ui.output_ui("chat_thread")),
    # FIX: remove style kwarg from panel_fixed (Shiny supplies its own style);
    # move padding into the inner div instead.
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

# ------------- Helpers -------------
def anthropic_messages_from_history(history):
    msgs = []
    for m in history:
        role = m["role"]
        if role not in ("user","assistant"):
            continue
        msgs.append({"role": role, "content":[{"type":"text","text": m["content"]}]})
    return msgs

def chat_reply(history, model):
    if client is None:
        return "Configuração necessária: defina ANTHROPIC_API_KEY e instale 'anthropic'."
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=900,
            temperature=0.2,
            system=(
                "Você é o Origin Software Assistant. Responda em português, com passos numerados, "
                "exemplos curtos e comandos quando apropriado. Seja objetivo e útil."
            ),
            messages=anthropic_messages_from_history(history),
        )
        parts = []
        try:
            for block in getattr(resp, "content", []):
                if getattr(block, "type", None) == "text":
                    parts.append(block.text)
        except Exception:
            parts = [str(resp)]
        return "\n".join(parts) if parts else str(resp)
    except Exception as e:
        traceback.print_exc()
        return f"Erro ao consultar o Claude: {e}"

# ------------- Server -------------
def server(input, output, session):
    # chat history = list of {role, content}
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

    # Apply theme when selection changes
    @reactive.Effect
    def _theme_apply():
        theme = input.theme() or "dark"
        session.send_custom_message("set_theme", theme)

    # Enter (Shift+Enter nova linha) via JS leve
    ui.tags.script("""
        document.addEventListener('keydown', (e)=>{
          if(e.target.id==='prompt' && e.key==='Enter' && !e.shiftKey){
            e.preventDefault();
            document.getElementById('send').click();
          }
        });
    """)

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
        reply = chat_reply(history(), model)
        typing.set(False)
        push("assistant", reply)

app = App(app_ui, server)
