# app.py
from shiny import App, ui, render, reactive
from dotenv import load_dotenv
import os
import sys
import traceback

# --- Load secrets from .env (works locally and on shinyapps.io if .env is bundled) ---
load_dotenv()
API_KEY = os.getenv("ANTHROPIC_API_KEY")
HAS_KEY = bool(API_KEY)

# Lazily import anthropic so the app runs even if the package isn't available yet
try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover
    Anthropic = None

# --- Create client only if we have the key and the package ---
client = None
if HAS_KEY and Anthropic is not None:
    try:
        client = Anthropic(api_key=API_KEY)
    except Exception:
        client = None

# ------------------------- UI -------------------------
app_ui = ui.page_fluid(
    ui.panel_title("🚀 Origin Software Assistant"),
    ui.p("Assistant para dúvidas de software e ciência de dados. "
         "Defina a variável de ambiente ANTHROPIC_API_KEY para habilitar o Claude."),
    ui.input_text("question", "Sua pergunta:", placeholder="Como plotar um gráfico de pizza no OriginPro?"),
    ui.input_text_area("context", "Contexto (opcional):", rows=4,
                       placeholder="Ex.: versão do Origin, dados, objetivo..."),
    ui.row(
        ui.column(6, ui.input_action_button("consultar", "🤖 Consultar Origin", class_="btn-primary")),
        ui.column(6, ui.output_text("status", inline=True)),
    ),
    ui.hr(),
    ui.card(
        ui.card_header("Resposta"),
        ui.output_text_verbatim("answer"),
    ),
    ui.hr(),
    ui.p("Dica: para uso local, crie um arquivo .env com ANTHROPIC_API_KEY=..."),
)

# ----------------------- Helpers ----------------------
def make_prompt(question: str, context: str) -> str:
    context_block = f"\n\nContexto:\n{context}" if context else ""
    return (
        "Você é o Origin Software Assistant. Responda de forma prática, passo a passo, "
        "com exemplos quando útil. Foque em OriginPro, Python e visualização de dados."
        f"\n\nPergunta:\n{question}{context_block}"
    )

def ask_claude(question: str, context: str) -> str:
    if client is None:
        return "Configuração necessária: defina ANTHROPIC_API_KEY e instale o pacote 'anthropic'."
    try:
        prompt = make_prompt(question, context)
        # Modelo econômico por padrão; ajuste conforme sua conta
        resp = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=800,
            temperature=0.2,
            system=(
                "Você é o Origin Software Assistant. Responda com precisão, em português, "
                "mostrando passos numerados, exemplos curtos e comandos quando apropriado."
            ),
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        )
        # Extrai o texto da resposta
        content = getattr(resp, "content", None)
        if isinstance(content, list) and content and "text" in content[0].__dict__:
            return content[0].text
        # fallback genérico
        return str(resp)
    except Exception as e:  # pragma: no cover
        traceback.print_exc()
        return f"Erro ao consultar o Claude: {e}"

# ----------------------- Server -----------------------
def server(input, output, session):
    requests = reactive.Value(0)

    @render.text
    def status():
        if HAS_KEY and client is not None:
            return "✅ Chave detectada (Claude habilitado)."
        elif HAS_KEY and client is None and Anthropic is None:
            return "⚠️ Chave presente, mas o pacote 'anthropic' não está instalado."
        else:
            return "❌ Configure ANTHROPIC_API_KEY para habilitar o Claude."

    @render.text
    def answer():
        return "Faça sua pergunta e clique em “Consultar Origin”."

    @reactive.Effect
    @reactive.event(input.consultar)
    def _on_consultar():
        q = (input.question() or "").strip()
        c = (input.context() or "").strip()
        if not q:
            ui.notification_show("Digite sua pergunta.", type="warning")
            return
        if client is None:
            ui.notification_show("Claude indisponível. Verifique ANTHROPIC_API_KEY e o pacote 'anthropic'.", type="error")
            return
        ui.notification_show("Consultando o Claude…", type="message", duration=1.5)
        requests.set(requests() + 1)
        txt = ask_claude(q, c)
        ui.update_text("answer", txt)

# ------------------------ App -------------------------
app = App(app_ui, server)
