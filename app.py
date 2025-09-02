# app.py (fixed)
from shiny import App, ui, render, reactive
from dotenv import load_dotenv
import os
import traceback

# Load .env (useful local; Connect Cloud can set env vars via Settings → Environment)
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

app_ui = ui.page_fluid(
    ui.panel_title("🚀 Origin Software Assistant"),
    ui.p("Assistant para dúvidas de software e ciência de dados. "
         "Defina ANTHROPIC_API_KEY para habilitar o Claude."),
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
)

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
        # Extrai os blocos de texto
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

def server(input, output, session):
    # Reactive state to hold the answer text
    answer_txt = reactive.Value("Faça sua pergunta e clique em “Consultar Origin”.")

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
        return answer_txt()

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
        txt = ask_claude(q, c)
        # Update the reactive value; outputs re-render automatically
        answer_txt.set(txt)

app = App(app_ui, server)
