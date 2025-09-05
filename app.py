# app_with_auth.py - Origin Software Assistant com sistema de autenticação
# Compatível com Shiny 1.4.0 para Posit Connect

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

# Paths para o sistema de autenticação
DATA_DIR = Path("data")
AUTH_DIR = DATA_DIR / "auth"
USER_DB_PATH = AUTH_DIR / "users.db"
CACHE_DIR = DATA_DIR / "cache"

# Criar diretórios necessários
for d in (DATA_DIR, AUTH_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------- Sistema de Autenticação ----------------

def hash_password(password: str) -> str:
    """Cria hash seguro da senha"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user_db():
    """Cria tabela de usuários se não existir"""
    with sqlite3.connect(USER_DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY,
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
        
        # Criar usuário admin padrão se não existir
        cur.execute("SELECT id FROM users WHERE username = 'admin'")
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

def validate_user(username: str, password: str) -> Tuple[bool, str, bool]:
    """
    Valida credenciais do usuário
    Retorna: (sucesso, mensagem_erro, is_admin)
    """
    if not username or not password:
        return False, "Usuário e senha são obrigatórios", False
    
    with sqlite3.connect(USER_DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            SELECT password_hash, active, subscription_expires, is_admin
            FROM users WHERE username = ?
        """, (username,))
        result = cur.fetchone()
        
        if not result:
            return False, "Usuário ou senha incorretos", False
        
        password_hash, active, subscription_expires, is_admin = result
        
        # Verificar senha
        if hash_password(password) != password_hash:
            return False, "Usuário ou senha incorretos", False
        
        # Verificar se usuário está ativo
        if not active:
            return False, "Usuário desativado. Entre em contato com o suporte.", False
        
        # Verificar se assinatura não expirou
        if subscription_expires:
            expiry = datetime.fromisoformat(subscription_expires)
            if datetime.utcnow() > expiry:
                return False, "Assinatura expirada. Renove seu acesso.", False
        
        # Atualizar último login
        cur.execute(
            "UPDATE users SET last_login = ? WHERE username = ?",
            (datetime.utcnow().isoformat(), username)
        )
        con.commit()
        
        return True, "Login realizado com sucesso!", bool(is_admin)

def add_user(username: str, password: str, email: str = "", months: int = 12) -> Tuple[bool, str]:
    """Adiciona novo usuário (apenas admin)"""
    with sqlite3.connect(USER_DB_PATH) as con:
        cur = con.cursor()
        try:
            expiry = datetime.utcnow() + timedelta(days=30*months)
            cur.execute("""
                INSERT INTO users(username, password_hash, email, created_at, active, subscription_expires, is_admin)
                VALUES(?, ?, ?, ?, 1, ?, 0)
            """, (username, hash_password(password), email, 
                  datetime.utcnow().isoformat(), expiry.isoformat()))
            con.commit()
            return True, f"Usuário '{username}' criado com sucesso!"
        except sqlite3.IntegrityError:
            return False, "Este nome de usuário já existe"

def list_users():
    """Lista todos os usuários"""
    with sqlite3.connect(USER_DB_PATH) as con:
        cur = con.cursor()
        cur.execute("""
            SELECT username, email, created_at, last_login, active, subscription_expires, is_admin
            FROM users ORDER BY created_at DESC
        """)
        return cur.fetchall()

# Criar banco de dados na inicialização
create_user_db()

# ---------------- CSS Customizado ----------------

FULL_CSS = """
:root{
  --bg:#F7F7F8; --panel:#FFFFFF;
  --bubble-user:#E5F2FF; --bubble-assistant:#F7F7F8;
  --border:#E2E2E3; --text:#0F172A; --muted:#6B7280;
  --accent:#10A37F; --accent-hover:#0E8B6F;
  --error:#EF4444; --success:#10B981;
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
html, body { height: 100%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
body { background: var(--bg); color: var(--text); }

/* Login Page */
.login-container {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.login-card {
  background: var(--panel);
  border-radius: 20px;
  box-shadow: 0 20px 60px rgba(0,0,0,0.3);
  padding: 40px;
  width: 100%;
  max-width: 400px;
  margin: 20px;
}

.login-header {
  text-align: center;
  margin-bottom: 30px;
}

.login-logo {
  font-size: 48px;
  margin-bottom: 16px;
}

.login-title {
  font-size: 24px;
  font-weight: 700;
  margin-bottom: 8px;
  background: linear-gradient(135deg, var(--accent), var(--accent-hover));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.login-subtitle {
  color: var(--muted);
  font-size: 14px;
}

/* Main App */
.app-header {
  background: var(--panel);
  border-bottom: 1px solid var(--border);
  padding: 16px 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: var(--shadow);
}

.app-header h1 {
  font-size: 20px;
  margin: 0;
}

/* Form Elements */
.form-group {
  margin-bottom: 20px;
}

.form-label {
  display: block;
  margin-bottom: 8px;
  font-size: 14px;
  font-weight: 500;
  color: var(--text);
}

input[type="text"],
input[type="password"],
input[type="email"],
input[type="number"],
select,
textarea {
  width: 100%;
  padding: 12px 16px;
  border: 2px solid var(--border);
  border-radius: 10px;
  font-size: 15px;
  background: var(--bg);
  color: var(--text);
  transition: all 0.2s;
}

input:focus,
select:focus,
textarea:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(25,195,125,0.1);
}

/* Buttons */
.btn {
  padding: 12px 24px;
  border-radius: 10px;
  border: none;
  font-weight: 600;
  font-size: 15px;
  cursor: pointer;
  transition: all 0.2s;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  text-align: center;
}

.btn-primary {
  background: linear-gradient(135deg, var(--accent), var(--accent-hover));
  color: white;
  width: 100%;
}

.btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(25,195,125,0.3);
}

.btn-secondary {
  background: var(--bg);
  color: var(--text);
  border: 2px solid var(--border);
  width: 100%;
}

.btn-secondary:hover {
  background: var(--panel);
  border-color: var(--accent);
}

.btn-logout {
  background: rgba(239,68,68,0.1);
  color: var(--error);
  border: 1px solid var(--error);
  padding: 8px 16px;
  width: auto;
}

/* Messages */
.alert {
  padding: 12px 16px;
  border-radius: 8px;
  margin-bottom: 20px;
  font-size: 14px;
}

.alert-error {
  background: rgba(239,68,68,0.1);
  color: var(--error);
  border: 1px solid rgba(239,68,68,0.2);
}

.alert-success {
  background: rgba(16,185,129,0.1);
  color: var(--success);
  border: 1px solid rgba(16,185,129,0.2);
}

.alert-info {
  background: rgba(59,130,246,0.1);
  color: #3b82f6;
  border: 1px solid rgba(59,130,246,0.2);
}

/* Admin Panel */
.admin-panel {
  background: var(--panel);
  border-radius: 12px;
  padding: 24px;
  margin: 20px;
  box-shadow: var(--shadow);
}

.admin-header {
  margin-bottom: 20px;
  padding-bottom: 16px;
  border-bottom: 2px solid var(--border);
}

.user-card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 12px;
}

.user-info {
  margin-bottom: 8px;
}

.user-name {
  font-weight: 600;
  font-size: 16px;
  margin-bottom: 4px;
}

.user-meta {
  font-size: 13px;
  color: var(--muted);
}

.status-badge {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 500;
}

.status-active {
  background: rgba(16,185,129,0.1);
  color: var(--success);
}

.status-inactive {
  background: rgba(239,68,68,0.1);
  color: var(--error);
}

/* Content Area */
.content-area {
  padding: 20px;
  background: var(--bg);
  min-height: calc(100vh - 80px);
}

/* Divider */
.divider {
  height: 1px;
  background: var(--border);
  margin: 20px 0;
}
"""

# ---------------- Interface do App ----------------

app_ui = ui.page_fluid(
    ui.tags.style(FULL_CSS),
    ui.tags.script("""
        // Theme handler
        document.documentElement.setAttribute('data-theme', 'dark');
    """),
    
    # Container principal com condicional
    ui.output_ui("main_content")
)

def server(input: Inputs, output: Outputs, session: Session):
    # Estado da sessão
    authenticated = reactive.Value(False)
    current_user = reactive.Value("")
    is_admin = reactive.Value(False)
    login_message = reactive.Value("")
    demo_shown = reactive.Value(False)
    
    @output
    @render.ui
    def main_content():
        """Renderiza login ou app principal baseado no estado de autenticação"""
        if not authenticated():
            # TELA DE LOGIN
            return ui.div({"class": "login-container"},
                ui.div({"class": "login-card"},
                    ui.div({"class": "login-header"},
                        ui.div({"class": "login-logo"}, "🔐"),
                        ui.h1({"class": "login-title"}, "Origin Software Assistant"),
                        ui.p({"class": "login-subtitle"}, "Entre com suas credenciais para acessar")
                    ),
                    
                    # Mensagem de feedback
                    ui.output_ui("login_feedback"),
                    
                    # Formulário
                    ui.div({"class": "form-group"},
                        ui.span({"class": "form-label"}, "👤 Usuário"),
                        ui.input_text("username", None, 
                            placeholder="Digite seu usuário",
                            width="100%"
                        )
                    ),
                    
                    ui.div({"class": "form-group"},
                        ui.span({"class": "form-label"}, "🔑 Senha"),
                        ui.input_password("password", None,
                            placeholder="Digite sua senha",
                            width="100%"
                        )
                    ),
                    
                    ui.input_action_button("login_btn", "🚀 Entrar",
                        class_="btn btn-primary"
                    ),
                    
                    ui.div({"class": "divider"}),
                    
                    ui.input_action_button("demo_btn", "👁️ Ver credenciais demo",
                        class_="btn btn-secondary"
                    ),
                    
                    # Info demo
                    ui.output_ui("demo_info")
                )
            )
        else:
            # APP PRINCIPAL (após login)
            return ui.TagList(
                # Header
                ui.div({"class": "app-header"},
                    ui.div(
                        ui.h1("🚀 Origin Software Assistant"),
                        ui.p({"style": "margin: 0; color: var(--muted); font-size: 14px;"}, 
                            f"Bem-vindo, {current_user()}{'  (Admin)' if is_admin() else ''}")
                    ),
                    ui.input_action_button("logout_btn", "🚪 Logout",
                        class_="btn btn-logout"
                    )
                ),
                
                # Admin Panel (condicional)
                ui.output_ui("admin_panel"),
                
                # Content Area
                ui.div({"class": "content-area"},
                    ui.h2("Chat com RAG"),
                    ui.p("Sistema de chat integrado com processamento de PDFs"),
                    
                    ui.div({"class": "divider"}),
                    
                    # Status do sistema
                    ui.div({"style": "background: var(--panel); padding: 20px; border-radius: 12px; margin: 20px 0;"},
                        ui.h3("📊 Status do Sistema"),
                        ui.p(f"✅ Autenticação ativa"),
                        ui.p(f"{'✅' if HAS_KEY else '❌'} API Key Anthropic {'configurada' if HAS_KEY else 'não configurada'}"),
                        ui.p(f"📁 Dados em: {DATA_DIR.absolute()}")
                    ),
                    
                    ui.div({"class": "divider"}),
                    
                    # Área para adicionar o chat original
                    ui.div({"style": "background: var(--panel); padding: 20px; border-radius: 12px;"},
                        ui.h3("💬 Área do Chat"),
                        ui.p({"style": "color: var(--muted);"},
                            "Integre aqui o código do chat original com RAG, "
                            "mantendo toda a funcionalidade de processar PDFs e "
                            "conversar com o Claude."
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
                return ui.div({"class": "alert alert-error"}, f"❌ {msg}")
        return ui.TagList()
    
    @output
    @render.ui
    def demo_info():
        """Mostra informações da conta demo"""
        if demo_shown():
            return ui.div({"class": "alert alert-info"},
                ui.strong("Conta Demo:"),
                ui.br(),
                "Usuário: ", ui.code("admin"),
                ui.br(),
                "Senha: ", ui.code("admin123")
            )
        return ui.TagList()
    
    @output
    @render.ui
    def admin_panel():
        """Painel administrativo para admins"""
        if not is_admin():
            return ui.TagList()
        
        users = list_users()
        
        return ui.div({"class": "admin-panel"},
            ui.div({"class": "admin-header"},
                ui.h2("🛠️ Painel Administrativo"),
                ui.p({"style": "margin: 0; color: var(--muted);"}, 
                    f"{len(users)} usuários cadastrados")
            ),
            
            # Adicionar usuário
            ui.div({"style": "margin-bottom: 24px;"},
                ui.h3("➕ Adicionar Novo Usuário"),
                ui.row(
                    ui.column(6,
                        ui.input_text("new_username", "Usuário", 
                            placeholder="Nome de usuário")
                    ),
                    ui.column(6,
                        ui.input_password("new_password", "Senha",
                            placeholder="Senha segura")
                    )
                ),
                ui.row(
                    ui.column(6,
                        ui.input_text("new_email", "Email (opcional)",
                            placeholder="email@exemplo.com")
                    ),
                    ui.column(6,
                        ui.input_numeric("new_months", "Meses de acesso",
                            value=12, min=1, max=36)
                    )
                ),
                ui.br(),
                ui.input_action_button("add_user_btn", "Criar Usuário",
                    class_="btn btn-primary",
                    width="200px"
                ),
                ui.output_text("add_user_feedback")
            ),
            
            ui.div({"class": "divider"}),
            
            # Lista de usuários
            ui.h3("👥 Usuários Cadastrados"),
            ui.TagList(*[
                ui.div({"class": "user-card"},
                    ui.div({"class": "user-info"},
                        ui.div({"class": "user-name"}, 
                            f"{'👑 ' if user[6] else ''}{user[0]}"
                        ),
                        ui.div({"class": "user-meta"},
                            f"Email: {user[1] or 'N/A'} | ",
                            f"Criado: {user[2][:10] if user[2] else 'N/A'} | ",
                            f"Último login: {user[3][:10] if user[3] else 'Nunca'}"
                        )
                    ),
                    ui.span({"class": f"status-badge {'status-active' if user[4] else 'status-inactive'}"},
                        "Ativo" if user[4] else "Inativo"
                    )
                ) for user in users
            ])
        )
    
    # Event Handlers
    @reactive.Effect
    @reactive.event(input.demo_btn)
    def show_demo():
        """Mostra credenciais demo"""
        demo_shown.set(True)
    
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
            # Force re-render
            ui.notification_show(f"Bem-vindo, {username}!", type="message", duration=3)
    
    @reactive.Effect
    @reactive.event(input.logout_btn)
    def handle_logout():
        """Processa logout"""
        authenticated.set(False)
        current_user.set("")
        is_admin.set(False)
        login_message.set("")
        demo_shown.set(False)
        ui.notification_show("Logout realizado com sucesso", type="message", duration=2)
    
    @output
    @render.text
    def add_user_feedback():
        """Feedback da criação de usuário"""
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
            ui.notification_show("Usuário e senha são obrigatórios", type="warning", duration=3)
            return
        
        success, message = add_user(username, password, email, months)
        
        if success:
            ui.notification_show(message, type="message", duration=3)
            # Limpar campos
            ui.update_text("new_username", value="")
            ui.update_password("new_password", value="")
            ui.update_text("new_email", value="")
        else:
            ui.notification_show(message, type="warning", duration=3)

app = App(app_ui, server)







