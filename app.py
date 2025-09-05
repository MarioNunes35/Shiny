# app_with_auth.py - Origin Software Assistant com sistema de autenticação
# Para Posit Connect / Shiny Python

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

# Criar diretórios necessários
for d in (DATA_DIR, AUTH_DIR):
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

def add_user(username: str, password: str, email: str = "", months: int = 12, created_by: str = "admin") -> Tuple[bool, str]:
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

def toggle_user_status(username: str) -> bool:
    """Ativa/desativa usuário"""
    with sqlite3.connect(USER_DB_PATH) as con:
        cur = con.cursor()
        cur.execute("UPDATE users SET active = 1 - active WHERE username = ?", (username,))
        con.commit()
        return cur.rowcount > 0

# Criar banco de dados na inicialização
create_user_db()

# ---------------- Claude Integration (mantido do original) ----------------

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

# ---------------- RAG Configuration (mantido do original) ----------------

RAG_FALLBACK = (os.getenv("RAG_FALLBACK", "auto") or "auto").lower()
RAG_MIN_TOPSCORE = float(os.getenv("RAG_MIN_TOPSCORE", "0.18"))
RAG_MIN_CTXCHARS = int(os.getenv("RAG_MIN_CTXCHARS", "300"))

# ... (resto das funções RAG do arquivo original)

# ---------------- CSS Customizado ----------------

LOGIN_CSS = """
:root{
  --bg:#F7F7F8; --panel:#FFFFFF;
  --border:#E2E2E3; --text:#0F172A; --muted:#6B7280;
  --accent:#10A37F; --accent-hover:#0E8B6F;
  --error:#EF4444; --success:#10B981;
  --shadow: 0 1px 3px rgba(0,0,0,0.08);
}
[data-theme='dark']{
  --bg:#202123; --panel:#2D2E30;
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
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.1); }
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

.form-control {
  width: 100%;
  padding: 12px 16px;
  border: 2px solid var(--border);
  border-radius: 10px;
  font-size: 15px;
  background: var(--bg);
  color: var(--text);
  transition: all 0.2s;
}

.form-control:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(25,195,125,0.1);
}

.form-control::placeholder {
  color: var(--muted);
  opacity: 0.7;
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
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.btn-primary {
  background: linear-gradient(135deg, var(--accent), var(--accent-hover));
  color: white;
}

.btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(25,195,125,0.3);
}

.btn-secondary {
  background: var(--bg);
  color: var(--text);
  border: 2px solid var(--border);
}

.btn-secondary:hover {
  background: var(--panel);
  border-color: var(--accent);
}

/* Messages */
.alert {
  padding: 12px 16px;
  border-radius: 8px;
  margin-bottom: 20px;
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 8px;
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
  margin: 20px 0;
  box-shadow: var(--shadow);
}

.admin-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 16px;
  border-bottom: 2px solid var(--border);
}

.user-list {
  display: grid;
  gap: 12px;
}

.user-card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.user-info {
  flex: 1;
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

.user-status {
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

/* Divider */
.divider {
  text-align: center;
  margin: 20px 0;
  position: relative;
}

.divider::before {
  content: '';
  position: absolute;
  left: 0;
  top: 50%;
  width: 100%;
  height: 1px;
  background: var(--border);
}

.divider-text {
  background: var(--panel);
  padding: 0 16px;
  position: relative;
  color: var(--muted);
  font-size: 13px;
}
"""

# ---------------- Interface do Login ----------------

def login_ui():
    """Interface de login"""
    return ui.page_fluid(
        ui.tags.style(LOGIN_CSS),
        ui.div({"class": "login-container"},
            ui.div({"class": "login-card"},
                ui.div({"class": "login-header"},
                    ui.div({"class": "login-logo"}, "🔐"),
                    ui.h1({"class": "login-title"}, "Origin Software Assistant"),
                    ui.p({"class": "login-subtitle"}, "Entre com suas credenciais para acessar")
                ),
                
                ui.output_ui("login_message"),
                
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
                    class_="btn btn-primary",
                    width="100%"
                ),
                
                ui.div({"class": "divider"},
                    ui.span({"class": "divider-text"}, "ou")
                ),
                
                ui.input_action_button("demo_btn", "👁️ Ver credenciais demo",
                    class_="btn btn-secondary",
                    width="100%"
                ),
                
                ui.output_ui("demo_info")
            )
        )
    )

def main_app_ui():
    """Interface principal do app (após login)"""
    # Aqui você coloca a UI do seu app original
    # Vou incluir um exemplo básico que pode ser expandido
    return ui.page_fluid(
        ui.tags.style(LOGIN_CSS),
        
        # Header com logout
        ui.div({"style": "background: var(--panel); padding: 16px; border-bottom: 1px solid var(--border);"},
            ui.row(
                ui.column(8,
                    ui.h2("🚀 Origin Software Assistant"),
                    ui.output_text("welcome_message")
                ),
                ui.column(4,
                    ui.div({"style": "text-align: right;"},
                        ui.input_action_button("logout_btn", "🚪 Logout",
                            class_="btn btn-secondary",
                            width="auto"
                        )
                    )
                )
            )
        ),
        
        # Admin panel (apenas para admins)
        ui.output_ui("admin_panel"),
        
        # Conteúdo principal do app
        ui.div({"style": "padding: 20px;"},
            ui.h3("Chat com RAG"),
            ui.p("Área principal do chat aqui..."),
            # Adicione aqui o resto da interface do seu app
        )
    )

# ---------------- Server Logic ----------------

def server(input: Inputs, output: Outputs, session: Session):
    # Estado da sessão
    authenticated = reactive.Value(False)
    current_user = reactive.Value("")
    is_admin = reactive.Value(False)
    
    @output
    @render.ui
    def login_message():
        """Mensagens de feedback do login"""
        return ui.TagList()
    
    @output
    @render.ui
    def demo_info():
        """Informações da conta demo"""
        return ui.TagList()
    
    @reactive.Effect
    @reactive.event(input.demo_btn)
    def show_demo_credentials():
        """Mostra credenciais demo"""
        @output
        @render.ui
        def demo_info():
            return ui.div({"class": "alert alert-info"},
                "ℹ️",
                ui.div(
                    ui.strong("Conta Demo:"),
                    ui.br(),
                    "Usuário: ", ui.code("admin"),
                    ui.br(),
                    "Senha: ", ui.code("admin123")
                )
            )
    
    @reactive.Effect
    @reactive.event(input.login_btn)
    def handle_login():
        """Processa tentativa de login"""
        username = input.username()
        password = input.password()
        
        success, message, admin = validate_user(username, password)
        
        @output
        @render.ui
        def login_message():
            if success:
                return ui.div({"class": "alert alert-success"}, "✅ ", message)
            else:
                return ui.div({"class": "alert alert-error"}, "❌ ", message)
        
        if success:
            authenticated.set(True)
            current_user.set(username)
            is_admin.set(admin)
            
            # Redirecionar para app principal
            ui.update_navs("main_nav", selected="app")
    
    @output
    @render.text
    def welcome_message():
        if authenticated():
            admin_text = " (Admin)" if is_admin() else ""
            return f"Bem-vindo, {current_user()}{admin_text}!"
        return ""
    
    @output
    @render.ui
    def admin_panel():
        """Painel administrativo"""
        if not is_admin():
            return ui.TagList()
        
        users = list_users()
        
        user_cards = []
        for user in users:
            username, email, created, last_login, active, expires, admin = user
            
            status_class = "status-active" if active else "status-inactive"
            status_text = "Ativo" if active else "Inativo"
            
            user_cards.append(
                ui.div({"class": "user-card"},
                    ui.div({"class": "user-info"},
                        ui.div({"class": "user-name"}, 
                            f"{'👑 ' if admin else ''}{username}"
                        ),
                        ui.div({"class": "user-meta"},
                            f"Email: {email or 'N/A'} | ",
                            f"Último login: {last_login[:10] if last_login else 'Nunca'} | ",
                            f"Expira: {expires[:10] if expires else 'Nunca'}"
                        )
                    ),
                    ui.div({"class": f"user-status {status_class}"}, status_text)
                )
            )
        
        return ui.div({"class": "admin-panel"},
            ui.div({"class": "admin-header"},
                ui.h3("🛠️ Painel Administrativo"),
                ui.span(f"{len(users)} usuários cadastrados")
            ),
            
            # Formulário para adicionar usuário
            ui.details(
                ui.summary("➕ Adicionar Novo Usuário"),
                ui.div({"style": "padding: 16px;"},
                    ui.row(
                        ui.column(6,
                            ui.input_text("new_username", "Usuário",
                                placeholder="Nome de usuário"
                            )
                        ),
                        ui.column(6,
                            ui.input_password("new_password", "Senha",
                                placeholder="Senha"
                            )
                        )
                    ),
                    ui.row(
                        ui.column(6,
                            ui.input_text("new_email", "Email (opcional)",
                                placeholder="email@exemplo.com"
                            )
                        ),
                        ui.column(6,
                            ui.input_numeric("new_months", "Meses de acesso",
                                value=12, min=1, max=36
                            )
                        )
                    ),
                    ui.br(),
                    ui.input_action_button("add_user_btn", "Criar Usuário",
                        class_="btn btn-primary"
                    ),
                    ui.output_text("add_user_message")
                )
            ),
            
            ui.br(),
            ui.h4("Usuários Cadastrados"),
            ui.div({"class": "user-list"}, *user_cards)
        )
    
    @output
    @render.text
    def add_user_message():
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
            @output
            @render.text
            def add_user_message():
                return "❌ Usuário e senha são obrigatórios"
            return
        
        success, message = add_user(username, password, email, months)
        
        @output
        @render.text
        def add_user_message():
            return f"{'✅' if success else '❌'} {message}"
    
    @reactive.Effect
    @reactive.event(input.logout_btn)
    def handle_logout():
        """Processa logout"""
        authenticated.set(False)
        current_user.set("")
        is_admin.set(False)
        ui.update_navs("main_nav", selected="login")

# ---------------- App Principal com Navegação ----------------

app_ui = ui.page_fluid(
    ui.navs_hidden(
        ui.nav_panel("login", login_ui()),
        ui.nav_panel("app", main_app_ui()),
        id="main_nav",
        selected="login"
    )
)

app = App(app_ui, server)







