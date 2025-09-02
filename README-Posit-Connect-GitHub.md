# Deploy via GitHub — Posit Connect Cloud (Passo a Passo)

Este guia mostra como publicar o seu app **Shiny para Python** no **Posit Connect Cloud** usando **GitHub**.  
Você *não precisa* enviar `.env` no deploy aqui — basta definir a variável em **Settings → Environment**.

---

## 1) Pré‑requisitos
- Conta no **Posit Connect Cloud** (login).
- Conta no **GitHub**.
- Arquivos do app:
  - `app.py` (entrypoint com `app = App(app_ui, server)`)
  - `requirements.txt`
  - `.python-version` (recomendado, ex.: `3.11`)

> Se estiver usando os arquivos que te enviei, já está tudo pronto.

---

## 2) Estrutura mínima do repositório
```
origin-software-assistant/
├─ app.py
├─ requirements.txt
└─ .python-version     # 3.11
```

> **Não** comite `.env` no Git. No Connect Cloud, você criará a variável via UI.

---

## 3) Criar o repositório no GitHub (uma vez)
No terminal, dentro da pasta do projeto:
```bash
git init
git add app.py requirements.txt .python-version
git commit -m "Publicação inicial do app Shiny para Python"
# crie um repositório vazio no GitHub (web) e substitua a URL abaixo
git remote add origin https://github.com/<seu-usuario>/<seu-repo>.git
git branch -M main
git push -u origin main
```

---

## 4) Conectar o GitHub ao Posit Connect Cloud
1. No Connect Cloud, menu esquerdo **Publish** (tela da sua captura).
2. Clique em **Install** para instalar o app do Posit Connect Cloud no seu GitHub.
3. Autorize o acesso e selecione o repositório (ou a organização) onde está o seu projeto.

---

## 5) Publicar a partir do repositório
1. De volta ao **Publish → Build your own**, escolha o repositório e a **branch** (`main`).
2. **Subdirectory**: deixe vazio (ou informe o caminho se seu app estiver em uma subpasta).
3. Confirme o **Runtime** (Python será detectado; a versão vem de `.python-version`).
4. Clique **Publish**.

> O Connect vai instalar as dependências do `requirements.txt` e subir o app.

---

## 6) Definir a variável de ambiente (Claude)
Após a publicação inicial:
1. Abra o conteúdo publicado (clicando no nome do app).
2. Vá em **Settings → Environment**.
3. Clique **Add Variable** e crie:
   - **Name**: `ANTHROPIC_API_KEY`
   - **Value**: `sk-ant-...` (sua chave)
4. Clique **Save** e depois **Restart** (ou **Apply Changes**).

> Não é necessário `.env` no Connect Cloud, mas se o seu `app.py` tiver `load_dotenv()`, não tem problema — a env definida via UI tem prioridade.  

---

## 7) Testar e ver logs
- Abra a URL do app (botão **Open** ou **View**).
- Para depurar, use **Logs** no topo da página do conteúdo.
- Você pode colocar no código um log simples:
  ```python
  import os
  print("ANTHROPIC_API_KEY carregada?", bool(os.getenv("ANTHROPIC_API_KEY")))
  ```
  (Nunca imprima a chave.)

---

## 8) Atualizar o app
- Faça alterações, `git commit` e `git push`.
- No Connect Cloud, entre no conteúdo e clique em **Publish a new version**.
  - (Dependendo da configuração do seu workspace, pode existir opção de publicar direto do último commit.)

---

## 9) Solução de problemas comuns
- **“application exited before accepting connections”**  
  Verifique:
  - Existe `app = App(app_ui, server)` em `app.py`?
  - Pacotes do `requirements.txt` (ex.: `shiny`, `anthropic`) instalaram? Veja **Logs**.
  - Se estiver usando `.env` errado: no **Connect Cloud** defina a `ANTHROPIC_API_KEY` via **Settings → Environment**.
- **Modelo/Claude não responde**  
  - Confirme a variável `ANTHROPIC_API_KEY` e permissões da sua conta no provedor.
- **Versão do Python/compatibilidade**  
  - Ajuste `.python-version` (ex.: `3.11`) e republique.

---

## 10) Publicar via CLI (alternativa sem GitHub)
1. No Connect Cloud: **API Keys** → **New API Key**.
2. No terminal:
   ```bash
   rsconnect add --server https://connect.posit.cloud --api-key '<SUA_API_KEY>' --name positcloud
   rsconnect deploy shiny . --name positcloud --title "Origin Software Assistant" --entrypoint app:app
   ```
3. Depois, **Settings → Environment** → `ANTHROPIC_API_KEY` → **Save** → **Restart**.

---

### Dicas finais
- Nunca exponha sua chave em repositório ou logs.
- Use `requirements.txt` enxuto e fixe a versão do Python.
- Se quiser autenticação por usuário/senha, dá para habilitar nas **Settings → Access** do conteúdo.

Bom deploy! 🚀
