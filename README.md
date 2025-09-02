# Origin Software Assistant (Shiny for Python)

App Shiny para Python com integração opcional ao Claude (Anthropic). Pronto para publicar no **Posit Connect Cloud** (via GitHub) ou no **shinyapps.io**.

## Estrutura
```
origin-software-assistant/
├─ app.py                # entrypoint (App Shiny)
├─ requirements.txt
├─ .python-version       # 3.11
├─ .env.example          # exemplo de env local (não commitar .env)
└─ README-Posit-Connect-GitHub.md  # guia passo a passo p/ Connect Cloud
```

> No Connect Cloud, defina `ANTHROPIC_API_KEY` em *Settings → Environment*.  
> No shinyapps.io, inclua um `.env` no bundle e carregue com `python-dotenv`.

## Rodar localmente
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # edite com sua chave real
python -m shiny run --reload app:app  # ou: python app.py (com shiny>=0.6)
```

## Publicar no Posit Connect Cloud (GitHub)
1. Suba este diretório para o GitHub (repositório vazio).
2. No Connect Cloud → **Publish → Build your own → Install** (autorize GitHub).
3. Selecione o repo/branch `main`, subdirectory vazio, e publique.
4. Depois de publicar: **Settings → Environment → Add** `ANTHROPIC_API_KEY=sk-ant-...` → **Save** → **Restart**.

Guia detalhado: veja **README-Posit-Connect-GitHub.md**.

## Licença
MIT — veja `LICENSE`.
