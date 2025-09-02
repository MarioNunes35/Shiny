Deploy (shinyapps.io):
----------------------
1) Crie um arquivo .env (NÃO faça commit):
   ANTHROPIC_API_KEY=sk-ant-...

2) (Opcional) fixe a versão do Python:
   echo "3.11" > .python-version

3) Faça o deploy (usando a conta já registrada com `rsconnect add`):
   rsconnect deploy shiny . --name shinyapps --title "Origin Software Assistant" --entrypoint app:app .env

Observações:
- O shinyapps.io não tem UI para variáveis de ambiente; por isso usamos .env + python-dotenv.
- Nunca exponha a chave nos logs. O app só registra o status (True/False).
- Se der "application exited before accepting connections", verifique:
    * se há `app = App(app_ui, server)`;
    * `requirements.txt`;
    * se o .env foi incluído no bundle; e
    * os Logs do aplicativo no dashboard.
