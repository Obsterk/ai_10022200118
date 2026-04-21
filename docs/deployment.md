# Deployment Guide

**ai_10022200118 — Obour Adomako Tawiah — CS4241 (2026)**

## Target: Streamlit Community Cloud (free)

### One-time setup

1. Create a free account at <https://share.streamlit.io> using your GitHub login.
2. On GitHub, create a public repo named **`ai_10022200118`**.
3. Push this project to it:
   ```bash
   cd ai_project
   git init
   git add .
   git commit -m "initial submission"
   git branch -M main
   git remote add origin https://github.com/<you>/ai_10022200118.git
   git push -u origin main
   ```
4. On the repo page → Settings → Collaborators → **Invite GodwinDansoAcity** (or `godwin.danso@acity.edu.gh`).

### Deploy

1. <https://share.streamlit.io> → **New app**
2. Repository: `<you>/ai_10022200118`
3. Branch: `main`
4. Main file path: `app/streamlit_app.py`
5. **Advanced settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "gsk_your_real_key_here"
   ```
6. Click **Deploy**. First boot takes ~3 minutes (downloads MiniLM + datasets).

### Submit

Email `godwin.danso@acity.edu.gh`:

- Subject: `CS4241-Introduction to Artificial Intelligence-2026:10022200118 Obour Adomako Tawiah`
- GitHub URL: `https://github.com/<you>/ai_10022200118`
- Deployed URL: `https://<your-app>.streamlit.app`
- Video: Loom / Google Drive link (≤2 min)
- Experiment logs: link to `experiment_logs/` folder
- Architecture: link to `architecture/architecture.svg`

## Alternative deployments

### Render (free web service)
1. Connect your GitHub repo to <https://dashboard.render.com>.
2. Service type: **Web Service**.
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run app/streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
5. Add `GROQ_API_KEY` as an env var.

### Hugging Face Spaces
1. Create a new Space → Streamlit → CPU basic (free).
2. Push this repo contents.
3. Add `GROQ_API_KEY` under Settings → Secrets.

## Local dev

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py        # once
python scripts/build_index.py          # once
cp .env.example .env                   # add GROQ_API_KEY
streamlit run app/streamlit_app.py
```

Then open <http://localhost:8501>.
