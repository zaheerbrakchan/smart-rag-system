# Deploy: Render (backend) + Vercel (frontend)

Repo: [smart-rag-system](https://github.com/zaheerbrakchan/smart-rag-system) (monorepo: `backend/` + `frontend/`).

---

## Part A — Backend on Render (do this first)

1. Sign in at [render.com](https://render.com) (GitHub login is fine).
2. **Dashboard → New + → Blueprint** (or **Web Service** if you prefer manual setup).
   - **Blueprint:** connect repo `zaheerbrakchan/smart-rag-system`. Render will read [`render.yaml`](./render.yaml) at the repo root.
   - **Web Service (manual):**
     - Connect the same GitHub repo.
     - **Root Directory:** `backend`
     - **Runtime:** Python 3
     - **Build Command:** `pip install -r requirements.txt`
     - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
3. In the service **Environment** tab, set:
   - **`OPENAI_API_KEY`** — your OpenAI secret key (required).
   - *(Optional)* **`ALLOWED_ORIGINS`** — e.g. `https://your-app.vercel.app` (comma-separated if multiple).  
     By default, CORS also allows `https://*.vercel.app` for previews.
4. Click **Create Web Service** / deploy. Wait until the service is **Live**.
5. Copy your API URL, e.g. `https://smart-rag-api.onrender.com` (yours may differ).

**Free tier:** the service **spins down** after idle time; the first request after sleep can take **30–60+ seconds**.

**Do not** commit `backend/.env` — only set secrets in Render.

---

## Part B — Frontend on Vercel

1. Sign in at [vercel.com](https://vercel.com) with GitHub.
2. **Add New → Project** → import `zaheerbrakchan/smart-rag-system`.
3. Configure the project:
   - **Root Directory:** click **Edit** → set to **`frontend`** (important).
   - Framework: **Next.js** (auto-detected).
4. **Environment Variables** (Production — and Preview if you want previews to work):
   - **`NEXT_PUBLIC_API_URL`** = your Render URL **without** a trailing slash, e.g.  
     `https://smart-rag-api.onrender.com`
5. **Deploy**.

6. Open your Vercel URL and test the chat. If the browser blocks requests, check:
   - `NEXT_PUBLIC_API_URL` matches your live Render URL exactly (`https`, no trailing `/`).
   - Render logs for errors (e.g. missing `OPENAI_API_KEY`).

---

## After you change env vars

- **Vercel:** Redeploy (or push a commit) so `NEXT_PUBLIC_*` is baked into the build.
- **Render:** Restart the service or trigger a manual deploy after changing env.

---

## Files added for deploy

| File | Purpose |
|------|---------|
| [`render.yaml`](./render.yaml) | Render Blueprint (web service + `rootDir: backend`) |
| [`backend/runtime.txt`](./backend/runtime.txt) | Python version hint for Render |
| [`frontend/vercel.json`](./frontend/vercel.json) | Vercel build hints for Next.js |

---

## Custom domain (optional)

If you use a custom domain on Vercel, add it to Render env **`ALLOWED_ORIGINS`**, e.g. `https://www.yourdomain.com`.

Set **`CORS_ALLOW_VERCEL_PREVIEWS=false`** on Render if you want to disable `*.vercel.app` and allow only listed origins.
