# VPS Deployment Runbook (Hostinger)

This is the exact runbook for your current setup on Hostinger VPS.

- VPS Hostname: `srv1610043.hstgr.cloud`
- VPS IPv4: `187.127.155.115`
- App Linux user: `gmuadmin`
- Repo path: `/home/gmuadmin/apps/smart-rag-system/smart-rag-system`
- Backend path: `/home/gmuadmin/apps/smart-rag-system/smart-rag-system/backend`
- Frontend path: `/home/gmuadmin/apps/smart-rag-system/smart-rag-system/frontend`

---

## 1) What is already done

- VPS purchased and reachable.
- `gmuadmin` user created and added to sudo group.
- Python/Node/Nginx/PostgreSQL installed.
- PostgreSQL 14 running on VPS.
- `pgvector` extension installed and enabled in DB.
- DB/user created:
  - DB: `gmu_db`
  - User: `gmu_db_user`
- Repo cloned to VPS.
- Backend venv created and requirements installed.
- Backend systemd service created and running.
- Frontend built and manually verified on `127.0.0.1:3000`.
- Frontend systemd service created.
- Nginx reverse proxy added.
- Public access by IP working: `http://187.127.155.115`

---

## 2) Production service files

### Backend service

File: `/etc/systemd/system/smart-rag-backend.service`

Expected:

```ini
[Unit]
Description=Smart RAG Backend
After=network.target

[Service]
User=gmuadmin
Group=gmuadmin
WorkingDirectory=/home/gmuadmin/apps/smart-rag-system/smart-rag-system/backend
Environment="PATH=/home/gmuadmin/apps/smart-rag-system/smart-rag-system/backend/.venv/bin"
ExecStart=/home/gmuadmin/apps/smart-rag-system/smart-rag-system/backend/.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000 --proxy-headers --http h11
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

### Frontend service

File: `/etc/systemd/system/smart-rag-frontend.service`

Expected:

```ini
[Unit]
Description=Smart RAG Frontend (Next.js)
After=network.target

[Service]
User=gmuadmin
Group=gmuadmin
WorkingDirectory=/home/gmuadmin/apps/smart-rag-system/smart-rag-system/frontend
Environment=NODE_ENV=production
ExecStart=/usr/bin/npm run start -- -p 3000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

---

## 3) Environment values (backend)

In `backend/.env`, keep these key values:

```env
DATABASE_URL=postgresql+asyncpg://gmu_db_user:YOUR_DB_PASSWORD@127.0.0.1:5432/gmu_db
DB_SSL_MODE=disable
DEBUG=false
PYTHONUNBUFFERED=1

V2_STREAM_TOKEN_DELAY_SEC=0
V2_FINAL_MAX_TOKENS=500
V2_TOOL_CONTEXT_CHAR_LIMIT=8000
V2_TIMING_LOG=true
```

For current IP-based access:

```env
ALLOWED_ORIGINS=http://187.127.155.115
```

When domain is ready, switch to:

```env
ALLOWED_ORIGINS=https://medbuddy.getmyuniversity.com
```

---

## 4) Environment values (frontend)

File: `frontend/.env.production`

Current IP-based:

```env
NEXT_PUBLIC_API_URL=http://187.127.155.115/api
NEXT_PUBLIC_USE_V2_CHAT=true
```

After domain + SSL:

```env
NEXT_PUBLIC_API_URL=https://medbuddy.getmyuniversity.com/api
NEXT_PUBLIC_USE_V2_CHAT=true
```

After changing frontend env, always rebuild:

```bash
cd /home/gmuadmin/apps/smart-rag-system/smart-rag-system/frontend
npm run build
sudo systemctl restart smart-rag-frontend
```

---

## 5) Nginx config (required for public access + streaming)

Example file: `/etc/nginx/sites-available/smart-rag`

```nginx
server {
    listen 80;
    server_name 187.127.155.115 medbuddy.getmyuniversity.com;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE-critical settings
        proxy_buffering off;
        proxy_cache off;
        gzip off;
        chunked_transfer_encoding on;
        proxy_read_timeout 3600;
        proxy_send_timeout 3600;
        add_header X-Accel-Buffering no;
    }
}
```

Enable/reload:

```bash
sudo ln -sf /etc/nginx/sites-available/smart-rag /etc/nginx/sites-enabled/smart-rag
sudo nginx -t
sudo systemctl reload nginx
```

---

## 6) Database checks

### Cluster status

```bash
sudo pg_lsclusters
```

Expected: `14 main ... online`.

### Extension + table check

```bash
sudo -u postgres psql -d gmu_db -c "CREATE EXTENSION IF NOT EXISTS vector;"
sudo -u postgres psql -d gmu_db -c "\dt public.data_*"
```

Expected table: `public.data_neet_assistant`.

---

## 7) Daily operations (quick commands)

### Restart all

```bash
sudo systemctl restart smart-rag-backend
sudo systemctl restart smart-rag-frontend
sudo systemctl reload nginx
```

### Status check

```bash
sudo systemctl status smart-rag-backend --no-pager
sudo systemctl status smart-rag-frontend --no-pager
sudo systemctl status nginx --no-pager
sudo pg_lsclusters
```

### Live logs

```bash
sudo journalctl -u smart-rag-backend -f
sudo journalctl -u smart-rag-frontend -f
```

### Health checks

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:3000
curl http://187.127.155.115
```

---

## 8) Tomorrow tasks (domain + SSL)

Domain DNS is managed in GoDaddy (`ns47/ns48.domaincontrol.com`), not Hostinger.

1. In GoDaddy DNS, add:
   - Type: `A`
   - Host: `medbuddy`
   - Value: `187.127.155.115`
2. Verify:
   - `nslookup medbuddy.getmyuniversity.com`
3. Update Nginx `server_name` to domain.
4. Issue SSL:
   - `sudo certbot --nginx -d medbuddy.getmyuniversity.com`
5. Update env values:
   - Frontend `NEXT_PUBLIC_API_URL=https://medbuddy.getmyuniversity.com/api`
   - Backend `ALLOWED_ORIGINS=https://medbuddy.getmyuniversity.com`
6. Rebuild/restart services.

---

## 9) Common failures and exact fixes

- `status=203/EXEC` in backend service:
  - Wrong `ExecStart` path in systemd file.
- Postgres `invalid IP mask "scram-sha-256"`:
  - Bad `pg_hba.conf` column order. Fix line format.
- `relation "public.data_neet_assistant" does not exist`:
  - Initialize vector store / upload at least one document.
- Curl upload `Failed to open/read local data`:
  - Wrong local file path in `-F "file=@..."`.
- OpenAI rate limit on large upload:
  - Upload smaller PDFs first and in batches.

---

## 10) Security reminders

- Rotate exposed secrets (OpenAI, DB, Twilio, R2, Supabase, JWT).
- Keep backend bound to `127.0.0.1` (Nginx should be the public entrypoint).
- Avoid opening PostgreSQL `5432` to all IPs; prefer restricted IPs or SSH tunnel.

