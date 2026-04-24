# MinIO on VPS (S3-compatible storage) — quick reference

This project can use S3-compatible object storage via the existing `R2_*` environment variables (historically Cloudflare R2). On the Hostinger VPS, we run **MinIO** locally and point those variables to MinIO.

## Where data is stored on disk

- MinIO data root: `/srv/minio/data`
- Bucket name used by the app: `neet-documents` (matches `R2_BUCKET_NAME`)
- Objects appear under: `/srv/minio/data/neet-documents/...`

> MinIO may show “virtual folders” (prefixes). The real files are still under `/srv/minio/data/...`.

## Key services

- **MinIO API:** `127.0.0.1:9000` (S3 API)
- **MinIO Console (optional):** `127.0.0.1:9001` (only expose via SSH tunnel or protected reverse proxy)

## Typical install layout (reference)

- Binary: `/usr/local/bin/minio`
- Client: `/usr/local/bin/mc`
- Service file: `/etc/systemd/system/minio.service`
- Env file: `/etc/default/minio`
- Linux user: `minio-user`
- Data directory owner: `minio-user:minio-user`

## Backend environment variables (app)

In `backend/.env`, point storage to MinIO:

```env
R2_ENDPOINT_URL=http://127.0.0.1:9000
R2_BUCKET_NAME=neet-documents
R2_ACCESS_KEY_ID=minioadmin
R2_SECRET_ACCESS_KEY=YOUR_MINIO_ROOT_PASSWORD
```

After changes:

```bash
sudo systemctl restart smart-rag-backend
```

## Service control (MinIO)

```bash
sudo systemctl status minio
sudo systemctl restart minio
sudo systemctl stop minio
sudo systemctl start minio
```

## Logs (MinIO)

```bash
sudo journalctl -u minio -f
sudo journalctl -u minio -n 200 --no-pager
```

## mc (MinIO Client) basics

Configure alias (example — use your real root user/password from `/etc/default/minio`):

```bash
sudo mc alias set local http://127.0.0.1:9000 minioadmin 'YOUR_MINIO_ROOT_PASSWORD'
```

List bucket top-level:

```bash
sudo mc ls local/neet-documents
```

Find objects recursively:

```bash
sudo mc find local/neet-documents
```

## Browsing on the Linux filesystem (not required, but useful)

```bash
cd /srv/minio/data/neet-documents
ls
```

Find PDFs:

```bash
sudo find /srv/minio/data/neet-documents -type f -iname '*.pdf' | head
```

> For names with spaces/`&`, always quote paths:
> `cd "/srv/minio/data/neet-documents/Jammu & Kashmir"`

## App stack restart (common)

```bash
sudo systemctl restart smart-rag-backend
sudo systemctl restart smart-rag-frontend
sudo systemctl reload nginx
```

## Security notes

- Do not expose MinIO API (`9000`) or console (`9001`) publicly without auth + firewall rules.
- Prefer keeping MinIO bound to localhost and using the app as the access layer.
- Rotate credentials if they were ever shared or committed.
