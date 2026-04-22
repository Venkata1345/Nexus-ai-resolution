# Deploy Nexus to AWS EC2 (free tier)

This is the full playbook for standing up Nexus on an AWS EC2 `t3.micro`
instance. Images are built on your laptop and pushed to GitHub Container
Registry (`ghcr.io`); the EC2 instance pulls them and runs them via
`docker compose`.

Total time the first time: ~1.5-2 hours.

---

## Architecture

```
  [your laptop]                          [ghcr.io]                    [AWS EC2 t3.micro]
    docker build         push          (pull when)                    docker compose up
  ┌───────────────┐   ─────────►    ┌──────────────┐   ─────────►   ┌────────────────────┐
  │ nexus-api     │                 │ nexus-api    │                │ api container      │
  │ nexus-ui      │                 │ nexus-ui     │                │ ui container       │
  └───────────────┘                 └──────────────┘                └─────────┬──────────┘
                                                                              │
                                                              ┌───────────────┴──────────┐
                                                              │  security group          │
                                                              │    22 (ssh)              │
                                                              │    80 (ui)               │
                                                              │    8000 (api docs)       │
                                                              └──────────┬───────────────┘
                                                                         │
                                                              http://<ec2-public-ip>/
```

---

## Part A — on your laptop (build + push images)

### A1. Generate the inference bundle

The deployed image is self-contained (no MLflow at runtime). First, export
the @production model + KB as a flat folder:

```bash
# Re-register the best model if you haven't already (one-time):
python -m src.router.register

# Export the bundle:
python -m src.router.bundle
# -> writes ./bundle/ with xgb_model.ubj, label_encoder.pkl, kb/, metadata.json
```

### A2. Authenticate to GitHub Container Registry

Create a **Personal Access Token (classic)** at
<https://github.com/settings/tokens> with scopes: `write:packages`,
`read:packages`, `delete:packages`.

Save the token somewhere safe — you'll only see it once. Then:

```bash
# On your laptop.
echo <YOUR_TOKEN> | docker login ghcr.io -u <YOUR_GITHUB_USERNAME> --password-stdin
```

### A3. Build and push both images

Replace `<USER>` with your lowercase GitHub username:

```bash
# Build with the ghcr tag baked in.
IMAGE_REGISTRY=ghcr.io/<USER> TAG=latest docker compose build

# Push both images:
docker push ghcr.io/<USER>/nexus-api:latest
docker push ghcr.io/<USER>/nexus-ui:latest
```

Each image is ~1 GB. First push is slow (~10-15 min on home internet);
subsequent pushes only transfer changed layers.

### A4. Make the packages public

By default GHCR packages are private. Go to
<https://github.com/<USER>?tab=packages>, click each of `nexus-api` and
`nexus-ui`, go to **Package settings** → **Change visibility** → **Public**.

(Public means your EC2 instance can pull without credentials. Keep them
private if you prefer; then step B5 also needs `docker login ghcr.io`.)

---

## Part B — on AWS (launch instance + run)

### B1. Launch a t3.micro

AWS Console → EC2 → **Launch instance**:

| Setting | Value |
|---|---|
| Name | `nexus-demo` |
| AMI | **Ubuntu Server 22.04 LTS (x86_64)** |
| Instance type | `t3.micro` |
| Key pair | Create or pick one (you'll need the `.pem` to SSH) |
| Network | Default VPC, auto-assign public IP |
| Security group | Create new, see B2 |
| Storage | **20 GiB** gp3 (default 8 GiB is too small for our image + swap) |

Launch. Wait ~30 seconds for the instance to reach **Running** state.

### B2. Security group inbound rules

Add these three rules:

| Type | Port | Source | Purpose |
|---|---|---|---|
| SSH | 22 | My IP | Your SSH access |
| Custom TCP | 80 | Anywhere (0.0.0.0/0) | Streamlit UI |
| Custom TCP | 8000 | Anywhere (0.0.0.0/0) | FastAPI (optional, for `/docs`) |

### B3. SSH in

```bash
# From your laptop:
chmod 400 ~/path/to/your-key.pem
ssh -i ~/path/to/your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

### B4. Install Docker + add swap

```bash
# Update and install Docker.
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose-v2

# Run docker without sudo for the 'ubuntu' user.
sudo usermod -aG docker ubuntu
newgrp docker   # re-login group without reconnecting

# Verify.
docker --version
docker compose version
```

Add 2 GB of swap so large pip / container starts don't OOM-kill:

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h   # confirm 2 Gi swap
```

### B5. Pull the stack

```bash
# Clone the repo (only the docker-compose.yml is actually needed, but
# cloning the whole thing is easier).
git clone https://github.com/<USER>/Nexus-ai-resolution.git nexus
cd nexus

# Create the .env with your secrets.
cat > .env <<'EOF'
GEMINI_API_KEY=<paste-your-real-key>
IMAGE_REGISTRY=ghcr.io/<USER>
TAG=latest
EOF

# If your GHCR packages are PRIVATE, first:
#   docker login ghcr.io -u <USER>

# Pull pre-built images.
docker compose pull

# Run detached.
docker compose up -d
```

First `docker compose pull` downloads ~1 GB; takes 2-5 min on EC2. First
`up` takes another 10-20 seconds to initialise containers.

### B6. Smoke test from your laptop

```bash
# UI:
open http://<EC2_PUBLIC_IP>/
# or:    curl -s http://<EC2_PUBLIC_IP>/_stcore/health

# API docs:
open http://<EC2_PUBLIC_IP>:8000/docs
# or:    curl -s http://<EC2_PUBLIC_IP>:8000/health
```

Both should respond. The UI's sidebar should light up with the model
panel once you send a message.

### B7. Persistence across reboots

`docker compose up -d` combined with `restart: unless-stopped` in the
compose file means containers come back if they crash. To survive an
instance reboot:

```bash
# systemd unit that brings the stack up on boot.
sudo tee /etc/systemd/system/nexus.service >/dev/null <<'EOF'
[Unit]
Description=Nexus docker-compose stack
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/ubuntu/nexus
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable nexus
sudo systemctl start nexus
sudo systemctl status nexus
```

---

## Operational cheat-sheet

```bash
# Tail logs
docker compose logs -f            # both services
docker compose logs -f api        # API only

# Restart a single service
docker compose restart api

# Redeploy after you push new images from your laptop
docker compose pull && docker compose up -d

# Memory + disk check
free -h
df -h

# If the UI is unreachable, check the containers:
docker compose ps
```

---

## Costs + expiry

- **Free tier (first 12 months):** `t3.micro` with 20 GiB EBS is $0.
- **After 12 months:** ~$8/mo for the instance + ~$2/mo storage.
- **Set a calendar reminder** for ~11 months out to decide: renew with a
  paid account, migrate to another provider, or tear down.

To tear down cleanly:
1. EC2 console → Instances → terminate the instance.
2. EC2 console → Volumes → delete any "available" volumes left behind.
3. GHCR → optional: delete the two package images.
