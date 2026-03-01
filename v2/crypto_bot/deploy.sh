#!/usr/bin/env bash
# ============================================================
#  deploy.sh — CryptoBot Deployment Script for DigitalOcean
#
#  Usage:
#    chmod +x deploy.sh
#    sudo ./deploy.sh
#
#  This script:
#    1. Installs system dependencies
#    2. Creates a 4GB swap file (crucial for XGBoost on 1GB VPS)
#    3. Creates a dedicated system user
#    4. Sets up the Python virtual environment
#    5. Installs requirements
#    6. Configures the systemd service
# ============================================================

set -euo pipefail

DEPLOY_DIR="/opt/crypto_bot"
SERVICE_NAME="crypto_bot"
VENV_DIR="$DEPLOY_DIR/venv"

echo "============================================"
echo "  CryptoBot Deployment Script"
echo "============================================"

# ---- 1. System Dependencies ----
echo ""
echo "[1/6] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv git

# ---- 2. Swap File (4GB — Required for XGBoost training on 1GB VPS) ----
echo ""
echo "[2/6] Configuring swap..."
if [ ! -f /swapfile ]; then
    fallocate -l 4G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    echo "  Created 4GB swap file"
else
    echo "  Swap file already exists"
fi
echo "  Swap: $(swapon --show | tail -1)"

# ---- 3. System User ----
echo ""
echo "[3/6] Creating service user..."
if ! id -u cryptobot &>/dev/null; then
    useradd --system --no-create-home --shell /bin/false cryptobot
    echo "  User 'cryptobot' created"
else
    echo "  User 'cryptobot' already exists"
fi

# ---- 4. Deploy Directory ----
echo ""
echo "[4/6] Setting up deployment directory..."
mkdir -p $DEPLOY_DIR/{data,logs,models}

# Copy project files (skip .env, data, logs)
rsync -av --exclude='.env' --exclude='data/' --exclude='logs/' \
    --exclude='models/*.json' --exclude='models/*.pkl' \
    --exclude='venv/' --exclude='__pycache__/' \
    --exclude='research/data/' --exclude='research/results/' \
    ./ $DEPLOY_DIR/

chown -R cryptobot:cryptobot $DEPLOY_DIR

# ---- 5. Python Environment ----
echo ""
echo "[5/6] Setting up Python virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
fi
$VENV_DIR/bin/pip install --upgrade pip -q
$VENV_DIR/bin/pip install -r $DEPLOY_DIR/requirements.txt -q
echo "  Installed $(wc -l < $DEPLOY_DIR/requirements.txt) packages"

# ---- 6. Systemd Service ----
echo ""
echo "[6/6] Configuring systemd service..."
cp $DEPLOY_DIR/bot.service /etc/systemd/system/${SERVICE_NAME}.service
systemctl daemon-reload
systemctl enable ${SERVICE_NAME}

echo ""
echo "============================================"
echo "  Deployment Complete!"
echo "============================================"
echo ""
echo "  Next Steps:"
echo "  1. Copy your .env file:        cp .env $DEPLOY_DIR/.env"
echo "  2. Copy trained models:         cp models/*.json $DEPLOY_DIR/models/"
echo "  3. Copy historical data:        cp data/*.csv $DEPLOY_DIR/data/"
echo "  4. Edit trading pair:           nano /etc/systemd/system/${SERVICE_NAME}.service"
echo "  5. Start the bot:               sudo systemctl start ${SERVICE_NAME}"
echo "  6. Check status:                sudo systemctl status ${SERVICE_NAME}"
echo "  7. View logs:                   sudo journalctl -u ${SERVICE_NAME} -f"
echo ""
