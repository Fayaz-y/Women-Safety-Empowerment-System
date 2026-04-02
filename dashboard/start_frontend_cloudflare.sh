#!/bin/bash
# ===================================================================
# Women Safety AI — Frontend Setup & Startup Script (macOS/Linux)
# ===================================================================
# Use this on your friend's computer to:
# 1. Configure the backend URL
# 2. Install dependencies
# 3. Start the Next.js dashboard

# Windows users: Use start_frontend_cloudflare.bat instead

set -e  # Exit on error

echo ""
echo "==================================================================="
echo " Women Safety AI Frontend — Cloudflare Setup"
echo "==================================================================="
echo ""

cd dashboard

# Ask for the Cloudflare domain
read -p "Enter your backend Cloudflare domain (e.g., women-safety.example.com): " DOMAIN

if [ -z "$DOMAIN" ]; then
    echo "ERROR: Domain cannot be empty!"
    exit 1
fi

# Create .env.local
echo "Creating .env.local..."
cat > .env.local << EOF
# Backend API configuration
NEXT_PUBLIC_API_URL=https://${DOMAIN}
NEXT_PUBLIC_WS_URL=wss://${DOMAIN}
EOF

echo "✓ Created .env.local with:"
cat .env.local
echo ""

# Install dependencies
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
else
    echo "✓ node_modules already exists, skipping install"
fi

echo ""
echo "==================================================================="
echo " Starting Next.js Dashboard..."
echo "==================================================================="
echo ""
echo "Frontend will be available at: http://localhost:3000"
echo "Backend is at: https://${DOMAIN}"
echo ""
echo "Press Ctrl+C to stop the development server"
echo ""

npm run dev
