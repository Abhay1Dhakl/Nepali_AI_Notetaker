#!/bin/bash

echo "=========================================="
echo "GitHub Authentication Fix for Mac M4 Pro"
echo "=========================================="
echo ""

# Step 1: Show your SSH public key
echo "📋 Your SSH Public Key:"
echo "Copy this ENTIRE key (including ssh-ed25519 at the start):"
echo ""
cat ~/.ssh/id_ed25519_abhay1dhakl.pub
echo ""
echo "=========================================="
echo ""

# Step 2: Instructions
echo "✅ TO FIX GITHUB AUTHENTICATION:"
echo ""
echo "1. Copy the SSH key shown above (the ENTIRE line)"
echo ""
echo "2. Go to: https://github.com/settings/ssh/new"
echo ""
echo "3. In GitHub:"
echo "   - Title: 'Mac Mini M4 Pro'"
echo "   - Key: Paste the SSH key you copied"
echo "   - Click 'Add SSH key'"
echo ""
echo "4. After adding the key to GitHub, come back here and press Enter"
echo ""
read -p "Press Enter after you've added the SSH key to GitHub..."
echo ""

# Step 3: Test SSH connection
echo "🔐 Testing SSH connection to GitHub..."
ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"
if [ $? -eq 0 ]; then
    echo "✅ SSH authentication working!"
else
    echo "⚠️  SSH test: You might see 'Permission denied' - that's okay if you just added the key."
    echo "   It can take a minute for GitHub to update."
fi
echo ""

# Step 4: Update git remote to use SSH
echo "🔧 Switching repository to use SSH..."
cd /Users/assabet_tech/Desktop/Nepali_AI_Notetaker
git remote set-url origin git@github.com:Abhay1Dhakl/Nepali_AI_Notetaker.git
echo "✅ Remote URL updated to SSH"
echo ""

# Step 5: Verify remote URL
echo "📍 Current remote URL:"
git remote -v
echo ""

# Step 6: Try to push
echo "=========================================="
echo "🚀 Ready to push! Run this command:"
echo ""
echo "   git push origin main"
echo ""
echo "=========================================="
