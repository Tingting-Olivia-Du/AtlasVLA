#!/bin/bash
# Script to push AtlasVLA to GitHub
# Run this after creating the repository on GitHub

set -e

echo "🚀 Pushing AtlasVLA to GitHub..."
echo ""

# Check if remote already exists
if git remote get-url origin >/dev/null 2>&1; then
    echo "Remote 'origin' already exists:"
    git remote get-url origin
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git remote set-url origin https://github.com/Tingting-Olivia-Du/AtlasVLA.git
    fi
else
    echo "Adding remote repository..."
    git remote add origin https://github.com/Tingting-Olivia-Du/AtlasVLA.git
fi

# Ensure we're on main branch
git branch -M main

# Show current status
echo ""
echo "Current git status:"
git status --short | head -10
echo ""

# Ask for confirmation
read -p "Ready to push to GitHub? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Push to GitHub
echo ""
echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ Successfully pushed to GitHub!"
echo ""
echo "Your repository is now available at:"
echo "https://github.com/Tingting-Olivia-Du/AtlasVLA"
echo ""
echo "Next steps:"
echo "1. Visit your repository on GitHub"
echo "2. Add repository description and topics"
echo "3. Enable Issues and Discussions (Settings → General)"
echo "4. Consider setting up branch protection (Settings → Branches)"
