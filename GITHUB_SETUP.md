# GitHub Repository Setup Guide

## Step 1: Create Repository on GitHub

1. Go to [GitHub](https://github.com) and sign in
2. Click the **"+"** icon in the top right, then select **"New repository"**
3. Fill in the repository details:
   - **Repository name**: `AtlasVLA` (or your preferred name)
   - **Description**: "Vision-Language-Action model based on VGGT for robot manipulation"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click **"Create repository"**

## Step 2: Connect Local Repository to GitHub

After creating the repository on GitHub, you'll see instructions. Run these commands:

```bash
cd /Users/tdu/Documents/GitHub/AtlasVLA

# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/AtlasVLA.git

# Or if you prefer SSH:
# git remote add origin git@github.com:YOUR_USERNAME/AtlasVLA.git

# Rename branch to main (if needed)
git branch -M main

# Make initial commit
git commit -m "Initial commit: AtlasVLA project setup"

# Push to GitHub
git push -u origin main
```

## Step 3: Verify

1. Go to your repository on GitHub
2. You should see all your files there
3. The README.md should be displayed on the main page

## Step 4: Optional - Add Repository Topics

On your GitHub repository page:
1. Click the gear icon next to "About"
2. Add topics like: `vla`, `vision-language-action`, `robotics`, `vggt`, `manipulation`, `pytorch`

## Step 5: Optional - Set Up GitHub Actions

The CI workflow (`.github/workflows/ci.yml`) is already included. It will run automatically on:
- Push to main/master branch
- Pull requests

## Step 6: Optional - Add Badges to README

You can add badges to your README.md. Here's an example:

```markdown
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)
```

## Important Notes

### VGGT Submodule

If `vggt/` is a git submodule from the original VGGT repository:

```bash
# If vggt is a submodule, add it properly:
git submodule add https://github.com/facebookresearch/vggt.git vggt
git commit -m "Add VGGT as submodule"
```

### Large Files

If you have large model files or datasets, consider using:
- [Git LFS](https://git-lfs.github.com/) for large files
- Or exclude them from git (already in .gitignore)

### Private Data

Make sure no sensitive data is committed:
- API keys
- Personal credentials
- Private datasets
- Model checkpoints (if private)

All of these should be in `.gitignore`.

## Next Steps

After setting up the repository:

1. **Add collaborators** (if working in a team)
2. **Set up branch protection** (Settings → Branches)
3. **Enable Issues and Discussions** (Settings → General)
4. **Add a description** to the repository
5. **Create releases** when you have stable versions

## Troubleshooting

### Authentication Issues

If you get authentication errors:

```bash
# For HTTPS, use a personal access token
# Or switch to SSH:
git remote set-url origin git@github.com:YOUR_USERNAME/AtlasVLA.git
```

### Push Rejected

If push is rejected:
```bash
# Pull first, then push
git pull origin main --allow-unrelated-histories
git push -u origin main
```
