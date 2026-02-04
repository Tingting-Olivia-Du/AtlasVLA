# VGGT Submodule

If vggt/ is from the original VGGT repository, you have two options:

## Option 1: Add as Git Submodule (Recommended)

This keeps vggt linked to the original repository:

```bash
git submodule add https://github.com/facebookresearch/vggt.git vggt
git commit -m "Add VGGT as submodule"
```

## Option 2: Include VGGT Code Directly

If you want to include vggt code directly in your repository:

```bash
# Remove .git from vggt if it exists
rm -rf vggt/.git
git add vggt/
git commit -m "Add VGGT code"
```

⚠️ **Note**: Make sure you comply with VGGT's license terms.
