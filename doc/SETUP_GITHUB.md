# 🚀 GitHub 仓库设置完整指南

## 当前状态

你的本地仓库已经准备好，但 GitHub 上的仓库还不存在。

## 📋 完整步骤

### 步骤 1: 创建 GitHub 仓库

1. **访问**: https://github.com/new
2. **填写信息**:
   ```
   Repository name: AtlasVLA
   Description: Vision-Language-Action model based on VGGT for robot manipulation
   Visibility: Public (推荐) 或 Private
   ```
3. **重要**: 
   - ❌ **不要**勾选 "Add a README file"
   - ❌ **不要**勾选 "Add .gitignore"
   - ❌ **不要**勾选 "Choose a license"
   （这些文件我们已经有了）
4. **点击**: "Create repository"

### 步骤 2: 连接本地仓库到 GitHub

创建仓库后，GitHub 会显示设置说明。**不要**按照 GitHub 的说明做，因为我们已经有了提交。

运行以下命令：

```bash
cd /Users/tdu/Documents/GitHub/AtlasVLA

# 添加 remote（如果还没有）
git remote add origin https://github.com/Tingting-Olivia-Du/AtlasVLA.git

# 或者如果 remote 已存在但 URL 错误，先删除再添加：
# git remote remove origin
# git remote add origin https://github.com/Tingting-Olivia-Du/AtlasVLA.git

# 确保在 main 分支
git branch -M main

# 验证 remote 配置
git remote -v
```

### 步骤 3: 推送到 GitHub

```bash
# 推送到 GitHub
git push -u origin main
```

## 🔐 认证设置

### 方式 A: HTTPS + Personal Access Token（推荐用于首次设置）

1. **创建 Token**:
   - 访问: https://github.com/settings/tokens
   - 点击 "Generate new token" → "Generate new token (classic)"
   - 填写名称: `AtlasVLA`
   - 选择权限: ✅ `repo` (全部)
   - 点击 "Generate token"
   - **复制 token**（只显示一次）

2. **推送时使用 token**:
   ```bash
   git push -u origin main
   # Username: Tingting-Olivia-Du
   # Password: <粘贴你的 token>
   ```

### 方式 B: SSH（推荐长期使用）

1. **检查 SSH key**:
   ```bash
   ls -la ~/.ssh/id_ed25519.pub
   # 或
   ls -la ~/.ssh/id_rsa.pub
   ```

2. **如果没有 SSH key，创建一个**:
   ```bash
   ssh-keygen -t ed25519 -C "tingtingdu06@gmail.com"
   # 按 Enter 使用默认路径
   # 可以设置密码或直接 Enter
   ```

3. **添加 SSH key 到 GitHub**:
   ```bash
   # 复制公钥
   cat ~/.ssh/id_ed25519.pub
   # 或
   cat ~/.ssh/id_rsa.pub
   ```
   
   - 访问: https://github.com/settings/keys
   - 点击 "New SSH key"
   - 粘贴公钥内容
   - 点击 "Add SSH key"

4. **切换到 SSH URL**:
   ```bash
   git remote set-url origin git@github.com:Tingting-Olivia-Du/AtlasVLA.git
   git push -u origin main
   ```

## ✅ 验证

推送成功后：

1. 访问: https://github.com/Tingting-Olivia-Du/AtlasVLA
2. 确认所有文件都在那里
3. README.md 应该显示在主页

## 🎯 后续步骤

推送成功后：

1. **添加仓库描述和主题**:
   - 点击 "About" 旁边的 ⚙️
   - 添加描述
   - 添加主题: `vla`, `vision-language-action`, `robotics`, `vggt`, `manipulation`, `pytorch`

2. **启用功能**:
   - Settings → General → Features
   - 启用 Issues
   - 启用 Discussions（可选）

3. **设置分支保护**（可选）:
   - Settings → Branches
   - 添加规则保护 main 分支

## 🆘 常见问题

### Q: 提示 "repository not found"
**A**: 确保已经在 GitHub 上创建了仓库

### Q: 提示 "authentication failed"
**A**: 
- HTTPS: 使用 Personal Access Token 而不是密码
- SSH: 确保 SSH key 已添加到 GitHub

### Q: 提示 "remote origin already exists"
**A**: 
```bash
git remote remove origin
git remote add origin https://github.com/Tingting-Olivia-Du/AtlasVLA.git
```

### Q: 推送被拒绝
**A**: 
```bash
# 如果仓库已存在但为空，强制推送（谨慎使用）
git push -u origin main --force
```

## 📝 快速命令参考

```bash
# 查看 remote
git remote -v

# 更新 remote URL
git remote set-url origin https://github.com/Tingting-Olivia-Du/AtlasVLA.git

# 切换到 SSH
git remote set-url origin git@github.com:Tingting-Olivia-Du/AtlasVLA.git

# 推送
git push -u origin main
```
