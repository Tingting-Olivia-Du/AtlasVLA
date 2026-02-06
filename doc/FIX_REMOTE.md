# 修复 GitHub Remote 配置

## 问题

你遇到了两个问题：
1. GitHub 仓库还不存在（需要先创建）
2. Remote origin 已经存在但指向错误的 URL

## 解决方案

### 步骤 1: 在 GitHub 上创建仓库

1. 访问 https://github.com/new
2. 填写信息：
   - **Repository name**: `AtlasVLA`
   - **Description**: `Vision-Language-Action model based on VGGT for robot manipulation`
   - **Visibility**: Public 或 Private
   - ⚠️ **重要**: **不要**勾选 "Initialize this repository with a README"
3. 点击 **"Create repository"**

### 步骤 2: 更新或设置 Remote

#### 如果 remote 已存在但 URL 错误：

```bash
cd /Users/tdu/Documents/GitHub/AtlasVLA

# 查看当前的 remote
git remote -v

# 更新 remote URL
git remote set-url origin https://github.com/Tingting-Olivia-Du/AtlasVLA.git

# 验证
git remote -v
```

#### 如果 remote 不存在：

```bash
git remote add origin https://github.com/Tingting-Olivia-Du/AtlasVLA.git
```

### 步骤 3: 推送到 GitHub

```bash
# 确保在 main 分支
git branch -M main

# 推送到 GitHub
git push -u origin main
```

## 如果推送失败

### 认证问题

如果提示需要认证：

**HTTPS 方式**:
- 使用 Personal Access Token（不是密码）
- 创建 token: https://github.com/settings/tokens
- 权限需要: `repo`

**SSH 方式**（推荐）:
```bash
# 切换到 SSH URL
git remote set-url origin git@github.com:Tingting-Olivia-Du/AtlasVLA.git

# 推送到 GitHub
git push -u origin main
```

### 仓库已存在但为空

如果仓库已创建但为空，直接推送即可。

### 仓库已存在且有内容

如果仓库已创建且有初始提交：

```bash
# 先拉取（允许不相关的历史）
git pull origin main --allow-unrelated-histories

# 解决可能的冲突后
git push -u origin main
```

## 快速修复脚本

运行以下命令一键修复：

```bash
cd /Users/tdu/Documents/GitHub/AtlasVLA

# 更新 remote URL
git remote set-url origin https://github.com/Tingting-Olivia-Du/AtlasVLA.git

# 确保在 main 分支
git branch -M main

# 验证 remote
git remote -v

# 然后推送到 GitHub（在创建仓库后）
# git push -u origin main
```
