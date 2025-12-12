# 关联远程仓库并推送代码

## 关联远程仓库

请将以下命令中的 `your_username` 替换为你的GitHub用户名，`your_repo` 替换为你在GitHub上创建的仓库名：

```bash
git remote add origin https://github.com/your_username/your_repo.git
```

## 推送代码到GitHub

```bash
git push -u origin main
```

## 身份验证说明

当你执行 `git push` 命令时，GitHub会要求你进行身份验证。推荐使用**个人访问令牌(PAT)**进行身份验证：

1. 生成个人访问令牌：
   - 登录GitHub，点击右上角头像 -> Settings
   - 左侧菜单找到 Developer settings -> Personal access tokens -> Tokens (classic)
   - 点击 Generate new token -> Generate new token (classic)
   - 设置token名称，选择expiration时间，勾选repo权限
   - 点击 Generate token
   - **重要：** 复制生成的token并保存好，离开页面后将无法再次查看

2. 使用token进行身份验证：
   - 当Git提示输入密码时，粘贴你刚才生成的个人访问令牌

## 推送成功后

推送成功后，你可以在GitHub上查看你的代码仓库。

## 后续工作

- 定期使用 `git push` 推送本地代码变更
- 使用 `git pull` 拉取远程仓库的更新

## 常见问题

如果推送失败，可能的原因：
1. 远程仓库地址错误：检查用户名和仓库名是否正确
2. 身份验证失败：确保使用的是正确的个人访问令牌
3. 分支名称不匹配：GitHub默认分支可能是main或master，确保本地分支名称与远程分支名称一致

如果遇到问题，可以尝试以下命令查看详细错误信息：
```bash
git push -v origin main
```