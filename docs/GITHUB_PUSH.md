# GitHub authentication and safe push commands

The local branch already contains the project history. Do not reset, amend, or force-push it to get the commits online.

## 1. Reauthenticate GitHub CLI

Run in the user's Terminal (not a shell where device-login networking is blocked):

```bash
gh auth logout --hostname github.com --user neelesh-kotte
gh auth login --hostname github.com --git-protocol https --web
gh auth status
gh api user --jq .login
```

Complete the browser/device prompt. Confirm `gh api user` prints `neelesh-kotte`. Never paste a token or device code into project files or chat.

## 2. Check local branch, commits, identity, and remote

```bash
cd /Users/neeleshkotte/MISHEAR-medication-asr-safety
git status --short --branch
git branch -vv
git log --oneline --decorate -5
git log --format='%h %an <%ae>' -3
git config --show-origin --get-regexp '^(user\.name|user\.email|remote\.origin\.(url|fetch)|branch\.main\.)' || true
git remote -v
```

The expected local chain is `00d35f8` → `385e82f` → `bd896b2`. These commits were created with the local author `MISHEAR Research <local@mishear.invalid>`. That does not prevent pushing. It does mean GitHub will display that commit author; set your real local identity for future commits if desired:

```bash
git config --local user.name "Your Name"
git config --local user.email "your-github-email-or-noreply-address"
```

This does not rewrite existing commits.

## 3. Configure the intended remote

```bash
git remote get-url origin
```

If no `origin` exists:

```bash
git remote add origin https://github.com/neelesh-kotte/MISHEAR-medication-asr-safety.git
```

If `origin` exists but points elsewhere, correct it only after confirming the intended destination:

```bash
git remote set-url origin https://github.com/neelesh-kotte/MISHEAR-medication-asr-safety.git
git remote -v
```

## 4. Push without overwriting remote history

Check whether the GitHub repository is empty:

```bash
git ls-remote --heads origin
```

If it prints no refs, it is empty; push the existing `main` history:

```bash
git push --dry-run -u origin main
git push -u origin main
```

If `main` exists remotely, inspect before pushing:

```bash
git fetch origin
git log --oneline --graph --decorate --all -12
git merge-base --is-ancestor origin/main main
echo $?
```

Use ancestry to distinguish safe histories:

```bash
if git merge-base --is-ancestor origin/main main; then
  git push --dry-run -u origin main
  git push -u origin main
elif git merge-base --is-ancestor main origin/main; then
  git merge --ff-only origin/main
  git push -u origin main
else
  echo "Histories diverge or are unrelated; inspect the graph before integrating."
fi
```

The first branch pushes only when the remote tip is an ancestor of local `main`. The second branch fast-forwards local `main` when GitHub contains later commits on the same history. If histories diverge, do not force-push. Inspect the graph and merge legitimate remote changes after review. If the only remote commit is an auto-created README/license and the histories are unrelated, preserve both histories with a reviewed merge:

```bash
git merge --no-commit --allow-unrelated-histories origin/main
# Resolve any reported conflicts, inspect the staged result, then:
git status
git diff --cached
git commit -m "Merge initial GitHub repository history"
git push -u origin main
```

After pushing, verify:

```bash
git status --short --branch
git ls-remote --heads origin main
git log --oneline --decorate -3
```
