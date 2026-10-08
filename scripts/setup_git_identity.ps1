# One-time setup so YOUR commits always use Dilnuka's GitHub-linked identity.
# Run from repo root:
#   powershell -ExecutionPolicy Bypass -File scripts/setup_git_identity.ps1

$ErrorActionPreference = "Stop"
$root = git rev-parse --show-toplevel
Set-Location $root

$name = "Dilnuka Liyanage"
$email = "114978797+Dilnuka@users.noreply.github.com"

Write-Host "Setting local git identity for this repo..."
git config --local user.name $name
git config --local user.email $email
# Copy hooks into .git/hooks instead of pointing core.hooksPath at .githooks.
# .githooks is versioned, so with hooksPath the hook in force changes with every
# checkout: a branch cut from an older main silently brings back old hook bugs.
# Copies in .git/hooks stay put across branches. Re-run after .githooks changes.
git config --local --unset core.hooksPath
# Absolute path: .NET file APIs ignore Set-Location. --git-common-dir also works from a worktree.
$hookDir = Join-Path (git rev-parse --path-format=absolute --git-common-dir) "hooks"
New-Item -ItemType Directory -Force $hookDir | Out-Null

$hooks = @("pre-commit", "prepare-commit-msg", "pre-push")
foreach ($h in $hooks) {
  $path = Join-Path $root ".githooks\$h"
  if (Test-Path $path) {
    $text = [System.IO.File]::ReadAllText($path) -replace "`r`n", "`n"
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText((Join-Path $hookDir $h), $text, $utf8NoBom)
  }
}

Write-Host ""
Write-Host "OK - commits in this repo will use:"
Write-Host ("  name : " + (git config --local user.name))
Write-Host ("  email: " + (git config --local user.email))
Write-Host ("  hooks: " + $hookDir + " (copied from .githooks)")
Write-Host ""
Write-Host "Also turn OFF commit co-author / attribution in your editor settings"
Write-Host "so tool names are never added to commit messages."
Write-Host ""
Write-Host "Verify anytime with: git config user.name ; git config user.email"
