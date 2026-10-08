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
git config --local core.hooksPath .githooks

$hooks = @("pre-commit", "prepare-commit-msg", "pre-push")
foreach ($h in $hooks) {
  $path = Join-Path $root ".githooks\$h"
  if (Test-Path $path) {
    $text = [System.IO.File]::ReadAllText($path) -replace "`r`n", "`n"
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($path, $text, $utf8NoBom)
  }
}

Write-Host ""
Write-Host "OK - commits in this repo will use:"
Write-Host ("  name : " + (git config --local user.name))
Write-Host ("  email: " + (git config --local user.email))
Write-Host ("  hooks: " + (git config --local core.hooksPath))
Write-Host ""
Write-Host "Also turn OFF commit co-author / attribution in your editor settings"
Write-Host "so tool names are never added to commit messages."
Write-Host ""
Write-Host "Verify anytime with: git config user.name ; git config user.email"
