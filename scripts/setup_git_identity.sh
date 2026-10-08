#!/usr/bin/env bash
# One-time setup so YOUR commits always use Dilnuka's GitHub-linked identity.
# Run from repo root:  bash scripts/setup_git_identity.sh
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

NAME="Dilnuka Liyanage"
EMAIL="114978797+Dilnuka@users.noreply.github.com"

echo "Setting local git identity for this repo..."
git config --local user.name "$NAME"
git config --local user.email "$EMAIL"
git config --local core.hooksPath .githooks

chmod +x .githooks/pre-commit .githooks/prepare-commit-msg .githooks/pre-push 2>/dev/null || true

echo ""
echo "OK — commits in this repo will use:"
echo "  name : $(git config --local user.name)"
echo "  email: $(git config --local user.email)"
echo "  hooks: $(git config --local core.hooksPath)"
echo ""
echo "Also turn OFF commit co-author / attribution in your editor settings"
echo "so tool names are never added to commit messages."
echo ""
echo "Verify anytime with:  git config user.name; git config user.email"
