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
# Copy hooks into .git/hooks instead of pointing core.hooksPath at .githooks.
# .githooks is versioned, so with hooksPath the hook in force changes with every
# checkout: a branch cut from an older main silently brings back old hook bugs.
# Copies in .git/hooks stay put across branches. Re-run after .githooks changes.
git config --local --unset core.hooksPath || true
HOOK_DIR="$(git rev-parse --git-common-dir)/hooks"
mkdir -p "$HOOK_DIR"
for h in pre-commit prepare-commit-msg pre-push; do
  tr -d '\r' < ".githooks/$h" > "$HOOK_DIR/$h"
  chmod +x "$HOOK_DIR/$h"
done

echo ""
echo "OK — commits in this repo will use:"
echo "  name : $(git config --local user.name)"
echo "  email: $(git config --local user.email)"
echo "  hooks: $HOOK_DIR (copied from .githooks)"
echo ""
echo "Also turn OFF commit co-author / attribution in your editor settings"
echo "so tool names are never added to commit messages."
echo ""
echo "Verify anytime with:  git config user.name; git config user.email"
