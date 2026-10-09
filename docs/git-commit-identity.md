# Git commit identity (Dilnuka / Component 2)

Commits must use your **GitHub-linked** name and email so GitHub shows your
profile (avatar + contribution graph). Placeholder emails and extra co-author
lines make work look like it belongs to someone else.

## Required identity

| Field | Value |
|---|---|
| Name | `Dilnuka Liyanage` |
| Email | `114978797+Dilnuka@users.noreply.github.com` |
| GitHub | `Dilnuka` |

That noreply address is tied to your GitHub account and is safe to use in public repos.

## One-time setup on your laptop

From the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_git_identity.ps1
```

or:

```bash
bash scripts/setup_git_identity.sh
```

This sets **local** `user.name` / `user.email` for this repo and copies the hooks in
`.githooks/` into `.git/hooks/`.

The hooks are copied rather than enabled with `core.hooksPath .githooks` on purpose:
`.githooks/` is versioned, so with `hooksPath` the hook in force would change on every
checkout, and a branch cut from an older `main` would bring old hook bugs back.
**Re-run the setup script whenever `.githooks/` changes.**

## What the hooks enforce

| Hook | Behaviour |
|---|---|
| `pre-commit` | Blocks commits if name/email is missing, placeholder, or not matching `.gitidentity` |
| `prepare-commit-msg` | Removes unwanted auto-inserted co-author / tool attribution lines |
| `pre-push` | Blocks push if any commit in the range still has a bad author email |

## Editor settings (do this once)

In your editor: turn **off** any setting that adds co-author or “made with …” lines to commits/PRs (often under Agent / Attribution / Git).

If that stays on, GitHub may list an extra author next to you even when `user.email` is correct.

## Check before every demo / viva

```bash
git config user.name
git config user.email
git log -5 --format="%h %an <%ae> %s"
```

You should only see `Dilnuka Liyanage <114978797+Dilnuka@users.noreply.github.com>` for your commits — no `example.com`, no extra co-author names.

## Note for teammates

`.gitidentity` and `scripts/setup_git_identity.*` are for **Dilnuka’s machine**.
Other members should set their own verified GitHub name/email; they can still enable `.githooks` for the co-author stripper, or leave hooks unset.
