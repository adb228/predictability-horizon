#!/usr/bin/env bash
# History and content scan of D-065, as rerun in D-075.
#
# Usage: scripts/history_scan.sh [ref]     (default ref: main)
#
# Scans every commit reachable from <ref>: commit identities and messages against
# the e-mail allowlist of CLAUDE.md, and the tree and every diff for e-mail
# addresses, token-like strings, phone-number-like strings, local paths,
# sensitive-looking file names and blobs over 1 MB. Read-only; needs git, grep,
# sed, awk and perl;
# git status runs with --no-optional-locks, so the index is not refreshed.
#
# Masking: every e-mail address that is not on the allowlist (the four addresses
# of CLAUDE.md) is printed as "<address outside the allowlist, redacted>", in every
# section, so the output can be pasted into the public log. The commit hashes in
# section 1 say where such an address is; look at it locally. Author and committer
# names are printed as they are; only addresses are masked. The script names no
# address or word beyond the allowlist: section 10 takes its words from the
# environment,
#   HISTORY_SCAN_WORDS="word1,word2" scripts/history_scan.sh
# and is skipped when that variable is unset.
#
# This file is excluded from the diff and tree sections, since its own patterns
# would match themselves.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
REF="${1:-main}"
SELF='scripts/history_scan.sh'

ALLOW_RE='^(172058618\+adb228@users\.noreply\.github\.com|cursoragent@cursor\.com|noreply@anthropic\.com|noreply@github\.com)$'
EMAIL_RE='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
# Python decorators and similar code that the e-mail pattern picks up.
DECOR_RE='@(pytest|dataclass|property|staticmethod|classmethod|jax|partial|functools)\b'

mask() {
  perl -pe 's/([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})/
    my $a = $1;
    ($a =~ m{\A(172058618\+adb228\@users\.noreply\.github\.com|cursoragent\@cursor\.com|noreply\@anthropic\.com|noreply\@github\.com)\z})
      ? $a : "<address outside the allowlist, redacted>"/gex'
}
indent() { sed 's/^/     /'; }
none_if_empty() { if [ -z "$1" ]; then echo "   none"; else printf '%s\n' "$1"; fi; }
# Added and removed lines of every diff reachable from REF, this file excluded.
diff_lines() { git log "$REF" -p --format= -- . ":(exclude)$SELF" | grep -E '^[+-]' | grep -vE '^(\+\+\+|---) '; }
messages() { git log "$REF" --format=%B; }

echo "scan of $(git rev-parse "$REF") ($REF)   ($(git --no-optional-locks status --porcelain | wc -l | tr -d ' ') uncommitted changes)" | mask
echo "commits reachable from $REF: $(git rev-list --count "$REF")"
echo
echo "1. Commit identities (author | committer):"
git log "$REF" --format='%an <%ae> | %cn <%ce>' | sort | uniq -c | mask
echo "   author or committer e-mails outside the allowlist:"
out=$(git log "$REF" --format='%ae%n%ce' | sort -u | grep -Ev "$ALLOW_RE" | mask | indent)
none_if_empty "$out"
echo "   commits with such an e-mail:"
out=$(git log "$REF" --format='%h %ae %ce %s' |
  ALLOW_RE="$ALLOW_RE" awk '!($2 ~ ENVIRON["ALLOW_RE"] && $3 ~ ENVIRON["ALLOW_RE"])' | mask | indent)
none_if_empty "$out"
echo
echo "2. E-mail addresses in tracked files at $REF (uv.lock and this file excluded):"
out=$(git grep -hoIE "$EMAIL_RE" "$REF" -- . ':(exclude)uv.lock' ":(exclude)$SELF" |
  sed "s/^$REF://" | grep -vE "$DECOR_RE" | sort | uniq -c | mask)
none_if_empty "$out"
echo
echo "3. E-mail addresses in the diffs of all history (decorators filtered):"
out=$(diff_lines | grep -oE "$EMAIL_RE" | grep -vE "$DECOR_RE" | sort | uniq -c | mask)
none_if_empty "$out"
echo
echo "4. E-mail addresses in commit messages:"
out=$(messages | grep -oE "$EMAIL_RE" | sort | uniq -c | mask)
none_if_empty "$out"
echo "   outside the allowlist:"
out=$(messages | grep -oE "$EMAIL_RE" | sort -u | grep -Ev "$ALLOW_RE" | mask | indent)
none_if_empty "$out"
echo
TOK_RE='(ghp_|gho_|ghu_|ghs_|ghr_|github_pat_)[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|xox[abprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|(password|passwd|secret|api[_-]?key|token)[[:space:]]*[:=][[:space:]]*["'"'"'][^"'"'"']{8,}'
echo "5. Token, key and password-like strings:"
echo "   diff lines: $(diff_lines | grep -ciE "$TOK_RE")"
diff_lines | grep -iE "$TOK_RE" | mask | cut -c1-160 | indent
echo "   commit-message lines: $(messages | grep -ciE "$TOK_RE")"
echo
PHONE_RE='(\+?1[ .-])?\(?[2-9][0-9]{2}\)?[ .-][0-9]{3}[ .-][0-9]{4}\b'
echo "6. Phone-number-like strings:"
echo "   diff lines: $(diff_lines | grep -cE "$PHONE_RE")"
diff_lines | grep -E "$PHONE_RE" | mask | cut -c1-160 | indent
echo "   commit-message lines: $(messages | grep -cE "$PHONE_RE")"
echo
PATH_RE='/Users/[A-Za-z0-9_.-]+|/home/[A-Za-z0-9_.-]+|/private/(tmp|var)/|/var/folders/'
echo "7. Local home and temporary paths:"
echo "   diff lines: $(diff_lines | grep -cE "$PATH_RE")"
diff_lines | grep -E "$PATH_RE" | mask | cut -c1-160 | indent
echo "   commit-message lines: $(messages | grep -cE "$PATH_RE")"
echo
echo "8. Sensitive-looking file names, in the tree and anywhere in history (deleted files included):"
ALL=$(git log "$REF" --format= --name-only | sort -u | grep .)
out=$(printf '%s\n' "$ALL" |
  grep -iE '(^|/)(\.env(\..*)?|.*\.pem|.*\.key|id_rsa.*|id_ed25519.*|.*\.p12|.*\.pfx|credentials.*|secrets?(\..*)?|.*\.keystore|\.netrc|\.npmrc|\.pypirc)$' |
  mask | sed 's/^/   /')
none_if_empty "$out"
echo "   files ever present in history: $(printf '%s\n' "$ALL" | grep -c .); tracked now: $(git ls-tree -r --name-only "$REF" | wc -l | tr -d ' ')"
echo
echo "9. Blobs larger than 1 MB anywhere in history:"
git rev-list --objects "$REF" | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' |
  awk '$1 == "blob" && $2 > 1048576 { print "   " $2 " " $3; f = 1 } END { if (!f) print "   none" }' | mask
echo
echo "10. Words from HISTORY_SCAN_WORDS (case-insensitive), by count only:"
if [ -z "${HISTORY_SCAN_WORDS:-}" ]; then
  echo "   skipped (HISTORY_SCAN_WORDS is not set)"
else
  IFS=',' read -r -a words <<< "$HISTORY_SCAN_WORDS"
  for i in "${!words[@]}"; do
    w=${words[$i]}
    echo "   word $((i + 1)): tracked files at $REF: $(git grep -ciIF -e "$w" "$REF" -- . ":(exclude)$SELF" | awk -F: '{ s += $NF } END { print s + 0 }') lines;" \
      "diff lines: $(diff_lines | grep -ciF -e "$w"); commit-message lines: $(messages | grep -ciF -e "$w")"
  done
fi
