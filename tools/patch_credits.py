#!/usr/bin/env python3
"""Rewrite one `git format-patch` file for patches/ (called by tools/export_patches.sh).

usage: patch_credits.py PATCH CREDITS_TSV PUBLIC_IDENT NOTICES_ADDED

- Drops "(cherry picked from commit ...)" lines from the message. Nothing else in the message changes.
- A commit authored by MiaAI-Lab must be one of her TensorFold pull-request commits in CREDITS_TSV (matched by
  subject); its notes name the PR and her upstream commit.
- Any other commit must be authored by PUBLIC_IDENT. A port of her recipe patches ("MiaAI-Lab recipe patch(es)
  NNNN" in the subject) must keep "Co-authored-by: MiaAI-Lab", and its notes name the recipe patches, the
  repository, the recipe commit and the licence. The recipe commit comes from "TensorFold @ <sha>" in the message,
  else from a ".../tree/<sha>" link the commit adds to THIRD_PARTY_NOTICES.md (NOTICES_ADDED, a file of those lines),
  else from a "port" row of CREDITS_TSV for that exact subject (a tests-only follow-up to a port).
- A change of this recipe's own to her ported code (an "extends" row of CREDITS_TSV for that exact subject) must keep
  "Co-authored-by: MiaAI-Lab" too; its notes name her patches and recipe commit, and say the change is this recipe's.
- The notes go right after the "---" line: `git am` ignores them, readers see them under the message.
Prints one summary line: kind, author, subject.
"""
import re
import sys

RECIPE_URL = "https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold"
MIA = "MiaAI-Lab <MiaAI-Lab@users.noreply.github.com>"


def die(msg):
    sys.exit(f"patch_credits: {msg}")


def main():
    path, tsv, ident, added = sys.argv[1:5]
    prs, recipes, ports, extends = {}, {}, {}, {}
    for line in open(tsv, encoding="utf-8"):
        if not line.strip() or line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if f[0] == "pr":
            prs[f[3]] = (f[1], f[2])
        elif f[0] == "recipe":
            recipes[f[1]] = (f[2], f[3])
        elif f[0] == "port":
            ports[f[2]] = f[1]
        elif f[0] == "extends":
            extends[f[3]] = (f[1], f[2])
    text = open(path, encoding="utf-8").read()
    head, sep, rest = text.partition("\n---\n")
    if not sep:
        die(f"{path}: no '---' line")
    lines = [ln for ln in head.split("\n") if not re.fullmatch(r"\(cherry picked from commit [0-9a-f]{40}\)", ln)]
    while lines and lines[-1] == "":
        lines.pop()
    head = "\n".join(lines)
    m = re.search(r"^From: (.*)$", head, re.M)
    author = m.group(1) if m else die(f"{path}: no From: header")
    sm = re.search(r"^Subject: \[PATCH[^\]]*\] (.*(?:\n [^\n]*)*)", head, re.M)
    subject = re.sub(r"\n ", " ", sm.group(1)) if sm else die(f"{path}: no Subject: header")
    body = head[sm.end():] if sm else ""
    notes = []
    if author == MIA:
        if subject not in prs:
            die(f"{path}: authored by MiaAI-Lab but not one of her pull-request commits in {tsv}: {subject!r}")
        pr, sha = prs[subject]
        kind = f"mia-pr#{pr}"
        notes = [
            f"Credit: MiaAI-Lab's pull request ashhart/TensorFold#{pr}, her commit {sha},",
            f"  https://github.com/ashhart/TensorFold/pull/{pr} (Apache-2.0, as TensorFold). Her authorship is kept.",
        ]
    else:
        if author != ident:
            die(f"{path}: author {author!r} is neither the public identity nor MiaAI-Lab")
        pm = re.search(r"MiaAI-Lab recipe patch(?:es)? ([^)]*)\)", subject)
        if subject in extends:
            short, patches = extends[subject]
            nums = re.findall(r"\b(\d{4})\b", patches)
            if not nums:
                die(f"{path}: no recipe patch numbers in its extends row")
            if not re.search(r"^Co-authored-by: MiaAI-Lab <MiaAI-Lab@users\.noreply\.github\.com>$", body, re.M | re.I):
                die(f"{path}: a change to MiaAI-Lab's ported code without 'Co-authored-by: MiaAI-Lab'")
            if short not in recipes:
                die(f"{path}: recipe commit {short} is not in {tsv}")
            full, release = recipes[short]
            kind = "extends"
            notes = [
                f"Credit: this changes MiaAI-Lab's code from her GLM-5.3-Flash EXL3 2x DGX Sparks recipe, "
                f"patch{'es' if len(nums) > 1 else ''} {', '.join(nums)},",
                f"  {RECIPE_URL}",
                f"  at commit {full} ({release}); Apache License 2.0, Copyright 2026 MiaAI-Lab.",
                "  The change is this recipe's; Co-authored-by: MiaAI-Lab is kept in the message.",
            ]
        elif pm:
            nums = re.findall(r"\b(\d{4})\b", pm.group(1))
            if not nums:
                die(f"{path}: no recipe patch numbers in the subject")
            if not re.search(r"^Co-authored-by: MiaAI-Lab <MiaAI-Lab@users\.noreply\.github\.com>$", body, re.M | re.I):
                die(f"{path}: a port of MiaAI-Lab's recipe without 'Co-authored-by: MiaAI-Lab'")
            shas = set(re.findall(r"TensorFold @ ([0-9a-f]{7,40})", body))
            if not shas:
                shas = set(re.findall(r"Sparks-TensorFold/tree/([0-9a-f]{7,40})", open(added, encoding="utf-8").read()))
            if not shas and subject in ports:
                shas = {ports[subject]}
            if len(shas) != 1:
                die(f"{path}: expected one recipe commit, found {sorted(shas) or 'none'}")
            short = shas.pop()[:7]
            if short not in recipes:
                die(f"{path}: recipe commit {short} is not in {tsv}")
            full, release = recipes[short]
            kind = "port"
            notes = [
                f"Credit: ported from MiaAI-Lab's GLM-5.3-Flash EXL3 2x DGX Sparks recipe, patch{'es' if len(nums) > 1 else ''} "
                f"{', '.join(nums)},",
                f"  {RECIPE_URL}",
                f"  at commit {full} ({release}); Apache License 2.0, Copyright 2026 MiaAI-Lab.",
                "  Co-authored-by: MiaAI-Lab is kept in the message.",
            ]
        else:
            kind = "ours"
    out = head + "\n" + sep.rstrip("\n") + "\n"
    if notes:
        out += "\n".join(notes) + "\n\n"
    out += rest
    open(path, "w", encoding="utf-8").write(out)
    print(f"{kind}\t{author.split(' <')[0]}\t{subject}")


if __name__ == "__main__":
    main()
