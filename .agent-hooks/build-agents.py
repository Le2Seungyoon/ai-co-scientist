#!/usr/bin/env python3
"""Generate the per-harness agent lanes and skills from `.agents/**`.

**Why generation and not a pointer.** Both harnesses deliver a lane definition by putting it in
that lane's system prompt, and neither dereferences a path for you:

  - A Claude Code subagent file's body *becomes* the system prompt. `@path` imports are not
    expanded there — an `@.agents/agents/x.md` line would arrive as literal text.
  - Codex's `developer_instructions` is a literal TOML string. Same story.

So a "one file, two pointers" layout would replace a guaranteed injection with "the model has to
remember to open a file" on **both** sides. Instead the source of truth stays single and the
per-harness files are build artifacts of it, committed so the harnesses can read them directly.

    .agents/agents/<name>.md      source: harness-neutral frontmatter + shared body
      |
      +--> .claude/agents/<name>.md    frontmatter (name, description, tools, model) + body
      +--> .codex/agents/<name>.toml   name, description, model, nickname_candidates,
                                       developer_instructions = the same body

    .agents/skills/<name>/**       source: copied verbatim to BOTH harnesses

Harness differences are frontmatter keys (`tools.claude`, `model.claude`, `model.codex`), never
sidecar files. A sidecar invites harness-shaped prose to accumulate in it, and instructions
almost never differ between harnesses — metadata does. There is no per-harness body fragment for
the same reason: a paragraph true of one harness and false of the other belongs *inside* the
shared text, labelled, so a rule and its variant cannot be read apart
(`.agents/rules/harness.md` -> Where a thing lives).

**`tools` has no Codex counterpart.** Codex has no per-lane tool allowlist, so `tools.claude` is
emitted for Claude Code only. Do not invent a Codex spelling for it: a `tools` line there would
read as a restriction nothing enforces. What Codex *does* have is `sandbox_mode`, unmeasured on
this machine — `docs/codex-verification-ledger.md` item 5.

Run: uv run python .agent-hooks/build-agents.py            # regenerate (writes)
     uv run python .agent-hooks/build-agents.py --check    # compare only (writes nothing)
     uv run python .agent-hooks/build-agents.py --hook     # PostToolUse: rebuild iff a source

`--check` exists because a generated file nobody regenerates drifts silently. Its first stdout
line is a machine-readable marker so a caller cannot mistake a Python traceback for drift:

    AGENTS_FRESH            exit 0
    AGENTS_STALE            exit 1   (and every drifted path is named)
    AGENTS_UNKNOWN: <why>   exit 2   (cannot judge -- not the same thing as drift)

Python 3.8 compatible: this runs under the bare `python` on PATH when a harness hook fires, and
that interpreter is 3.8 on this machine (`.agents/rules/harness.md`).
"""
import difflib
import json
import os
import re
import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

SRC_DIR = ".agents/agents"
CLAUDE_DIR = ".claude/agents"
CODEX_DIR = ".codex/agents"

# Skills are copied verbatim to both harnesses -- no per-harness delta, because a skill body is
# read by both and anything true of only one belongs in a labelled paragraph inside it.
SKILLS_SRC = ".agents/skills"
SKILL_OUT = (".claude/skills", ".codex/skills")

# Codex validates SKILL.md frontmatter strictly and rejects unknown keys, so the source is held
# to the intersection both harnesses accept. The two harnesses also disagree on which field is
# the skill's identity -- Claude Code uses the DIRECTORY name and ignores `name:`, Codex uses
# `name:` and ignores the directory. Neither errors on a mismatch; the skill just registers under
# two different names. `load_skill` therefore requires them to be equal.
SKILL_KEYS = ("name", "description")

# Frontmatter keys understood in the source. An unknown key is a hard error rather than a silent
# drop: the whole point of this file is that the source is the only place to edit, so a key that
# generates nothing would be an edit that appears to work and does not.
SRC_KEYS = ("name", "description", "tools.claude", "model.claude", "model.codex",
            "nickname_candidates")
CLAUDE_ONLY = ("tools.claude", "model.claude")
CODEX_ONLY = ("model.codex", "nickname_candidates")
# How a Claude-only dotted key is spelled in the Claude output.
CLAUDE_RENAME = {"tools.claude": "tools", "model.claude": "model"}


class BuildError(Exception):
    """Something under `.agents/**` cannot be turned into an output file."""


# ---------------------------------------------------------------- source parsing


def split_frontmatter(text, path):
    """Return (ordered [(key, raw_value)], body).

    Deliberately not a YAML parser. The values here are plain scalars and one inline list, and a
    real YAML round-trip would requote the Korean descriptions -- which would make the generated
    `.claude/agents/*.md` differ from the hand-written originals for no reason. Raw text in, raw
    text out.
    """
    if not text.startswith("---\n"):
        raise BuildError("%s: must start with a `---` frontmatter fence" % path)
    end = text.find("\n---\n", 3)
    if end == -1:
        raise BuildError("%s: frontmatter fence is never closed" % path)
    block = text[4:end + 1]
    body = text[end + len("\n---\n"):]

    fields = []
    for lineno, line in enumerate(block.splitlines(), start=2):
        if not line.strip():
            continue
        if ":" not in line:
            raise BuildError("%s:%d: frontmatter line is not `key: value`" % (path, lineno))
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        if not key or line[0].isspace():
            raise BuildError("%s:%d: nested frontmatter is not supported" % (path, lineno))
        fields.append((key, value))
    return fields, body


def parse_list(raw, path, key):
    """Parse an inline JSON list frontmatter value, e.g. `["a", "b"]`."""
    try:
        value = json.loads(raw)
    except ValueError:
        raise BuildError("%s: `%s` must be an inline JSON list, got %r" % (path, key, raw))
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise BuildError("%s: `%s` must be a list of strings, got %r" % (path, key, raw))
    return value


def load_source(root, name):
    src_path = os.path.join(root, SRC_DIR, "%s.md" % name)
    with open(src_path, encoding="utf-8") as fh:
        fields, body = split_frontmatter(fh.read(), src_path)

    for key, _ in fields:
        if key not in SRC_KEYS:
            raise BuildError(
                "%s: unknown frontmatter key `%s` (known: %s). Nothing would generate it."
                % (src_path, key, ", ".join(SRC_KEYS))
            )
    meta = dict(fields)
    if meta.get("name", name) != name:
        raise BuildError(
            "%s: frontmatter `name: %s` != filename `%s`" % (src_path, meta.get("name"), name))
    for required in ("name", "description"):
        if required not in meta:
            raise BuildError("%s: frontmatter is missing `%s`" % (src_path, required))
    if not body.strip():
        raise BuildError("%s: body is empty" % src_path)

    if "nickname_candidates" in meta:
        meta["nickname_candidates"] = parse_list(
            meta["nickname_candidates"], src_path, "nickname_candidates")
    return fields, body, meta


def source_names(root):
    d = os.path.join(root, SRC_DIR)
    if not os.path.isdir(d):
        raise BuildError("%s does not exist under %s" % (SRC_DIR, root))
    names = sorted(f[:-len(".md")] for f in os.listdir(d) if f.endswith(".md"))
    if not names:
        raise BuildError("%s holds no agent sources" % SRC_DIR)
    return names


# ---------------------------------------------------------------- renderers


def render_claude(fields, body):
    """`.claude/agents/<name>.md` -- frontmatter then the shared body, and nothing else.

    Emitted in source order so the generated file is byte-identical to the hand-written one it
    replaces. `model:` in particular must survive: each lane's value was chosen deliberately and
    the reasoning sits in a comment at the top of the body.
    """
    lines = ["---"]
    for k, v in fields:
        if k in CODEX_ONLY:
            continue
        lines.append("%s: %s" % (CLAUDE_RENAME.get(k, k), v))
    lines.append("---")
    return "\n".join(lines) + "\n\n" + body.lstrip("\n")


def toml_basic_string(s):
    out = s.replace("\\", "\\\\").replace('"', '\\"')
    out = out.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
    return '"%s"' % out


def toml_multiline_string(s):
    """A TOML `\"\"\"...\"\"\"` literal. Escaped so no body text can terminate it early."""
    out = s.replace("\\", "\\\\").replace('"""', '""\\"')
    if out.endswith('"'):
        out = out[:-1] + '\\"'
    return '"""\n' + out + '\n"""'


def render_codex(body, meta):
    """`.codex/agents/<name>.toml` -- the same body Claude gets, no per-harness delta.

    `name` and `description` are written even though the live copies of both may be the
    `[agents.*]` entry in `.codex/config.toml` rather than this file. Whether Codex reads them
    here is UNMEASURED -- `docs/codex-verification-ledger.md` item 3. If it turns out they are
    inert, delete them here rather than leaving keys that look like configuration and are not.
    """
    lines = [
        "name = %s" % toml_basic_string(meta["name"]),
        "description = %s" % toml_basic_string(meta["description"]),
    ]
    if "model.codex" in meta:
        lines.append("model = %s" % toml_basic_string(meta["model.codex"]))
    if "nickname_candidates" in meta:
        rendered = ", ".join(toml_basic_string(v) for v in meta["nickname_candidates"])
        lines.append("nickname_candidates = [%s]" % rendered)
    lines.append("")
    lines.append("developer_instructions = %s" % toml_multiline_string(body.strip("\n")))
    return "\n".join(lines) + "\n"


def load_skill(root, name):
    """Validate one skill source directory and return {relative path: text} for its files."""
    src_dir = os.path.join(root, SKILLS_SRC, name)
    entry = os.path.join(src_dir, "SKILL.md")
    if not os.path.isfile(entry):
        raise BuildError(
            "%s/%s: no SKILL.md -- both harnesses require that name" % (SKILLS_SRC, name))
    with open(entry, encoding="utf-8") as fh:
        fields, body = split_frontmatter(fh.read(), entry)
    meta = dict(fields)
    for key, _ in fields:
        if key not in SKILL_KEYS:
            raise BuildError(
                "%s: frontmatter key `%s` is not accepted by both harnesses (allowed: %s). "
                "Codex rejects the file outright." % (entry, key, ", ".join(SKILL_KEYS))
            )
    for required in SKILL_KEYS:
        if not meta.get(required):
            raise BuildError("%s: frontmatter is missing `%s`" % (entry, required))
    if meta["name"] != name:
        raise BuildError(
            "%s: frontmatter `name: %s` != directory `%s`. Claude Code would register it as "
            "`%s` and Codex as `%s` -- one skill, two names, no error from either."
            % (entry, meta["name"], name, name, meta["name"])
        )
    if not body.strip():
        raise BuildError("%s: body is empty" % entry)

    files = {}
    for dirpath, _, filenames in os.walk(src_dir):
        for filename in sorted(filenames):
            abspath = os.path.join(dirpath, filename)
            rel = os.path.relpath(abspath, src_dir).replace(os.sep, "/")
            try:
                with open(abspath, encoding="utf-8") as fh:
                    files[rel] = fh.read()
            except UnicodeDecodeError:
                raise BuildError(
                    "%s/%s/%s: not UTF-8 text; cannot be copied" % (SKILLS_SRC, name, rel))
    return files


def skill_names(root):
    d = os.path.join(root, SKILLS_SRC)
    if not os.path.isdir(d):
        return []
    return sorted(n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n)))


def render_skills(root):
    out = {}
    for name in skill_names(root):
        for rel, text in load_skill(root, name).items():
            for target in SKILL_OUT:
                out["%s/%s/%s" % (target, name, rel)] = text
    return out


def orphan_skill_dirs(root):
    """Generated skill directories with no source -- a rename leaves the old name registered."""
    live = set(skill_names(root))
    stale = []
    for target in SKILL_OUT:
        d = os.path.join(root, target)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if os.path.isdir(os.path.join(d, name)) and name not in live:
                stale.append("%s/%s" % (target, name))
    return stale


def render_all(root):
    """Return {relative path: intended contents} for every generated file."""
    out = {}
    for name in source_names(root):
        fields, body, meta = load_source(root, name)
        out["%s/%s.md" % (CLAUDE_DIR, name)] = render_claude(fields, body)
        out["%s/%s.toml" % (CODEX_DIR, name)] = render_codex(body, meta)
    out.update(render_skills(root))
    return out


# ---------------------------------------------------------------- commands


def read_if_exists(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


def run_build(root):
    rendered = render_all(root)
    written = []
    for rel, text in sorted(rendered.items()):
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if read_if_exists(path) == text:
            continue
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        written.append(rel)
    if written:
        print("build-agents: rewrote " + ", ".join(written))
    else:
        print("build-agents: %d generated file(s) already current" % len(rendered))
    for rel in orphan_skill_dirs(root):
        print("build-agents: WARNING %s/ has no source -- delete it by hand" % rel)
    return 0


def run_check(root):
    rendered = render_all(root)
    drifted = []
    for rel, text in sorted(rendered.items()):
        actual = read_if_exists(os.path.join(root, rel))
        if actual != text:
            drifted.append((rel, actual, text))
    orphans = orphan_skill_dirs(root)
    if not drifted and not orphans:
        print("AGENTS_FRESH — %d generated file(s) match %s + %s"
              % (len(rendered), SRC_DIR, SKILLS_SRC))
        return 0
    print("AGENTS_STALE — %d generated file(s) differ, %d orphaned skill dir(s):"
          % (len(drifted), len(orphans)))
    for rel, _, _ in drifted:
        print("  %s" % rel)
    for rel in orphans:
        print("  %s/  (no source under %s/ -- renamed or deleted; remove it)" % (rel, SKILLS_SRC))
    print("Edit the source under `.agents/agents/` or `.agents/skills/`, never the generated")
    print("file, then run `uv run python .agent-hooks/build-agents.py` and commit both.")
    for rel, actual, text in drifted:
        print("\n--- diff: %s (on disk -> would generate)" % rel)
        if actual is None:
            print("  (missing on disk)")
            continue
        diff = difflib.unified_diff(
            actual.splitlines(keepends=True),
            text.splitlines(keepends=True),
            fromfile="a/%s" % rel,
            tofile="b/%s" % rel,
        )
        sys.stdout.writelines(diff)
    return 1


def run_hook(root):
    """PostToolUse: rebuild only when the edited file was one of the sources.

    The hook matcher can only select on tool name, so the path test lives here. Advisory --
    a build failure is reported to the session, never used to block the edit.
    """
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    tool_input = data.get("tool_input") or {}
    paths = [tool_input.get("file_path") or ""]
    # Codex delivers an edit as apply_patch with the whole patch in `command` and no file_path.
    paths.extend(re.findall(
        r"^\*\*\* (?:Update|Add|Delete) File: (.+)$",
        tool_input.get("command") or "",
        re.M,
    ))
    paths = [path.strip().replace("\\", "/") for path in paths if path.strip()]

    def is_source(path):
        for src in (SRC_DIR, SKILLS_SRC):
            if ("/%s/" % src in path or path.startswith("%s/" % src)) and path.endswith(".md"):
                return True
        return False

    if not any(is_source(path) for path in paths):
        return 0
    try:
        run_build(root)
        return 0
    except (BuildError, OSError) as exc:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": (
                "[build-agents] %s changed but the per-harness files could NOT be regenerated: "
                "%s. `.claude/agents/**` and `.codex/agents/**` are now stale -- fix the source "
                "and run `uv run python .agent-hooks/build-agents.py`." % (SRC_DIR, exc)
            ),
        }}))
        return 0


def main(argv):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args = list(argv)
    if "--root" in args:
        i = args.index("--root")
        try:
            root = os.path.abspath(args[i + 1])
        except IndexError:
            print("AGENTS_UNKNOWN: --root needs a path")
            return 2
        del args[i:i + 2]

    mode = "build"
    for flag in ("--check", "--hook"):
        if flag in args:
            mode = flag[2:]
            args.remove(flag)
    if args:
        print("AGENTS_UNKNOWN: unrecognised argument(s) %s" % args)
        return 2

    try:
        if mode == "check":
            return run_check(root)
        if mode == "hook":
            return run_hook(root)
        return run_build(root)
    except BuildError as exc:
        print("AGENTS_UNKNOWN: %s" % exc)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
