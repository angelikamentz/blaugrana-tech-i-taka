"""PreToolUse guard: keeps Claude Code away from the match data and out of the rest of the disk.

The competition terms forbid making the match data (or any extract of it) available to a hosted
LLM. CLAUDE.md says so in words; this hook enforces it on every tool call, for the main session
and for subagents alike. It answers "deny" with a reason, which Claude sees and has to work around
the honest way: write a script and hand it to the user.

What it denies, in this order:

1. Data, wherever it lives -- the folder MTAH_DATA_DIR points at, data/ and
   data_DO_NOT_PUT_DATA_HERE/ in the repo, any match asset by name (BAR-ATH_* other than the public
   ground truth and label CSVs, *.mp4, *_atd.txt, ...), the private full-match ground truth, and
   the usual binary extract formats (.npy, .parquet, .pkl, ...).
2. Anything outside this repository, except Claude Code's own per-project folders (memory and
   scratchpad/temp) and, for reads only, installed skills and plugins.
3. Shell commands that would run the data loaders, walk the whole repo recursively (grep -r and
   find ignore .gitignore, so they would wander into the data folders), or reference the data
   directory through MTAH_DATA_DIR.
4. Edits to the guard itself and to the Claude Code settings that register it.

File-tool checks (Read, Edit, Write, Glob, Grep, ...) are exact. Shell checks are a best-effort
reading of the command string and can be fooled by a determined enough command; games.py's
asset_paths() refuses to run under Claude Code as a second line.

Standard library only. Any error inside the hook denies the call (fail closed).
"""

import datetime
import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.realpath(os.path.join(HERE, "..", ".."))
HOME = os.path.expanduser("~")
LOG = os.path.join(HERE, "guard.log")

# Claude Code names its per-project folders after the project path with every separator replaced.
SLUG = re.sub(r"[^A-Za-z0-9]", "-", REPO)

DATA_DIRS_IN_REPO = ("data", "data_DO_NOT_PUT_DATA_HERE")

# Match asset file names (see games.SUFFIXES) and the private ground truth (see .gitignore).
DATA_NAME_SUFFIXES = (
    ".mp4", ".gzip", ".pl3container",
    "_atd.txt", "_atd.xml", "_homography.csv", "_smart_tagging.xml",
    "_possession_gt.csv", "_possession_gt_meta.json", "_possession_gt_reference.csv",
    "possession_by_frame.pkl",
)
# Binary formats an extract of the data would be cached in. Nothing legitimate in this repo uses
# them; intermediates belong in the data directory.
EXTRACT_SUFFIXES = (".npy", ".npz", ".parquet", ".pkl", ".pickle", ".feather", ".h5", ".hdf5",
                    ".joblib", ".arrow")

SHELL_TOOLS = ("Bash", "PowerShell")
WRITE_TOOLS = ("Edit", "Write", "MultiEdit", "NotebookEdit")
PATH_KEYS = ("file_path", "path", "notebook_path", "file_paths", "paths", "root", "out_dir",
             "from", "glob", "pattern")

WHAT_TO_DO = ("If you need something from the match data, write a script that reads it through "
              "games.asset_paths(), prints only aggregates, and ask the user to run it in their "
              "own terminal and paste back the summary. Do not try another route to the same "
              "files. If this block looks wrong, stop and ask the user.")


class Deny(Exception):
    pass


# --------------------------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------------------------

def _norm(path):
    """Absolute, symlink-resolved, lower-case, forward-slash form for prefix comparisons."""
    return os.path.realpath(path).replace("\\", "/").rstrip("/").lower()


def _under(path_n, root_n):
    return path_n == root_n or path_n.startswith(root_n + "/")


REPO_N = _norm(REPO)
DATA_ROOTS_N = [_norm(os.path.join(REPO, d)) for d in DATA_DIRS_IN_REPO]
EXT_DATA = os.environ.get("MTAH_DATA_DIR", "").strip().strip('"')
if EXT_DATA:
    DATA_ROOTS_N.append(_norm(EXT_DATA))

# Claude Code's own folders for this project: memory + transcripts, and scratchpad + spilled
# tool output. Read and write.
ALLOWED_ROOTS_N = [
    _norm(os.path.join(HOME, ".claude", "projects", SLUG)),
    _norm(os.path.join(tempfile.gettempdir(), "claude", SLUG)),
]
# On macOS and Linux, Claude Code puts its scratchpad under /tmp/claude-<uid>/ (on macOS that is
# /private/tmp), not under tempfile.gettempdir() (/var/folders/... on macOS). The desktop app can
# also name the folder after a pre-started session instead of the project, so allow the whole
# per-user root. The data checks above still apply inside it.
if hasattr(os, "getuid"):
    ALLOWED_ROOTS_N.append(_norm("/tmp/claude-{}".format(os.getuid())))
# Read-only: installed skills and plugins, whose reference files skills sometimes point at.
READ_ONLY_ROOTS_N = [
    _norm(os.path.join(HOME, ".claude", "skills")),
    _norm(os.path.join(HOME, ".claude", "plugins")),
]
PROTECTED_N = [
    _norm(os.path.join(REPO, ".claude", "hooks")),
    _norm(os.path.join(REPO, ".claude", "settings.json")),
    _norm(os.path.join(REPO, ".claude", "settings.local.json")),
]
PUBLIC_GT_N = _norm(os.path.join(REPO, "evaluation", "gt"))
RESULTS_N = _norm(os.path.join(REPO, "evaluation", "results"))

SPECIAL_OK = {"/dev/null", "/dev/stdin", "/dev/stdout", "/dev/stderr", "nul", "$null"}

ENV_ALIASES = {
    "home": HOME, "userprofile": HOME,
    "temp": tempfile.gettempdir(), "tmp": tempfile.gettempdir(),
    "appdata": os.environ.get("APPDATA", ""), "localappdata": os.environ.get("LOCALAPPDATA", ""),
    "programdata": os.environ.get("PROGRAMDATA", ""), "systemroot": os.environ.get("SYSTEMROOT", ""),
    "claude_project_dir": REPO,
}
_ENV_RE = re.compile(r"\$env:(\w+)|\$\{(\w+)\}|\$(\w+)|%(\w+)%", re.I)


def _expand(token):
    """Expand ~ and the env vars that name directories. Unknown variables are left alone."""
    def sub(m):
        name = next(g for g in m.groups() if g)
        value = ENV_ALIASES.get(name.lower())
        return value if value else m.group(0)
    token = _ENV_RE.sub(sub, token)
    if token == "~" or token.startswith(("~/", "~\\")):
        token = HOME + token[1:]
    return token


def _posix_to_windows(token):
    """Git Bash spellings: /c/Users/... -> c:/Users/..., /tmp -> the temp dir."""
    m = re.match(r"^/([a-zA-Z])(/.*)?$", token)
    if m:
        return m.group(1) + ":" + (m.group(2) or "/")
    if token == "/tmp" or token.startswith("/tmp/"):
        return tempfile.gettempdir() + token[4:]
    return token


def check_path(raw, cwd, writing, shell=False):
    """Raise Deny if `raw` (as given to a tool or found in a command) is off limits."""
    if not raw or not isinstance(raw, str):
        return
    token = raw.strip().strip("'\"")
    if not token or token.lower() in SPECIAL_OK:
        return
    token = _expand(token)
    if os.name == "nt":
        token = _posix_to_windows(token)
    path = token if os.path.isabs(token) or re.match(r"^[a-zA-Z]:", token) else os.path.join(cwd, token)
    try:
        path_n = _norm(path)
    except (OSError, ValueError):
        return
    name = path_n.rsplit("/", 1)[-1]

    # 1. data -- denied everywhere, including inside the allowed roots
    for root_n in DATA_ROOTS_N:
        if _under(path_n, root_n):
            raise Deny("'{}' is inside a match-data folder.".format(raw))
    if any(name.endswith(s.lower()) for s in DATA_NAME_SUFFIXES):
        raise Deny("'{}' looks like a match asset or the private ground truth.".format(raw))
    if name.startswith("bar-ath_"):
        public_gt = _under(path_n, PUBLIC_GT_N) and "_possession_gt_public" in name
        label_csv = _under(path_n, RESULTS_N) and name.endswith(".csv")
        if not (public_gt or label_csv):
            raise Deny("'{}' is a match file (only the public ground truth and label CSVs in "
                       "evaluation/ are readable).".format(raw))
    if any(name.endswith(s) for s in EXTRACT_SUFFIXES):
        raise Deny("'{}' has a binary extract format; extracts of the data are off limits."
                   .format(raw))

    # 2. the guard itself
    if writing or shell == "write":
        if any(_under(path_n, p) for p in PROTECTED_N):
            raise Deny("'{}' is the data guard or the settings that load it. Only the user "
                       "edits these.".format(raw))

    # 3. outside the repository
    if _under(path_n, REPO_N) or any(_under(path_n, r) for r in ALLOWED_ROOTS_N):
        return
    if not writing and not shell and any(_under(path_n, r) for r in READ_ONLY_ROOTS_N):
        return
    raise Deny("'{}' is outside this repository.".format(raw))


# --------------------------------------------------------------------------------------------
# tool inputs
# --------------------------------------------------------------------------------------------

def _paths_in(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for v in value:
            yield from _paths_in(v)
    elif isinstance(value, dict):
        for k, v in value.items():
            if k in PATH_KEYS or k == "files":
                yield from _paths_in(v)


def check_tool_input(tool, tool_input, cwd):
    writing = tool in WRITE_TOOLS
    base = cwd
    if tool in ("Grep", "Glob") and isinstance(tool_input.get("path"), str):
        base = os.path.join(cwd, tool_input["path"])
    for key, value in tool_input.items():
        if key == "files" and isinstance(value, dict):  # Artifact: {published: source}
            for v in value.values():
                for p in _paths_in(v):
                    check_path(p, cwd, writing)
            continue
        if key not in PATH_KEYS:
            continue
        for p in _paths_in(value):
            if key in ("glob", "pattern"):
                if tool == "Grep" and key == "pattern":
                    continue  # a regex, not a path
                # Check the literal part before the first wildcard, and the name pattern itself.
                literal = re.split(r"[*?\[{]", p, 1)[0]
                if literal:
                    check_path(literal, base, writing)
                check_path(p.replace("*", "x"), base, writing)
            else:
                check_path(p, cwd, writing)


# --------------------------------------------------------------------------------------------
# shell commands
# --------------------------------------------------------------------------------------------

_TOKEN_RE = re.compile(r'"[^"]*"|\'[^\']*\'|[^\s;|&<>()`]+')
_RECURSIVE_RE = re.compile(
    r"\bgrep\b[^|;&\n]*\s(-[a-zA-Z]*[rR]|--recursive|--dereference-recursive)\b"
    r"|\bfind\b|\btree\b|\bdu\b|\bls\b[^|;&\n]*\s-[a-zA-Z]*R"
    r"|\b(Get-ChildItem|gci|dir|ls)\b[^|;&\n]*\s-r(ecurse)?\b|\bdir\b[^|;&\n]*\s/s\b"
    r"|\brg\b[^|;&\n]*\s(-u+|--no-ignore\S*)\b"
    r"|\b(cp|scp|rsync|zip|tar|7z)\b[^|;&\n]*\s-[a-zA-Z]*[rRa]|\bCompress-Archive\b|\bCopy-Item\b[^|;&\n]*-Recurse",
    re.I)
_RUNNER_RE = re.compile(r"(?<![\w.\-/\\])(python[\d.]*|py|ipython|jupyter|papermill|pytest|uv|poetry"
                        r"|nbconvert|streamlit)(\.exe)?\b", re.I)
_LOADER_RE = re.compile(r"baseline(\.py)?\b|\bload_(atd|homography|smart_tagging)\b|\.ipynb\b", re.I)
_WRITE_VERB_RE = re.compile(
    r">|\bsed\b[^|;&\n]*\s-i|\b(rm|mv|cp|tee|touch|truncate|chmod|ln|dd|install|patch)\b"
    r"|\b(Set-Content|Add-Content|Out-File|Remove-Item|Move-Item|Copy-Item|Rename-Item|New-Item"
    r"|Clear-Content)\b|\bgit\s+(checkout|restore|rm|mv|stash|reset|apply)\b|\bopen\(", re.I)


def check_command(command, cwd):
    if not isinstance(command, str):
        return
    low = command.lower()

    # Reading the data directory through the variable that points at it.
    if "mtah_data_dir" in low and not re.match(r"^\s*(grep|rg|git\s+grep)\b", command):
        raise Deny("the command references MTAH_DATA_DIR, the match-data location.")
    if "data_do_not_put_data_here" in low:
        raise Deny("the command references data_DO_NOT_PUT_DATA_HERE/.")
    if re.search(r"(?<![\w.\-/\\])(\./|\.\\)?data[/\\]|['\"]data['\"]", command):
        raise Deny("the command references the data/ folder.")
    if EXT_DATA:
        forms = {EXT_DATA.lower(), EXT_DATA.lower().replace("\\", "/")}
        m = re.match(r"^([a-z]):[\\/](.*)$", EXT_DATA.lower())
        if m:
            forms.add("/{}/{}".format(m.group(1), m.group(2).replace("\\", "/")))
        if any(f.rstrip("/\\") in low.replace("\\\\", "\\") for f in forms):
            raise Deny("the command references the match-data directory.")

    # Running the starter loaders, the baseline or the notebooks loads the data.
    if _RUNNER_RE.search(command) and _LOADER_RE.search(command):
        raise Deny("the command runs a starter loader, the baseline or a notebook, which loads "
                   "the match data.")

    tokens = []
    for t in _TOKEN_RE.findall(command):
        t = t.strip("'\"")
        tokens.append(t)
        if "=" in t:  # --file=path, VAR=path
            tokens.append(t.split("=", 1)[1])

    writes = bool(_WRITE_VERB_RE.search(command))
    for t in tokens:
        if not t or t.startswith("-") and "=" not in t:
            continue
        # A quoted string with spaces that starts with "/" is far more often an awk or sed
        # program than a path; only drive-letter paths are taken from those.
        if " " in t and not re.match(r"^[a-zA-Z]:[\\/]", t):
            continue
        check_path(t, cwd, writing=False, shell="write" if writes else True)

    # Recursive walks that would cover the repo root also cover the data folders.
    if _RECURSIVE_RE.search(command):
        dirs = []
        for t in tokens:
            if not t or t.startswith("-"):
                continue
            p = os.path.join(cwd, _expand(t))
            if os.path.isdir(p):
                dirs.append(_norm(p))
        if not dirs or any(REPO_N == d or REPO_N.startswith(d + "/") for d in dirs):
            raise Deny("a recursive shell command over the repo root would walk into the data "
                       "folders (grep -r and find ignore .gitignore). Use the Grep/Glob tools, "
                       "`git ls-files` / `git grep`, or name a subdirectory such as evaluation/.")


# --------------------------------------------------------------------------------------------

def decide(event):
    tool = event.get("tool_name", "")
    tool_input = event.get("tool_input") or {}
    cwd = event.get("cwd") or REPO
    if tool in SHELL_TOOLS:
        check_command(tool_input.get("command"), cwd)
    else:
        check_tool_input(tool, tool_input, cwd)


def _log(tool, reason):
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write("{}\t{}\t{}\n".format(datetime.datetime.now().isoformat(timespec="seconds"),
                                          tool, reason))
    except OSError:
        pass


def _deny(reason):
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": "Data guard: {} {}".format(reason, WHAT_TO_DO),
    }}))


def main():
    tool = "?"
    try:
        event = json.loads(sys.stdin.read())
        tool = event.get("tool_name", "?")
        decide(event)
    except Deny as e:
        _log(tool, str(e))
        _deny(str(e))
    except Exception as e:  # fail closed
        _log(tool, "guard error: {!r}".format(e))
        _deny("the guard hit an internal error ({}), so the call is blocked to be safe."
              .format(type(e).__name__))
    return 0


if __name__ == "__main__":
    sys.exit(main())
