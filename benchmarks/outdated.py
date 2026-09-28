"""`run.py outdated`: the version of each library the benchmarks pin, the one in the published
results (docs/benchmarks/results.md) and the latest release in its registry, to know which library
to rerun with `run.py --update`: one with a newer release ("newer"), or whose pin moved since the
published results ("not rerun").

Each adapter of run.ADAPTERS with a "release" is checked, with the standard library only:
- ("pypi", package): pinned in requirements.txt, released on PyPI;
- ("crates", crate): pinned in the adapter's Cargo.lock, released on crates.io;
- ("maven", group, artifact): pinned in the adapter's build.sh (<NAME>_VERSION=), released on
  Maven Central;
- ("julia", package): pinned in the adapter's Manifest.toml, released in Julia's General registry;
- ("github", repository, header): pinned to a commit in the adapter's build.sh (COMMIT=, VERSION=);
  newer when the repository has a newer tag, or when its default branch changed the header.

`--issue` keeps one open GitHub issue, ISSUE_TITLE, listing the newer releases and the pins not
rerun, with `gh`: it creates it, updates its body when the list changes, and closes it when no
library is newer and the published results measure every pin. It doesn't touch the issue when a
check failed. `--dry-run` prints what it would do instead.
"""

import concurrent.futures
import gzip
import json
import os
import re
import subprocess
import tomllib
import urllib.error
import urllib.request
import xml.etree.ElementTree as ElementTree

import run

ISSUE_TITLE = "New releases of benchmarked libraries"
PUBLISHED = run.ROOT.parent / "docs" / "benchmarks" / "results.md"
USER_AGENT = "genoxide-benchmarks (https://github.com/tachsin/genoxide)"
TIMEOUT = 30


def fetch(url):
    """The body of a GET request."""
    # compressed: PyPI's JSON of a library with many wheels, e.g. SciPy's, is megabytes
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        # the unauthenticated limit is 60 requests an hour
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=TIMEOUT) as response:
        body = response.read()
        return gzip.decompress(body) if response.headers.get("Content-Encoding") == "gzip" else body


def stable(version):
    """Whether a version is a release: numbers only, e.g. 1.4.0, not 2.0.0-rc1."""
    return re.fullmatch(r"v?\d+(\.\d+)*", version) is not None


def version_key(version):
    """A version's numbers, to compare it, without a leading v, a +build suffix or trailing zeros:
    7.5 is 7.5.0."""
    numbers = [int(number) for number in re.findall(r"\d+", version.split("+")[0])]
    while numbers and numbers[-1] == 0:
        numbers.pop()
    return tuple(numbers)


def newest(versions):
    return max((v for v in versions if stable(v)), key=version_key, default=None)


def adapter_file(name, file):
    return run.ROOT / "adapters" / name / file


def shell_variable(name, pattern):
    """The value of the first variable of the adapter's build.sh whose name matches `pattern`."""
    text = adapter_file(name, "build.sh").read_text(encoding="utf-8")
    match = re.search(rf"^{pattern}=\"?([^\"\s]+)\"?$", text, re.MULTILINE)
    if not match:
        raise ValueError(f"no {pattern}= in adapters/{name}/build.sh")
    return match.group(1)


def pinned_version(name, release):
    """The version the benchmarks pin, where the adapter's build reads it."""
    kind = release[0]
    if kind == "pypi":
        text = (run.ROOT / "requirements.txt").read_text(encoding="utf-8")
        match = re.search(rf"^{re.escape(release[1])}==(\S+)", text, re.MULTILINE | re.IGNORECASE)
        if not match:
            raise ValueError(f"{release[1]} isn't pinned in requirements.txt")
        return match.group(1)
    if kind == "crates":
        lock = tomllib.loads(adapter_file(name, "Cargo.lock").read_text(encoding="utf-8"))
        return next(package["version"] for package in lock["package"] if package["name"] == release[1])
    if kind == "maven":
        return shell_variable(name, r"[A-Z]+_VERSION")
    if kind == "julia":
        manifest = tomllib.loads(adapter_file(name, "Manifest.toml").read_text(encoding="utf-8"))
        return manifest["deps"][release[1]][0]["version"]
    if kind == "github":
        return shell_variable(name, "VERSION")
    raise ValueError(f"unknown registry {kind}")


def pinned_in(name, release, pinned):
    """The files to change to update a library's pin."""
    kind = release[0]
    if kind == "pypi":
        return ["benchmarks/requirements.txt"]
    folder = f"benchmarks/adapters/{name}"
    if kind == "crates":
        return [f"{folder}/Cargo.toml", f"{folder}/Cargo.lock"]
    if kind == "julia":
        return [f"{folder}/Project.toml", f"{folder}/Manifest.toml"]
    # a shell script may repeat the version, e.g. Jenetics' run.sh
    return [f"{folder}/{file}" for file in ("build.sh", "run.sh")
            if adapter_file(name, file).is_file() and pinned in adapter_file(name, file).read_text(encoding="utf-8")]


def latest_version(name, release, pinned):
    """The latest release in the library's registry."""
    kind = release[0]
    if kind == "pypi":
        return json.loads(fetch(f"https://pypi.org/pypi/{release[1]}/json"))["info"]["version"]
    if kind == "crates":
        crate = json.loads(fetch(f"https://crates.io/api/v1/crates/{release[1]}"))["crate"]
        return crate["max_stable_version"] or crate["max_version"]
    if kind == "maven":
        group, artifact = release[1], release[2]
        metadata = ElementTree.fromstring(fetch(
            f"https://repo1.maven.org/maven2/{group.replace('.', '/')}/{artifact}/maven-metadata.xml"))
        return newest(element.text for element in metadata.iter("version"))
    if kind == "julia":
        package = release[1]
        versions = tomllib.loads(fetch(
            f"https://raw.githubusercontent.com/JuliaRegistries/General/master/{package[0].upper()}/{package}/Versions.toml"
        ).decode("utf-8"))
        return newest(version for version, entry in versions.items() if not entry.get("yanked"))
    if kind == "github":
        repository, header = release[1], release[2]
        api = f"https://api.github.com/repos/{repository}"
        tag = newest(tag["name"] for tag in json.loads(fetch(f"{api}/tags?per_page=100"))).lstrip("v")
        base = pinned.split("+")[0]
        if version_key(tag) > version_key(base):
            return tag
        # the pinned commit is past the last tag: newer when the default branch changed the header since
        branch = json.loads(fetch(api))["default_branch"]
        commit = shell_variable(name, "COMMIT")
        compare = json.loads(fetch(f"{api}/compare/{commit}...{branch}"))
        if any(file["filename"] == header for file in compare.get("files", [])):
            return f"{base}+{compare['commits'][-1]['sha'][:7]}"
        return pinned
    raise ValueError(f"unknown registry {kind}")


def published_versions():
    """The versions in the published results, docs/benchmarks/results.md: its "- <library> <version>"
    lines, before the invalid runs."""
    versions = {}
    if not PUBLISHED.is_file():
        return versions
    for line in PUBLISHED.read_text(encoding="utf-8").splitlines():
        if line.startswith("Invalid runs"):
            break
        match = re.fullmatch(r"- (\w+) (\S+)", line.strip())
        if match:
            versions[match.group(1)] = match.group(2)
    return versions


def is_newer(pinned, latest):
    """Whether the latest release is past the pinned version; a later commit after the same tag
    (1.0.5+<commit>) is too."""
    if version_key(latest) != version_key(pinned):
        return version_key(latest) > version_key(pinned)
    commit = latest.partition("+")[2]
    return bool(commit) and commit != pinned.partition("+")[2]


def same_version(a, b):
    """Whether two versions are the same release, and the same commit after it if either has one:
    7.5 is 7.5.0, 1.0.5+f9b15e7 isn't 1.0.5+0c2d1e4."""
    return version_key(a) == version_key(b) and a.partition("+")[2] == b.partition("+")[2]


def check_library(name):
    """A row of the table: pinned, published and latest versions, whether the latest is newer, and
    the error that kept it from being checked, if any."""
    release = run.ADAPTERS[name]["release"]
    row = {"library": name, "pinned": None, "latest": None, "newer": False, "not_rerun": False, "error": None,
           "pinned_in": []}
    try:
        row["pinned"] = pinned_version(name, release)
        row["pinned_in"] = pinned_in(name, release, row["pinned"])
        row["latest"] = latest_version(name, release, row["pinned"])
        if row["latest"] is None:
            raise ValueError("no release found")
        row["newer"] = is_newer(row["pinned"], row["latest"])
    except urllib.error.HTTPError as error:
        row["error"] = f"HTTP {error.code} {error.reason}"
    except urllib.error.URLError as error:
        row["error"] = f"{error.reason}"
    except Exception as error:  # noqa: BLE001 - a library that can't be checked doesn't stop the others
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def check(libraries):
    """The rows of the libraries with a "release", in the order of ADAPTERS, checked in parallel. A
    library is "not_rerun" when its pin differs from the version in the published results."""
    names = [name for name in libraries if run.ADAPTERS[name].get("release")]
    published = published_versions()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(check_library, names))
    for row in rows:
        row["published"] = published.get(row["library"])
        row["not_rerun"] = bool(row["pinned"] and row["published"]
                                and not same_version(row["pinned"], row["published"]))
    return rows


def status(row):
    """What the table says about a row: its error, or whether it's newer and not rerun."""
    if row["error"]:
        return f"error: {row['error']}"
    return ", ".join(word for word, flag in (("newer", row["newer"]), ("not rerun", row["not_rerun"])) if flag)


def table(rows):
    """The rows as aligned text columns."""
    lines = [("library", "pinned", "published", "latest", "")]
    for row in rows:
        lines.append((row["library"], row["pinned"] or "-", row["published"] or "-", row["latest"] or "-", status(row)))
    widths = [max(len(line[column]) for line in lines) for column in range(4)]
    return "\n".join("  ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip() + (
        "  " + line[4] if line[4] else "") for line in lines)


def rerun_command(names):
    """The commands that rerun `names` into the published results."""
    return [f"python run.py check --libraries {names}", f"python run.py --update --libraries {names}",
            "python run.py publish"]


def issue_body(rows):
    """The body of the issue: the libraries with a newer release, those whose pin the published
    results don't measure, and how to rerun them."""
    newer = [row for row in rows if row["newer"]]
    not_rerun = [row for row in rows if row["not_rerun"]]
    lines = []
    if newer:
        lines += ["Libraries with a release newer than the version the benchmarks pin:", "",
                  "| Library | Pinned | Latest | Pinned in |", "|---|---|---|---|"]
        for row in newer:
            files = ", ".join(f"`{file}`" for file in row["pinned_in"])
            lines.append(f"| {run.LIBRARY_NAMES.get(row['library'], row['library'])} | {row['pinned']} "
                         f"| {row['latest']} | {files} |")
        lines.append("")
    if not_rerun:
        lines += ["Libraries whose pin moved since the published results, not rerun yet:", "",
                  "| Library | Pinned | Published |", "|---|---|---|"]
        for row in not_rerun:
            lines.append(f"| {run.LIBRARY_NAMES.get(row['library'], row['library'])} | {row['pinned']} "
                         f"| {row['published']} |")
        lines.append("")
    names = " ".join(row["library"] for row in rows if row["newer"] or row["not_rerun"])
    lines += ["To rerun them" + (", after updating the pins of the newer ones" if newer else "") + ", in `benchmarks/` "
              "([README](https://github.com/tachsin/genoxide/blob/main/benchmarks/README.md#running)):", "", "```sh",
              *rerun_command(names), "```", "",
              "The library-releases workflow updates this issue every week, and closes it when no library is "
              "newer and the published results measure every pin."]
    return "\n".join(lines) + "\n"


def issue_action(rows, issue):
    """What to do with the issue, from the rows and the open issue ({"number", "body"} or None):
    ("create", body), ("edit", body), ("close", None) or ("none", None)."""
    if any(row["newer"] or row["not_rerun"] for row in rows):
        body = issue_body(rows)
        if issue is None:
            return "create", body
        if issue["body"].replace("\r\n", "\n").strip() != body.strip():
            return "edit", body
        return "none", None
    return ("close", None) if issue else ("none", None)


def gh(*arguments, stdin=None):
    return subprocess.run(["gh", *arguments], input=stdin, capture_output=True, text=True, check=True).stdout


def open_issue():
    """The open issue titled ISSUE_TITLE, the oldest if there are several, or None."""
    issues = json.loads(gh("issue", "list", "--state", "open", "--search", f'in:title "{ISSUE_TITLE}"',
                           "--json", "number,title,body", "--limit", "20"))
    issues = sorted((issue for issue in issues if issue["title"] == ISSUE_TITLE), key=lambda issue: issue["number"])
    return issues[0] if issues else None


def update_issue(rows, dry_run):
    """Creates, updates or closes the issue; with `dry_run`, prints what it would do. Leaves it as it
    is, and fails, when a library couldn't be checked."""
    failed = [row["library"] for row in rows if row["error"]]
    if failed:
        raise SystemExit(f"issue: not updated, {', '.join(failed)} couldn't be checked")
    issue = open_issue()
    action, body = issue_action(rows, issue)
    where = f"#{issue['number']}" if issue else f'"{ISSUE_TITLE}"'
    if dry_run:
        print(f"issue (dry run): {action} {where}")
        if body:
            print(body)
        return
    if action == "create":
        print(gh("issue", "create", "--title", ISSUE_TITLE, "--body-file", "-", stdin=body).strip())
    elif action == "edit":
        gh("issue", "edit", str(issue["number"]), "--body-file", "-", stdin=body)
        print(f"issue: updated #{issue['number']}")
    elif action == "close":
        gh("issue", "close", str(issue["number"]), "--comment",
           "Every benchmarked library is at its latest release, and the published results measure it.")
        print(f"issue: closed #{issue['number']}")
    else:
        print(f"issue: {where} is up to date" if issue else "issue: no library is newer or not rerun")


def outdated(libraries, issue=False, dry_run=False):
    """Prints the table and, with `issue`, updates the issue."""
    rows = check(libraries)
    print(table(rows))
    newer = [row["library"] for row in rows if row["newer"]]
    not_rerun = [row["library"] for row in rows if row["not_rerun"]]
    if newer:
        print(f"\nnewer releases: {', '.join(newer)}. Update their pins.")
    if not_rerun:
        print(f"\npinned but not rerun: {', '.join(not_rerun)}. The published results measure other versions.")
    if newer or not_rerun:
        names = " ".join(row["library"] for row in rows if row["newer"] or row["not_rerun"])
        print("\nTo rerun them: " + "; ".join(f"`{command}`" for command in rerun_command(names)) + ".")
    if issue:
        update_issue(rows, dry_run)
