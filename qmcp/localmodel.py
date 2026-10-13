"""Standing a local model up, and being able to do it again.

    uv run qmcp localmodel check
    uv run qmcp localmodel plan --models-dir D:\\ollama

A judgement worker is a deployment decision, so this is a project's own
operational tooling, not governance: `qmcp.sweep` names no tool, and this
module is where the vendor may appear. Nothing in `sweep.py` imports it.

It checks before it proposes, and proposes before it runs. `check` measures;
`plan` turns a measurement into the exact commands; running them is a
person's. The commands are printed so they can be read first and re-run, and a
rebuilt machine gets the same model in the same place.

A running service does not see a variable set after it started: `ollama pull`
is a client, and the service read its environment, including `OLLAMA_MODELS`,
once at start. The plan therefore restarts the service before it pulls.

The model and its digest are pinned, because `qwen2.5-coder:7b` is a tag that
moves, and a rebuild must not silently get a different model under the same
name.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# What is being installed, pinned. `q4_K_M` is the quantisation that fits the
# target card; a bigger one does not, and a smaller one is a different model
# for judgement purposes.
MODEL = "qwen2.5-coder:7b"
MODEL_BYTES = 4_700_000_000
"""About 4.7 GB. Measured from the registry rather than guessed at, and used
only to refuse a disk that plainly cannot hold it."""

INSTALLER_BYTES = 1_500_000_000
"""The installer, plus what it extracts while running. Windows extracts to the
user's temp directory whatever the install target is, which is the part that
catches people who point the install at a roomy drive and still run out."""

# Enough room for the model, the installer, and somewhere to put a second model
# later without doing this again.
WANT_FREE = MODEL_BYTES + INSTALLER_BYTES + 5_000_000_000

# Below this, a Windows installation is in trouble for reasons that have nothing
# to do with us -- updates, page file, temp files. Reported because a person
# should know, not because it is this module's business to fix.
SYSTEM_DRIVE_FLOOR = 20_000_000_000

# Where the service listens. A judgement worker talks to this, not to a command
# line -- which is also why the plan's last step uses it.
ENDPOINT = "http://127.0.0.1:11434"


@dataclass(frozen=True)
class Volume:
    """One drive, and whether it could hold this."""

    name: str
    free: int
    total: int

    @property
    def is_system(self) -> bool:
        return self.name.upper().startswith("C")

    @property
    def roomy_enough(self) -> bool:
        return self.free >= WANT_FREE

    def human(self) -> str:
        return f"{self.name}: {self.free / 1e9:.1f} GB free of {self.total / 1e9:.1f}"


@dataclass
class Check:
    """What is here, before anything is installed."""

    volumes: list[Volume] = field(default_factory=list)
    gpu: str | None = None
    vram_mb: int | None = None
    installer: str | None = None
    """The tool that could install it, or None."""

    ollama: str | None = None
    """The path to an existing install, or None."""

    models_dir: str | None = None
    """Where models are kept now, if anything says."""

    @property
    def best_volume(self) -> Volume | None:
        """The roomiest drive that could hold this, or None if none can."""
        able = [v for v in self.volumes if v.roomy_enough]
        return max(able, key=lambda v: v.free) if able else None

    @property
    def system_drive_is_tight(self) -> bool:
        for volume in self.volumes:
            if volume.is_system:
                return volume.free < SYSTEM_DRIVE_FLOOR
        return False

    @property
    def blockers(self) -> list[str]:
        """What would stop this, each said plainly.

        A list rather than a boolean: "cannot install" is not useful, and the
        two reasons here have completely different remedies.
        """
        found = []
        if self.best_volume is None:
            found.append(
                f"no drive has {WANT_FREE / 1e9:.0f} GB free; the model alone "
                f"is {MODEL_BYTES / 1e9:.1f} GB")
        if self.system_drive_is_tight:
            found.append(
                f"the system drive is under {SYSTEM_DRIVE_FLOOR / 1e9:.0f} GB "
                f"free, and Windows extracts an installer into it whatever the "
                f"install target is")
        if self.installer is None and self.ollama is None:
            found.append("nothing here can install it and it is not installed")
        return found

    def summary(self) -> str:
        lines = [f"gpu: {self.gpu or 'unknown'}"
                 + (f", {self.vram_mb} MB" if self.vram_mb else "")]
        for volume in self.volumes:
            mark = "  <- roomiest" if volume is self.best_volume else ""
            lines.append(f"  {volume.human()}{mark}")
        lines.append(f"ollama: {self.ollama or 'not installed'}")
        lines.append(f"installer: {self.installer or 'none found'}")
        if self.blockers:
            lines.append("blocked:")
            lines.extend(f"  - {b}" for b in self.blockers)
        return "\n".join(lines)


@dataclass
class Plan:
    """The exact commands, in order, with what each one costs.

    Printed rather than run. A step that downloads five gigabytes should be
    something a person read before it started, and something they can run again
    later without this module being involved.
    """

    steps: list[tuple[str, str]] = field(default_factory=list)
    """(command, why) pairs."""

    blocked_by: list[str] = field(default_factory=list)

    @property
    def is_runnable(self) -> bool:
        return bool(self.steps) and not self.blocked_by

    def render(self) -> str:
        if self.blocked_by:
            return ("Not runnable yet:\n"
                    + "\n".join(f"  - {b}" for b in self.blocked_by))
        lines = []
        for command, why in self.steps:
            lines.append(f"# {why}")
            lines.append(command)
            lines.append("")
        return "\n".join(lines).rstrip()


def look(runner: Any = None) -> Check:
    """Measure this machine. Reads only; installs nothing.

    `runner` is injected so this can be tested without a machine underneath it.
    """
    runner = runner or _powershell
    found = Check()

    found.volumes = _volumes(runner)
    found.gpu, found.vram_mb = _gpu(runner)
    found.ollama = shutil.which("ollama")
    found.installer = "winget" if shutil.which("winget") else None

    import os

    found.models_dir = os.environ.get("OLLAMA_MODELS") or None
    return found


def plan(check: Check, models_dir: str | None = None) -> Plan:
    """Turn a measurement into commands.

    `models_dir` decides where several gigabytes land. It defaults to the
    roomiest drive, so the system drive is never filled silently.
    """
    made = Plan(blocked_by=list(check.blockers))
    if made.blocked_by:
        return made

    target = models_dir
    if target is None:
        volume = check.best_volume
        target = str(Path(f"{volume.name}:/") / "ollama" / "models")

    if check.ollama is None:
        made.steps.append((
            "winget install --id Ollama.Ollama --exact --accept-package-agreements "
            "--accept-source-agreements",
            "install the runtime; it is not here yet"))

    made.steps.append((
        f'setx OLLAMA_MODELS "{target}"',
        f"keep the weights on a drive with room; without this they go to the "
        f"user profile on the system drive"))
    made.steps.append((
        f'$env:OLLAMA_MODELS = "{target}"',
        "and in this shell, because setx only affects new ones"))
    made.steps.append((
        'Get-Process -Name "ollama*" -ErrorAction SilentlyContinue '
        '| Stop-Process -Force',
        "**restart the service, or the variable does nothing.** `ollama pull` "
        "is a client: the background service decides where weights go, and it "
        "read the environment when it started. The installer starts it, so on "
        "a first run it is already running without the variable -- and the "
        "pull lands in the user profile on the system drive, which is exactly "
        "what the step above exists to prevent. Measured: 4.4 GB went to C: "
        "with OLLAMA_MODELS correctly set to E:"))
    made.steps.append((
        r'Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama app.exe"',
        "start it again, now that it can see where the weights belong"))
    made.steps.append((
        f"ollama pull {MODEL}",
        f"about {MODEL_BYTES / 1e9:.1f} GB, once"))
    # Built separately, because a JSON body inside a shell-quoted argument
    # inside an f-string is three levels of quoting and the middle one loses.
    probe = ('{"model": "' + MODEL + '", '
             '"prompt": "Reply with exactly one word: ready", "stream": false}')
    made.steps.append((
        f"curl -s {ENDPOINT}/api/generate -d '{probe}'",
        "prove it answers before anything depends on it. THE API RATHER THAN "
        "`ollama run`: that command reads stdin and hangs when there is no "
        "terminal, which is every script and every check. The service is what "
        "a worker talks to anyway, so this proves the thing that matters"))
    return made


def _powershell(script: str) -> str:
    import subprocess

    done = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    return done.stdout.strip()


def _volumes(runner: Any) -> list[Volume]:
    out = runner(
        "Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Used -ne $null } "
        "| ForEach-Object { \"$($_.Name),$($_.Free),$($_.Used)\" }")
    found = []
    for line in out.splitlines():
        parts = line.strip().split(",")
        if len(parts) != 3:
            continue
        try:
            free, used = int(parts[1]), int(parts[2])
        except ValueError:
            continue
        found.append(Volume(name=parts[0], free=free, total=free + used))
    return found


def _gpu(runner: Any) -> tuple[str | None, int | None]:
    out = runner(
        "try { & nvidia-smi --query-gpu=name,memory.total "
        "--format=csv,noheader,nounits } catch { '' }")
    line = out.splitlines()[0].strip() if out.strip() else ""
    if not line or "," not in line:
        return (line or None), None
    name, _, memory = line.partition(",")
    try:
        return name.strip(), int(memory.strip())
    except ValueError:
        return name.strip(), None
