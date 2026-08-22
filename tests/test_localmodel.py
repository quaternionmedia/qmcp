"""Standing a local model up, and refusing to when it would fail.

THE TEST WORTH READING IS THE SYSTEM-DRIVE ONE. Pointing the install at a roomy
drive and running out anyway is the failure this module exists to catch: Windows
extracts an installer into the user's temp directory whatever the target is, and
on the machine this was written for the system drive had 460 MB free while two
other drives had hundreds of gigabytes. A check that only looked at the target
would have said yes.
"""

from __future__ import annotations

import pytest

from qmcp.localmodel import (
    MODEL,
    MODEL_BYTES,
    SYSTEM_DRIVE_FLOOR,
    WANT_FREE,
    Check,
    Volume,
    look,
    plan,
)

GB = 1_000_000_000


def drives(*pairs):
    """(name, free_gb) -> volumes with a plausible total."""
    return [Volume(name=name, free=int(free * GB), total=int((free + 500) * GB))
            for name, free in pairs]


def ready(**kwargs):
    found = Check(volumes=drives(("C", 100), ("E", 300)),
                  gpu="NVIDIA GeForce RTX 3070", vram_mb=8192,
                  installer="winget", ollama=None)
    for key, value in kwargs.items():
        setattr(found, key, value)
    return found


# --- refusing before it fails --------------------------------------------------


def test_a_full_system_drive_blocks_even_when_another_drive_is_empty():
    """THE ONE THAT MATTERS.

    Measured on the real machine: C: had 460 MB and E: had 302 GB. The
    installer extracts into the user's temp directory on C: whatever the target
    is, so "there is room on E:" is not an answer.

    Mutation: check only `best_volume` and this fails.
    """
    tight = ready(volumes=drives(("C", 0.5), ("E", 300)))

    assert tight.system_drive_is_tight is True
    assert tight.blockers, "a system drive at 0.5 GB was not a blocker"
    assert any("system drive" in b for b in tight.blockers)
    assert plan(tight).is_runnable is False


def test_no_drive_with_room_is_a_different_blocker_from_a_tight_system_drive():
    """Two reasons with completely different remedies. "Cannot install" would
    tell somebody neither of them.

    Mutation: collapse them into one message and this fails.
    """
    # NO SYSTEM DRIVE HERE, AND THAT IS FORCED RATHER THAN CHOSEN. The floor
    # for the system drive is higher than what the install wants, so any C:
    # roomy enough to clear the floor is also roomy enough to hold the model --
    # the two blockers cannot co-occur while a C: exists. Two earlier attempts
    # at this fixture were asserting the opposite of what they set up.
    nowhere = ready(volumes=drives(("D", 3), ("E", 2)))
    assert nowhere.best_volume is None
    assert any("no drive has" in b for b in nowhere.blockers)
    assert not any("system drive" in b for b in nowhere.blockers)


def test_nothing_to_install_with_is_a_blocker_only_when_it_is_absent():
    """An installed runtime needs no installer."""
    assert any("nothing here can install"
               in b for b in ready(installer=None).blockers)
    assert not ready(installer=None, ollama="C:/ollama.exe").blockers


def test_a_blocked_plan_says_why_rather_than_printing_commands():
    """A plan that printed the commands anyway would be one somebody pasted."""
    rendered = plan(ready(volumes=drives(("C", 0.5), ("E", 300)))).render()
    assert "Not runnable" in rendered
    assert "winget install" not in rendered


# --- where the weights go ------------------------------------------------------


def test_the_weights_are_pointed_somewhere_with_room():
    """THE OTHER ONE THAT MATTERS.

    Ollama's default puts them in the user profile, which is on the drive that
    was already full. Several gigabytes landing there is how a machine that was
    just cleared fills up again.

    Mutation: drop the `OLLAMA_MODELS` step and this fails.
    """
    steps = plan(ready()).steps
    joined = " ".join(command for command, _ in steps)
    assert "OLLAMA_MODELS" in joined
    assert "E:" in joined, "the roomiest drive was not chosen"


def test_the_roomiest_drive_is_chosen_and_not_the_first_one():
    found = ready(volumes=drives(("C", 100), ("D", 150), ("E", 300)))
    assert found.best_volume.name == "E"
    assert "E:" in plan(found).render()


def test_an_explicit_directory_wins_over_the_roomiest_drive():
    """The default is a guess about intent. Somebody who says where means it.

    Asserted against the commands rather than the whole rendering: the
    explanations cite the drive that this went wrong on once, and a test that
    reads the prose fails whenever somebody writes a clearer comment.
    """
    steps = plan(ready(), models_dir=r"D:\somewhere\else").steps
    commands = " ".join(command for command, _ in steps)
    assert r"D:\somewhere\else" in commands
    assert "E:" not in commands


def test_the_variable_is_set_for_this_shell_as_well_as_the_next():
    """`setx` affects new shells only. A plan that stopped there would pull
    several gigabytes into the old location in the very shell that ran it.

    Mutation: drop the `$env:` step and this fails.
    """
    commands = [command for command, _ in plan(ready()).steps]
    assert any(c.startswith("setx OLLAMA_MODELS") for c in commands)
    assert any(c.startswith("$env:OLLAMA_MODELS") for c in commands)


# --- what gets installed -------------------------------------------------------


def test_the_model_is_pinned_rather_than_left_as_a_moving_tag():
    """A rebuild that silently got a different model would be a different
    judgement worker wearing the same name."""
    assert ":" in MODEL, f"{MODEL!r} names no version at all"
    assert MODEL in plan(ready()).render()


def test_the_install_step_is_skipped_when_it_is_already_installed():
    """Re-running this is the point. A plan that reinstalled every time would
    be one nobody runs twice.

    Mutation: always emit the install step and this fails.
    """
    rendered = plan(ready(ollama="C:/ollama.exe")).render()
    assert "winget install" not in rendered
    assert f"ollama pull {MODEL}" in rendered


def test_the_plan_ends_by_proving_it_answers_over_the_api():
    """Installed is not the same as working, and the difference is one command.

    THE API, NOT `ollama run`. That command reads stdin and hangs with no
    terminal -- it did, for ten minutes, in the run that put this here. The
    service is what a worker talks to, so proving the service is proving the
    thing that matters.

    Mutation: go back to `ollama run` and this fails.
    """
    from qmcp.localmodel import ENDPOINT

    last, why = plan(ready()).steps[-1]
    assert ENDPOINT in last
    assert "ollama run" not in last
    assert "prove" in why


def test_every_step_says_why_it_is_there():
    """A command without a reason is one somebody runs on trust."""
    for command, why in plan(ready()).steps:
        assert command.strip() and why.strip()


# --- the numbers ---------------------------------------------------------------


def test_the_space_wanted_covers_more_than_the_model():
    """The installer and its extraction are real, and a check sized to the
    model alone passes right up until the moment it matters."""
    assert WANT_FREE > MODEL_BYTES * 2


def test_the_floor_is_stated_rather_than_buried():
    assert SYSTEM_DRIVE_FLOOR > 0
    tight = ready(volumes=drives(("C", (SYSTEM_DRIVE_FLOOR / GB) - 1), ("E", 300)))
    assert tight.system_drive_is_tight is True
    fine = ready(volumes=drives(("C", (SYSTEM_DRIVE_FLOOR / GB) + 1), ("E", 300)))
    assert fine.system_drive_is_tight is False


def test_a_machine_with_no_system_drive_is_not_reported_as_tight():
    """`is_tight` must not be true merely because nothing matched."""
    assert ready(volumes=drives(("D", 300))).system_drive_is_tight is False


# --- measuring ------------------------------------------------------------------


def test_look_reads_the_machine_through_an_injected_runner():
    """So this is testable without a machine underneath it, and so the parsing
    is exercised rather than the environment."""
    def answers(script):
        if "PSDrive" in script:
            return "C,500000000,900000000\nE,300000000000,100000000000"
        if "nvidia-smi" in script:
            return "NVIDIA GeForce RTX 3070, 8192"
        return ""

    found = look(runner=answers)
    assert [v.name for v in found.volumes] == ["C", "E"]
    assert found.gpu == "NVIDIA GeForce RTX 3070"
    assert found.vram_mb == 8192


def test_a_machine_with_no_gpu_reports_none_rather_than_guessing():
    found = look(runner=lambda script: "" if "nvidia" in script else "C,1,1")
    assert found.gpu is None
    assert found.vram_mb is None


def test_an_unparseable_drive_line_is_skipped_rather_than_crashing():
    """A `Get-PSDrive` that printed something unexpected should cost one row,
    not the whole check."""
    def answers(script):
        if "PSDrive" in script:
            return "C,500000000,900000000\nnonsense\nE,not-a-number,1"
        return ""

    assert [v.name for v in look(runner=answers).volumes] == ["C"]


def test_the_system_floor_sits_above_what_the_install_wants():
    """Stated, because it is why the two blockers cannot both apply to a
    machine with a system drive: a C: with room to clear the floor necessarily
    has room for the model. A floor below `WANT_FREE` would make the tight
    branch unreachable and the check quietly weaker than it reads.
    """
    assert SYSTEM_DRIVE_FLOOR > WANT_FREE


# --- the step that is easy to leave out ---------------------------------------


def test_the_service_is_restarted_before_anything_is_pulled():
    """THE ONE THAT COST 4.4 GB.

    `ollama pull` is a client. The background service decides where weights go
    and it read its environment when it started -- and the installer starts it,
    so on a first run it is already running without the variable. Setting
    `OLLAMA_MODELS` and pulling straight after puts the model exactly where the
    variable said not to, silently, and the plan that did that had a comment
    warning about it two lines above.

    Mutation: drop the restart steps and this fails.
    """
    commands = [command for command, _ in plan(ready()).steps]
    pull = next(i for i, c in enumerate(commands) if "ollama pull" in c)
    stopped = next(i for i, c in enumerate(commands) if "Stop-Process" in c)
    started = next(i for i, c in enumerate(commands) if "Start-Process" in c)

    assert stopped < started < pull, (
        "the service must be stopped and started again before the pull")


def test_the_variable_is_set_before_the_service_is_restarted():
    """Restarting it before the variable exists achieves nothing, which is the
    same bug one step earlier."""
    commands = [command for command, _ in plan(ready()).steps]
    setx = next(i for i, c in enumerate(commands) if c.startswith("setx"))
    stopped = next(i for i, c in enumerate(commands) if "Stop-Process" in c)
    assert setx < stopped
