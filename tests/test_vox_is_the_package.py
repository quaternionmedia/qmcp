"""`import vox` must reach the vox package, not a directory that shares its name.

qmcp's editable install puts the project root on `sys.path`. With the
submodule at a top-level `vox/` -- a directory with no `__init__.py` --
`import vox` resolved to that directory as an empty namespace package. Its
submodules still imported, so a check on `vox.stt` passed, while
`from vox import HttpSTT` failed and `qmcp human voice` reported vox as not
importable. These import for real, with nothing faked.
"""

from pathlib import Path


def test_vox_resolves_to_the_package_itself():
    import vox

    assert vox.__file__ is not None, (
        f"vox is a namespace package over {list(vox.__path__)} -- a directory named "
        "vox is shadowing the installed package"
    )
    from vox import HttpSTT, VoiceSession  # names only vox/__init__ provides

    assert HttpSTT and VoiceSession


def test_nothing_named_vox_sits_at_the_project_root():
    """The project root is on sys.path, so a `vox/` there shadows the package.
    A clone updated across the move to vendor/vox can keep the old one."""
    root = Path(__file__).resolve().parents[1]
    assert not (root / "vox").exists(), f"stray {root / 'vox'} shadows the vox package"
