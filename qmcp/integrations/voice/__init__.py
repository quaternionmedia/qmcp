"""Voice HITL: answer qmcp's human-in-the-loop queue by voice.

Vendors the `vox` seam (STT over HTTP against an engine, TTS locally) to speak a
pending approval's prompt aloud and parse the spoken reply as yes/no,
instead of a human typing `qmcp human respond`.

Usage:
    from qmcp.integrations.voice import VoiceApprovalLoop
    from vox import HttpSTT
    from vox.adapters import JOE
    from vox.adapters.pyttsx3 import Pyttsx3TTS

    stt = HttpSTT("http://127.0.0.1:8000", contract=JOE)
    loop = VoiceApprovalLoop(stt=stt, tts=Pyttsx3TTS())
    loop.run_once("deploy-001")       # answer one pending request
    loop.run_forever()                # keep answering as new ones arrive
"""

from qmcp.integrations.voice.adapter import (
    UnclearResponse,
    VoiceApprovalLoop,
    choose_option,
    match_option,
    parse_yes_no,
    say_options,
    speakable,
    speakably,
)

__all__ = [
    "VoiceApprovalLoop",
    "UnclearResponse",
    "parse_yes_no",
    "choose_option",
    "match_option",
    "say_options",
    "speakable",
    "speakably",
]
