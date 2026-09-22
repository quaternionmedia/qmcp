"""Voice HITL: answer qmcp's human-in-the-loop queue by voice.

Vendors the `vox` seam (STT via a joe engine, TTS locally) to speak a
pending approval's prompt aloud and parse the spoken reply as yes/no,
instead of a human typing `qmcp human respond`.

Usage:
    from qmcp.integrations.voice import VoiceApprovalLoop
    from vox import JoeSTT, Pyttsx3TTS

    loop = VoiceApprovalLoop(stt=JoeSTT(), tts=Pyttsx3TTS())
    loop.run_once("deploy-001")       # answer one pending request
    loop.run_forever()                # keep answering as new ones arrive
"""

from qmcp.integrations.voice.adapter import (
    UnclearResponse,
    VoiceApprovalLoop,
    parse_yes_no,
)

__all__ = ["VoiceApprovalLoop", "UnclearResponse", "parse_yes_no"]
