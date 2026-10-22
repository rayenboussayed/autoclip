"""The autoclip engine: configuration, media, transcription, moment selection, rendering.

This package holds everything that is not a user interface. The command line (``autoclip.cli``)
and the desktop app (``autoclip.app``) are thin layers on top of it: they build a :class:`Config`,
hand it to :func:`autoclip.core.pipeline.process_video` and show what happened.

The modules are deliberately free of heavy imports: ``moviepy``, ``faster_whisper``, ``cv2`` and
the other optional libraries are imported inside the functions that need them, so importing a
single module stays cheap.
"""
