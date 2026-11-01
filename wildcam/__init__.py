"""wildcam: find rare animals, including distant low-contrast herds, in fixed time-lapse cameras.

Per camera:
    frames (time-ordered) -> time-of-day background model (median of recent same-period frames)
      -> normalised change map -> horizon band from the long-term median image
      -> proposals (small blobs near the horizon, larger ones elsewhere) grouped into herds
      [+ optional MegaDetector for near, clearly visible animals]
      -> crop verifier (frozen CNN features of frame and background crops)
      -> temporal support from neighbouring frames -> image score -> triage
"""

__version__ = "0.1.0"
