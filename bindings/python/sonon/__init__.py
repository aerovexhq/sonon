"""
Sonon: High-Performance Robotics-Aimed Acoustic DSP & Wake-Word Spotting Package.
"""

from .client import KeywordEvent, SononEngine, SononShm

__version__ = "0.1.0"
__all__ = ["SononEngine", "SononShm", "KeywordEvent", "__version__"]
