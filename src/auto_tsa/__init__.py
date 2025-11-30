"""
AutoTSA: high-performance AutoML for time series forecasting.
"""

from .config import AutoTSAConfig
from .search import AutoTSA

__all__ = ["AutoTSA", "AutoTSAConfig"]
