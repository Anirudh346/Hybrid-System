"""Initialize models package.

This backend can run in multiple modes (legacy Beanie and current SQLAlchemy).
Import optional models defensively so missing optional dependencies (e.g. beanie)
don't prevent the API from starting in MySQL/SQLAlchemy deployments.
"""

from models.device import Device

__all__ = ["Device"]

try:
    from models.user import User
    __all__.append("User")
except Exception:
    User = None  # type: ignore

try:
    from models.favorite import Favorite
    __all__.append("Favorite")
except Exception:
    Favorite = None  # type: ignore

try:
    from models.saved_search import SavedSearch
    __all__.append("SavedSearch")
except Exception:
    SavedSearch = None  # type: ignore

try:
    from models.price_history import PriceHistory
    __all__.append("PriceHistory")
except Exception:
    PriceHistory = None  # type: ignore

try:
    from models.comparison import Comparison
    __all__.append("Comparison")
except Exception:
    Comparison = None  # type: ignore
