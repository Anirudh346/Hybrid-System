"""Initialize models package"""
from models.device import Device
from models.user import User
from models.favorite import Favorite
from models.saved_search import SavedSearch
from models.price_history import PriceHistory
from models.comparison import Comparison

__all__ = [
    "Device",
    "User",
    "Favorite",
    "SavedSearch",
    "PriceHistory",
    "Comparison"
]
