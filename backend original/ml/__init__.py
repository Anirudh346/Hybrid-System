"""Initialize ML package"""
from ml.recommender import recommender, DeviceRecommender
from ml.nlp_parser import nlp_parser, NLPQueryParser

__all__ = [
    "recommender",
    "DeviceRecommender",
    "nlp_parser",
    "NLPQueryParser"
]
