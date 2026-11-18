from .reference_engine import ReferenceEngine
from .stable_engine import FacePowers, SMat, StableEngine
from .tmmax_engine import TmmaxEngine

ENGINES = {"stable": StableEngine, "tmmax": TmmaxEngine, "reference": ReferenceEngine}


def get_engine(name: str):
    return ENGINES[name]()


__all__ = ["ENGINES", "FacePowers", "ReferenceEngine", "SMat", "StableEngine", "TmmaxEngine", "get_engine"]
