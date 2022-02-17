from arclib.stream import *
from arclib.operator import *
from arclib.pipeline import *


def test_pipeline1():
    data = [1, 2, 3, 4]
    p = Pipeline() \
        .iterator_source(data) \
        .map(lambda x: x + 1) \
        .filter(lambda x: x % 2 == 0) \
        .execute()


def test_pipeline2():
    data = [1, 2, 3, 4]
    s0 = Pipeline().iterator_source(data)
    s1 = Map(lambda x: x + 1)(s0)
    s2 = Filter(lambda x: x % 2 == 0)(s1)
    s2.execute()
