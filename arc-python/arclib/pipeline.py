from arclib.operator import *
from arclib.stream import *
from arclib.context import *

class Pipeline:
    def __init__(self):
        self.ctx = Context()

    def iterator_source(self, iter):
        return IteratorSource(iter)(self.ctx)

    def execute(self):
        self.ctx.execute()
