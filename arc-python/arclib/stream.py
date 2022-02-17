class Stream:

    def __init__(self, producer):
        self.ctx = producer.ctx
        self.producer = producer
        self.consumers = []
        self.register()

    def register(self):
        self.ctx.streams.append(self)
        self.id = len(self.ctx.streams)

    def __str__(self):
        return "s{}".format(self.id)

    def map(self, f):
        from arclib.operator import Map
        return Map(f)(self)

    def filter(self, f):
        from arclib.operator import Filter
        return Filter(f)(self)

    def flat_map(self, f):
        from arclib.operator import FlatMap
        return FlatMap(f)(self)

    def split(self, f):
        from arclib.operator import Split
        return Split(f)(self)

    def merge(self, other):
        from arclib.operator import Union
        return Union()(self, other)

    def key_by(self, f):
        from arclib.operator import KeyBy
        return KeyBy(f)(self)

    def execute(self):
        self.ctx.execute()
