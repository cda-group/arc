from arclib.stream import *
from arclib.types import *
from arclib.utils import *


class Operator:

    def instance(self):
        return "{}{}".format(self.name, self.id)

    def register(self):
        self.id = len(self.ctx.operators)
        self.ctx.operators.append(self)

    def __call__(self, istream):
        self.ctx = istream.ctx
        self.register()
        istream.consumers.append(self)
        ostream = Stream(self)
        self.istreams = [istream]
        self.ostreams = [ostream]
        return ostream

    def execute(self):
        self.ctx.execute()

    def implementation(self):
        return trim(self._implementation())


class Map(Operator):
    def __init__(self, f):
        self.name = "Map"
        self.args = [Arg(f, Fun([Any], Any))]

    def _implementation(self):
        return """
        | task Map[A, B](f: fun(A):B): ~A -> ~B {
        |     loop {
        |         on event => emit f(event)
        |     }
        | }
        """


class Filter(Operator):
    def __init__(self, f):
        self.name = "Filter"
        self.args = [Arg(f, Fun([Any], Bool))]

    def _implementation(self):
        return """
        | task Filter[A](f: fun(A):bool): ~A -> ~A {
        |     loop {
        |         on event => if f(event) {
        |             emit event
        |         }
        |     }
        | }
        """


class FlatMap(Operator):
    def __init__(self, f):
        self.name = "FlatMap"
        self.args = [Arg(f, Fun([Any], Vec(Any)))]

    def _implementation(self):
        return """
        | task FlatMap[A,B](f: fun(A):[B]): ~A -> ~B {
        |     loop {
        |         on event => for event in f(event) {
        |             emit event
        |         }
        |     }
        | }
        """


class Split(Operator):
    def __init__(self, f):
        self.name = "Split"
        self.args = [Arg(f, Fun([Any], Bool))]

    def __call__(self, istream):
        self.ctx = istream.ctx
        self.register()
        istream.consumers.append(self)
        ostream0 = Stream(self)
        ostream1 = Stream(self)
        self.istreams = [istream]
        self.ostreams = [ostream0, ostream1]
        return ostream0, ostream1

    def _implementation(self):
        return """
        | task Split[T](f: fun(T):bool): ~T -> (A(~T), B(~T)) {
        |     loop {
        |         on event => if f(event) {
        |             emit A(event)
        |         } else {
        |             emit B(event)
        |         }
        |     }
        | }
        """


class Union(Operator):
    def __init__(self):
        self.name = "Union"
        self.args = []

    def __call__(self, istream0, istream1):
        self.ctx = istream0.ctx
        self.register()
        istream0.consumers.append(self)
        istream1.consumers.append(self)
        ostream = Stream(self)
        self.istreams = [istream0, istream1]
        self.ostreams = [ostream]
        return ostream

    def _implementation(self):
        return """
        | task Union[T](): (A(~T), B(~T)) -> ~T {
        |     loop {
        |         on A(event) => emit event,
        |         on B(event) => emit event,
        |     }
        | }
        """


class Fold(Operator):
    def __init__(self, f, init):
        self.name = "KeyBy"
        self.args = [Arg(f, Fun([Any], Any)), Arg(init, Any)]

    def _implementation(self):
        return """
        | task Fold[A,T](f: fun(A,T):A, id: T): ~A -> ~T {
        |     var agg = id;
        |     loop {
        |         on event => {
        |             agg = f(agg, event);
        |             emit agg
        |         }
        |     }
        | }
        """


class KeyBy(Operator):
    def __init__(self, f):
        self.name = "KeyBy"
        self.args = [Arg(f, Fun([Any], Any))]

    def _implementation(self):
        return """
        | task KeyBy[T,K](f: fun(T):K): ~{v:T} -> ~{k:K,v:T} {
        |     loop {
        |         on event => emit f(event),
        |     }
        | }
        """


class IteratorSource(Operator):
    def __init__(self, iter):
        self.name = "IteratorSource"
        self.args = [Arg(iter, Iter(Any))]

    def __call__(self, ctx):
        self.ctx = ctx
        self.register()
        ostream = Stream(self)
        self.istreams = []
        self.ostreams = [ostream]
        return ostream

    def _implementation(self):
        return """
        | task IteratorSource[T](iter: Iterator[T]): () -> ~{k:T,v:T} {
        |     loop {
        |         match iter.next() {
        |             Some(event) => emit event,
        |             None => break
        |         }
        |     }
        | }
        """
