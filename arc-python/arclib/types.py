from arclib.codegen import *


# Arguments of an operator
class Arg():
    def __init__(self, value, type):
        self.value = value
        self.type = type


class Type():
    def is_function(self): return False

    def generate_arc(self):
        sb = StringBuilder()
        self.write_arc(sb)
        return str(sb)

    def generate_rs(self):
        sb = StringBuilder()
        self.write_rs(sb)
        return str(sb)


class Nominal(Type):
    def __init__(self, name):
        self.name = name

    def write_arc(self, w):
        w.write(self.name)

    def write_rs(self, w):
        w.write(self.name)


class Fun(Type):
    def __init__(self, inputs, output):
        self.inputs = inputs
        self.output = output

    def is_function(self): return True

    def write_arc(self, w):
        w.write("fun")
        with Paren(w):
            if len(self.inputs) > 0:
                for elem in self.inputs[0:-1]:
                    elem.write_arc(w)
                    self.write(', ')
                self.inputs[-1].write_arc(w)
        w.write(":")
        self.output.write_arc(w)

    def write_rs(self, w):
        pass


class Vec(Type):
    def __init__(self, elem_type):
        self.elem_type = elem_type

    def write_arc(self, w):
        with Brack(w):
            self.elem_type.write_arc(w)

    def write_rs(self, w):
        w.write("Vec")
        with Angle(w):
            self.elem_type.write_rs(w)


class Iter(Type):
    def __init__(self, elem_type):
        self.elem_type = elem_type

    def write_arc(self, w):
        w.write("Iter")
        with Brack(w):
            self.elem_type.write_arc(w)

    def write_rs(self, w):
        w.write("Iterator")
        with Angle(w):
            self.elem_type.write_rs(w)


Any = Nominal("Any")
Bool = Nominal("bool")
