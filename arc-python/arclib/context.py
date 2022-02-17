from arclib.codegen import generate

class Context:
    def __init__(self):
        self.operators = []
        self.streams = []

    def execute(self):
        generate(self.operators)

