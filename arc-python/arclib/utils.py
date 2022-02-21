import re


def trim(s):
    return re.sub(r' +\| ', '', s)


class Writer():
    def __init__(self, path):
        self.file = open(path, 'w')

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.file.close()

    def write(self, s='\n'):
        self.file.write(s)

    def write_trim(self, s):
        self.write(trim(s))

    def writeln(self, s=''):
        self.write(s)
        self.write('\n')

    def write_sep(self, list):
        if len(list) > 0:
            for elem in list[0:-1]:
                self.write(str(elem))
                self.write(', ')
            self.write(str(list[-1]))


class StringBuilder(Writer):
    def __init__(self):
        self.parts = []

    def __str__(self):
        return ''.join(self.parts)

    def write(self, s='\n'):
        self.parts.append(s)


class Delim():
    def __enter__(self):
        self.writer.write(self.l)

    def __exit__(self, type, value, traceback):
        self.writer.write(self.r)


class Brace(Delim):
    def __init__(self, writer):
        self.l = ' {\n'
        self.r = '}'
        self.writer = writer


class Paren(Delim):
    def __init__(self, writer):
        self.l = '('
        self.r = ')'
        self.writer = writer


class Angle(Delim):
    def __init__(self, writer):
        self.l = '<'
        self.r = '>'
        self.writer = writer


class Brack(Delim):
    def __init__(self, writer):
        self.l = '['
        self.r = ']'
        self.writer = writer
