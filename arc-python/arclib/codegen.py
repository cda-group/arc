from arclib.utils import *
import cloudpickle


def write_args(operator, arc, rs):
    extern_funs = []
    args = []
    for (id, arg) in enumerate(operator.args):
        name = "{}_arg{}".format(operator.instance(), id)
        path = "target/pickles/{}".format(name)

        # Pickle argument and write it to a file
        with open(path, 'wb') as file:
            cloudpickle.dump(arg.value, file)

        if arg.type.is_function():
            args.append('{}'.format(name))

            inputs = [t.generate_arc() for t in arg.type.inputs]
            output = arg.type.output.generate_arc()

            vars = ["x{}".format(i) for (i, _) in enumerate(inputs)]
            params = ", ".join(["{}: {}".format(v, t)
                                for (v, t) in zip(vars, inputs)])

            extern_funs.append('''
            | extern fun {name}({params}): {output};
            '''.format(name=name, params=params, output=output))

            inputs = [t.generate_rs() for t in arg.type.inputs]
            output = arg.type.output.generate_rs()

            vars = ["x{}".format(i) for (i, _) in enumerate(inputs)]
            params = ", ".join(["{}: {}".format(v, t)
                                for (v, t) in zip(vars, inputs)])
            tuple = "({})".format("".join(["{},".format(var) for var in vars]))

            rs.write_trim('''
            | fn {name}({params}) -> {output} {{
            |     Python::with_gil(|py| py.unpickle("{path}")?.call1({tuple})?.extract()).unwrap()
            | }}
            '''.format(name=name, path=path, params=params, output=output, tuple=tuple))

        else:
            # Non-function values are lifted into functions
            # and then evaluated
            args.append('{}()'.format(name))

            output = arg.type.generate_arc()

            extern_funs.append('''
            | extern fun {name}(): {output};
            '''.format(name=name, output=output))

            output = arg.type.generate_rs()

            rs.write_trim('''
            | fn {name}() -> {output} {{
            |     Python::with_gil(|py| py.unpickle("{path}").extract()).unwrap()
            | }}
            '''.format(name=name, output=output, path=path))
    arc.write_sep(args)
    return extern_funs


# Serialize all arguments of an operator
def generate(operators):
    with Writer('target/main.arc') as arc, Writer('target/main.rs') as rs:
        arc_extern_funs = []
        # Write implementations
        for operator in sorted(operators, key=lambda x: x.name):
            arc.write(operator.implementation())

        # Write instances
        operators = sorted(operators, key=lambda x: x.id)
        arc.writeln()
        arc.write('fun main()')
        with Brace(arc):
            for operator in operators:
                arc.write('    val ')
                if len(operator.ostreams) == 1:
                    arc.write('{}'.format(operator.ostreams[0]))
                else:
                    with Paren(arc):
                        arc.write_sep(operator.ostreams)
                arc.write(' = {}'.format(operator.name))
                with Paren(arc):
                    arc_extern_funs.extend(write_args(operator, arc, rs))
                with Paren(arc):
                    arc.write_sep(operator.istreams)
                arc.writeln(';')
        arc.writeln()
        for arc_extern_fun in arc_extern_funs:
            arc.write_trim(arc_extern_fun)
