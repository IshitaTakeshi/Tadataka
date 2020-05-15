from sympy.utilities.codegen import codegen


def generate_c_code(name, expr, prefix):
    codegen((name, expr), prefix=prefix, language="C", to_files=True)
