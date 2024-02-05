
# a set of standalone functions to generate different statements and expressions in numba IR

from numba.core import ir, ir_utils, config, errors
from numba.core.ir_utils import *


def mk_array_assign_stmt(c, a, b, index, loc, scope):
    """ make c[index] = a * b
    """
    # create a new array assignment statement
    # c[index] = a * b
    # create the target variable
    target = ir.Var(scope, mk_unique_var("target"), loc)
    # create the expression
    expr = ir.Expr.binop(ir.BINOPS_TO_OPERATORS['*'], a, b, loc)
    # create the assignment statement
    stmt = ir.Assign(expr, target, loc)
    # create the getitem expression
    getitem = ir.Expr.getitem(c, index, loc)
    # create the setitem statement
    setitem = ir.SetItem(c, index, target, loc)
    # return the statements
    return [stmt, setitem]


def mk_loop(loop_body, index_var, start, stop, step, loc, scope):
    """ make a for loop
    """
    range_label = next_label()
    header_label = next_label()
    typemap = {}
    calltypes = {}
    start = ir.Var(scope, mk_unique_var("start"), loc)
    stop = ir.Var(scope, mk_unique_var("stop"), loc)
    step = ir.Var(scope, mk_unique_var("step"), loc)
    index_variable = ir.Var(scope, mk_unique_var("index_variable"), loc)
    range_block = mk_range_block(
        typemap,
        start,
        stop,
        step,
        calltypes,
        scope,
        loc)
    new_blocks = {}
    range_block.body[-1].target = header_label  # fix jump target
    phi_var = range_block.body[-2].target
    new_blocks[range_label] = range_block
    header_block = mk_loop_header(typemap, phi_var, calltypes,
                                  scope, loc)
    header_block.body[-2].target = index_variable
    new_blocks[header_label] = header_block
    return new_blocks