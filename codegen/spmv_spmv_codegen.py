import numpy as np
from numba import njit, float64, int32
from numba.core import ir, ir_utils, config
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.ir_utils import mk_unique_var, next_label, mk_range_block, mk_loop_header, find_topo_order, \
    replace_var_names, dprint_func_ir
from numba.core.untyped_passes import IRProcessing


# from codegen.writer import mk_array_assign_stmt

# not usable

def mk_fused_loop_nest(
        loc, scope, nl, l_ptr, par_ptr, ids, types,
        next_block_label, first_loop_body_block, second_loop_body_block):
    # nl_var = ir.Var(scope, nl, loc)
    l_ptr_var = ir.Var(scope, l_ptr, loc)
    par_ptr_var = ir.Var(scope, par_ptr, loc)
    ids_var = ir.Var(scope, ids, loc)
    types_var = ir.Var(scope, types, loc)
    l_index = ir.Var(scope, mk_unique_var('l1'), loc)
    p_index = ir.Var(scope, mk_unique_var('p1'), loc)
    i_index = ir.Var(scope, mk_unique_var('i1'), loc)
    type_map = {}
    call_types = {}
    new_blocks = {}
    # creating level loop
    l_range_block = mk_range_block(
        type_map,
        0,
        nl,
        1,
        call_types,
        scope,
        loc)
    l_phi_var = l_range_block.body[-2].target
    l_header_block = mk_loop_header(type_map, l_phi_var, call_types,
                                    scope, loc)
    l_header_block.body[-1].target = 'l_range_block'
    l_range_block.body[-1].truebr = 'l_body_block'
    l_range_block.body[-1].falsebr = next_block_label
    new_blocks['l_header_block'] = l_header_block
    new_blocks['l_range_block'] = l_range_block
    l_index_ass = ir.Assign(l_phi_var, l_index, loc)
    l_body_block = ir.Block(scope, loc)
    l_body_block.body.append(l_index_ass)
    # creating partition loop
    p_loop_blocks, p_phi_var = mk_pointer_loop(loc, scope, l_index, l_ptr_var, 'p')
    p_header_block = p_loop_blocks['p_header_block']
    p_range_block = p_loop_blocks['p_range_block']
    p_range_block.body[-1].truebr = 'p_body_block'
    p_range_block.body[-1].falsebr = 'l_range_block'
    for st in p_header_block.body:
        l_body_block.body.append(st)
    new_blocks['l_body_block'] = l_body_block
    new_blocks['p_range_block'] = p_range_block  # Fix branch targets
    p_index_ass = ir.Assign(p_phi_var, p_index, loc)
    p_body_block = ir.Block(scope, loc)
    p_body_block.body.append(p_index_ass)
    i_loop_blocks, i_phi_var = mk_pointer_loop(loc, scope, p_index, par_ptr_var, 'i')
    i_header_block = i_loop_blocks['i_header_block']
    i_range_block = i_loop_blocks['i_range_block']
    i_range_block.body[-1].truebr = 'i_body_block'
    i_range_block.body[-1].falsebr = 'p_range_block'
    for st in i_header_block.body:
        p_body_block.append(st)
    new_blocks['p_body_block'] = p_body_block
    new_blocks['i_range_block'] = i_range_block  # Fix branch targets
    i_body_block = ir.Block(scope, loc)   # loops body will be here in a conditional statement
    i_index_ass = ir.Assign(i_phi_var, i_index, loc)
    i_body_block.append(i_index_ass)
    get_type_expr = ir.Expr.getitem(types_var, i_index, loc)
    type_var = ir.Var(scope, mk_unique_var('t'), loc)
    type_ass = ir.Assign(get_type_expr, type_var, loc)
    i_body_block.append(type_ass)
    get_iter_expr = ir.Expr.getitem(ids_var, i_index, loc)
    first_loop_body_block.body[0].value = get_iter_expr
    second_loop_body_block.body[0].value = get_iter_expr

    # if condition
    # bool_func = ir.Var(scope, 'bool_func', loc)
    # bool_func_ass = ir.Assign(ir.Global('bool', bool.__class__, loc), bool_func, loc)
    # i_body_block.append(bool_func_ass)

    type_cond = ir.Branch(type_var, 'second_loop_body_block', 'first_loop_body_block', loc)
    i_body_block.append(type_cond)
    first_loop_body_block.body[-1].target = 'i_range_block'
    second_loop_body_block.body[-1].target = 'i_range_block'
    new_blocks['first_loop_body_block'] = first_loop_body_block
    new_blocks['second_loop_body_block'] = second_loop_body_block
    return new_blocks




config.DEBUG_ARRAY_OPT = 3


@register_pass(mutates_CFG=False, analysis_only=False)
class LoopFusion1(FunctionPass):
    _name = "loop_fusion1"

    def __init__(self):
        super().__init__(self)

    def find_adjacent_loops(self, cfg):
        """Find adjacent loops that can be fused.
        """
        # Find all loops
        all_loops, adjacent_loops = [], []

        for lp in sorted(cfg.loops()):
            all_loops.append(cfg.loops()[lp])
        # for every two items in all_loops, check if they are adjacent
        for i in range(len(all_loops) - 1):
            if all_loops[i].exits == all_loops[i + 1].entries:
                adjacent_loops.append((all_loops[i], all_loops[i + 1]))
        return adjacent_loops

    # good numba overview : https://medium.com/rapids-ai/the-life-of-a-numba-kernel-a-compilation-pipeline-taking-user-defined-functions-in-python-to-cuda-71cc39b77625
    #
    def run_pass(self, state):
        """
        Do fusion of parfor nodes.
        """

        # a very specific and dirty way to fuse two adjacent loops. Not generic, just for testing
        mutate = True
        func_ir = state.func_ir
        blocks = func_ir.blocks
        dprint_func_ir(func_ir, "before maximize fusion down")
        cfg = ir_utils.compute_cfg_from_blocks(blocks)
        for node in cfg.loops():
            print(node)
        order = find_topo_order(blocks)
        # find adjacent loops
        # adj_loops = self.find_adjacent_loops(cfg)
        # var_lp1 = blocks[adj_loops[0][0].header].body[0].target
        # idx_lp1, idx_lp2 = blocks[42].body[0].target, blocks[80].body[0].target
        # body_loop2 = {}
        # # copy body of loop2
        # body_loop2[80] = blocks[80].copy()
        # # modify the jump target to loop 1
        # body_loop2[80].body[-1].target = 40
        # # replace idx_lp2 in body_loop2 with idx_lp1
        # replace_var_names(body_loop2, {idx_lp2.name: idx_lp1.name})
        # #tgt = body_loop2[80].body[-2].target
        # #val = body_loop2[42].body[2].value
        #
        # for i, stmt in enumerate(body_loop2[80].body[1:4]):
        #     blocks[42].body.insert(4+i, stmt)
        # #blocks[42].body[-1].target = 80
        # # remove loop 2 header
        # del (blocks[80])
        # del(blocks[62])
        # del(blocks[78])
        # blocks[40].terminator.falsebr = 100
        #
        # aa = ir.Var(blocks[0].scope, mk_unique_var("aa"), state.func_ir.loc)
        # bb = ir.Var(blocks[0].scope, mk_unique_var("bb"), state.func_ir.loc)
        # A = ir.Var(blocks[0].scope, mk_unique_var("A"), state.func_ir.loc)
        # iii = ir.Var(blocks[0].scope, mk_unique_var("iii"), state.func_ir.loc)
        # tt, ttt = mk_array_assign_stmt(A, aa, iii, iii, state.func_ir.loc, func_ir.blocks[0].scope)

        # create a new loop, not used here but can be used, see the writer file where there should be a function to create a loop
        # range_label = next_label()
        # header_label = next_label()
        # typemap = {}
        # calltypes = {}
        scope = func_ir.blocks[0].scope
        loc = state.func_ir.loc
        # print(mk_fusion_loop(loc,scope,100))
        # start = ir.Var(scope, mk_unique_var("start"), loc)
        # stop = ir.Var(scope, mk_unique_var("stop"), loc)
        # step = ir.Var(scope, mk_unique_var("step"), loc)
        # index_variable = ir.Var(scope, mk_unique_var("index_variable"), loc)
        # range_block = mk_range_block(
        #     typemap,
        #     start,
        #     stop,
        #     step,
        #     calltypes,
        #     scope,
        #     loc)
        # new_blocks = {}
        # range_block.body[-1].target = header_label  # fix jump target
        # phi_var = range_block.body[-2].target
        # new_blocks[range_label] = range_block
        # header_block = mk_loop_header(typemap, phi_var, calltypes,
        #                               scope, loc)
        # header_block.body[-2].target = index_variable
        # new_blocks[header_label] = header_block
        # jump to this new inner loop
        # init_block.body.append(ir.Jump(range_label, loc))
        # header_block.body[-1].falsebr = block_label

        # prev_header_label = header_label  # to set truebr next loop

        # generate a loop that adds two arrays
        # dprint_func_ir(func_ir, "after maximize fusion down")
        # add rand to the last block

        return mutate  # the pass has not mutated the IR


class MyCompiler(CompilerBase):  # custom compiler extends from CompilerBase

    def define_pipelines(self):
        # define a new set of pipelines (just one in this case) and for ease
        # base it on an existing pipeline from the DefaultPassBuilder,
        # namely the "nopython" pipeline
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
        # Add the new pass to run after IRProcessing
        pm.add_pass_after(LoopFusion1, IRProcessing)
        # pm.add_pass_after(ParFdd, IRProcessing)
        # finalize
        pm.finalize()
        # return as an iterable, any number of pipelines may be defined!
        return [pm]


# @njit(float64[:](int32[:], int32[:], float64[:], float64[:], int32, int32[:], int32[:], int32[:], int32[:]),
#       pipeline_class=MyCompiler)
def spmv_spmv(Ap, Ai, Ax, B, m, l_ptr, par_ptr, ids, types):
    y = np.zeros(m)
    z = np.zeros(m)
    for i in range(m):
        for j in range(Ap[i], Ap[i + 1]):
            y[i] += Ax[j] * B[Ai[j]]
    for i in range(m):
        for j in range(Ap[i], Ap[i + 1]):
            z[i] += Ax[j] * y[Ai[j]]
    return y


@njit(pipeline_class=MyCompiler)
def test_if(a):
    b = 0
    if a == 1:
        b += 1
    else:
        b += 2
    return b


def fused_spmv_spmv(Ap, Ai, Ax, B, m, lnum, lptr, pr_ptr, id, type):
    for i1 in range(lnum):
        for j1 in range(lptr[i1], lptr[i1 + 1]):
            for k1 in range(pr_ptr[j1], pr_ptr[j1 + 1]):
                ii = id[k1]
                t = type[k1]


# @njit(int32(int32), pipeline_class=MyCompiler)
# def test_if_condition(x):
#     y = 0
#     z = 0
#     if x == 0:
#         y += 1
#     else:
#         z +=1
#     return y
# print(numba.__version__)
n = 100
A = np.ones(n)
IA = np.zeros(n + 1, dtype=np.int32)
JA = np.zeros(n, dtype=np.int32)
x = np.random.random(n)
for i in range(n):
    IA[i] = i
    JA[i] = i
IA[n] = n
# y = spmv_spmv(IA, JA, A, x, n)
y = test_if(4)
# print(y)
# y = test_if_condition(1)
print(y)
