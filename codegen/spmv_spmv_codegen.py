import numpy as np
import numba
from numba import njit, float64, int32
from numba.core import ir, ir_utils, config
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.ir_utils import mk_unique_var, next_label, mk_range_block, mk_loop_header, find_topo_order, \
    replace_var_names, dprint_func_ir
from numba.core.untyped_passes import IRProcessing, ReconstructSSA
from scipy.io import mmread
from numba.core import typed_passes
import graphviz as gv

# from codegen.writer import mk_array_assign_stmt

RANGE_JUMP_INDEX = -1
RANGE_PHI_VAR_INDEX = -2
HEADER_BR_INDEX = -1
HEADER_PHI_VAR_INDEX = -2
LOOP_VAR_INDEX = 0
LOOP_BODY_INDEX = 0
LOOP_EXIT_INDEX = -1


def mk_pointer_loop(loc, scope, ptr_index, ptr_var, prefix, header_block_label):
    type_map = {}
    call_types = {}
    start_index = ptr_index
    end_index = ir.Var(scope, mk_unique_var('end_index'), loc)
    const1_var = ir.Var(scope, mk_unique_var('const_one'), loc)
    const1_ass = ir.Assign(
        ir.Const(1, loc),
        const1_var,
        loc
    )
    end_index_ass = ir.Assign(
        ir.Expr.binop(
            ir.BINOPS_TO_OPERATORS['+'],
            start_index, const1_var, loc),
        end_index,
        loc)
    p_start = ir.Var(scope, mk_unique_var('p_start'), loc)
    p_end = ir.Var(scope, mk_unique_var('p_end'), loc)
    start_ass = ir.Assign(ir.Expr.getitem(ptr_var, start_index, loc), p_start, loc)

    end_ass = ir.Assign(ir.Expr.getitem(ptr_var, end_index, loc), p_end, loc)
    range_block = mk_range_block(
        type_map,
        p_start,
        p_end,
        1,
        call_types,
        scope,
        loc
    )
    phi_var = range_block.body[-2].target
    header_block = mk_loop_header(type_map, phi_var, call_types,
                                  scope, loc)
    range_block.body.insert(0, end_ass)
    range_block.body.insert(0, start_ass)
    range_block.body.insert(0, end_index_ass)
    range_block.body.insert(0, const1_ass)
    # need to update target of jump instruction of header block after
    header_label = prefix + '_header_block'
    range_label = prefix + '_range_block'
    print(range_label)
    range_block.body[-1].target = header_block_label
    return {
        header_label: header_block,
        range_label: range_block
    }, phi_var


def mk_fused_loop_nest(
        loc, scope, nl, l_ptr, par_ptr, ids, types,
        next_block_label, first_loop_body_blocks, second_loop_body_blocks,
        new_labels, loop1_labels, loop2_labels):
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
    l_phi_var = l_range_block.body[RANGE_PHI_VAR_INDEX].target
    l_header_block = mk_loop_header(type_map, l_phi_var, call_types,
                                    scope, loc)
    l_range_block.body[RANGE_JUMP_INDEX].target = new_labels['l_header_block']
    l_header_block.body[HEADER_BR_INDEX].truebr = new_labels['l_body_block']
    l_header_block.body[HEADER_BR_INDEX].falsebr = next_block_label
    new_blocks[new_labels['l_header_block']] = l_header_block
    new_blocks[new_labels['l_range_block']] = l_range_block
    l_index_var = l_header_block.body[HEADER_PHI_VAR_INDEX].target
    l_index_ass = ir.Assign(l_index_var, l_index, loc)
    l_body_block = ir.Block(scope, loc)
    l_body_block.body.append(l_index_ass)
    # creating partition loop
    p_loop_blocks, p_phi_var = mk_pointer_loop(loc, scope, l_index, l_ptr_var, 'p', new_labels['p_header_block'])
    p_header_block = p_loop_blocks['p_header_block']
    p_range_block = p_loop_blocks['p_range_block']
    p_header_block.body[HEADER_BR_INDEX].truebr = new_labels['p_body_block']
    p_header_block.body[HEADER_BR_INDEX].falsebr = new_labels['l_header_block']
    for st in p_range_block.body:
        l_body_block.body.append(st)
    new_blocks[new_labels['l_body_block']] = l_body_block
    new_blocks[new_labels['p_header_block']] = p_header_block  # Fix branch targets
    p_index_ass = ir.Assign(p_header_block.body[HEADER_PHI_VAR_INDEX].target, p_index, loc)
    p_body_block = ir.Block(scope, loc)
    p_body_block.body.append(p_index_ass)
    i_loop_blocks, i_phi_var = mk_pointer_loop(loc, scope, p_index, par_ptr_var, 'i', new_labels['i_header_block'])
    i_header_block = i_loop_blocks['i_header_block']
    i_range_block = i_loop_blocks['i_range_block']
    i_header_block.body[HEADER_BR_INDEX].truebr = new_labels['i_body_block']
    i_header_block.body[HEADER_BR_INDEX].falsebr = new_labels['p_header_block']
    for st in i_range_block.body:
        p_body_block.body.append(st)
    new_blocks[new_labels['p_body_block']] = p_body_block
    new_blocks[new_labels['i_header_block']] = i_header_block  # Fix branch targets
    i_body_block = ir.Block(scope, loc)  # loops body will be here in a conditional statement
    i_index_ass = ir.Assign(i_header_block.body[HEADER_PHI_VAR_INDEX].target, i_index, loc)
    i_body_block.append(i_index_ass)
    get_type_expr = ir.Expr.getitem(types_var, i_index, loc)
    type_var = ir.Var(scope, mk_unique_var('t'), loc)
    type_ass = ir.Assign(get_type_expr, type_var, loc)
    i_body_block.append(type_ass)
    get_iter_expr = ir.Expr.getitem(ids_var, i_index, loc)
    first_loop_body_blocks[loop1_labels[LOOP_BODY_INDEX]].body[
        LOOP_VAR_INDEX].value = get_iter_expr  # hardcoded for now, need to be fixed, loop1_body = 32
    del second_loop_body_blocks[loop2_labels[LOOP_BODY_INDEX]].body[
        LOOP_VAR_INDEX]  # hardcoded for now, need to be fixed, loop2_body=106
    i_body_block.append(first_loop_body_blocks[loop1_labels[LOOP_BODY_INDEX]].body[LOOP_VAR_INDEX])
    del first_loop_body_blocks[loop1_labels[LOOP_BODY_INDEX]].body[LOOP_VAR_INDEX]

    # if condition
    # bool_func = ir.Var(scope, 'bool_func', loc)
    # bool_func_ass = ir.Assign(ir.Global('bool', bool.__class__, loc), bool_func, loc)
    # i_body_block.append(bool_func_ass)

    type_cond = ir.Branch(type_var, loop2_labels[LOOP_BODY_INDEX], loop1_labels[LOOP_BODY_INDEX],
                          loc)  # hardcoded for now, need to be fixed
    i_body_block.append(type_cond)
    new_blocks[new_labels['i_body_block']] = i_body_block
    first_loop_body_blocks[loop1_labels[LOOP_EXIT_INDEX]].body[RANGE_JUMP_INDEX].target = new_labels[
        'i_header_block']  # hardcoded for now, need to be fixed # merge these two lines
    second_loop_body_blocks[loop2_labels[LOOP_EXIT_INDEX]].body[RANGE_JUMP_INDEX].target = new_labels[
        'i_header_block']  # hardcoded for now, need to be fixed
    # new_blocks[new_labels['first_loop_body_block']] = first_loop_body_block
    # new_blocks[new_labels['second_loop_body_block']] = second_loop_body_block
    for key, block in first_loop_body_blocks.items():
        new_blocks[key] = block
    for key, block in second_loop_body_blocks.items():
        new_blocks[key] = block
    return new_blocks


config.DEBUG_ARRAY_OPT = 3


@register_pass(mutates_CFG=False, analysis_only=False)
class LoopFusion1(FunctionPass):
    _name = "loop_fusion1"

    def __init__(self):
        super().__init__(self)

    # This needs to be fixed, it is not right for all cases
    def find_adjacent_loops(self, cfg):
        """Find adjacent loops that can be fused.
        """
        # Find all loops
        all_loops, adjacent_loops = [], []

        for lp in sorted(cfg.loops()):
            all_loops.append(cfg.loops()[lp])
        # for every two items in all_loops, check if they are adjacent
        for i in range(len(all_loops) - 1):
            for j in range(i, len(all_loops)):
                if all_loops[i].exits == all_loops[j].entries:
                    adjacent_loops.append((all_loops[i], all_loops[j]))
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
        print("loops:")
        for l in cfg.loops():
            print(l)
        print("------------------------------------")
        order = find_topo_order(blocks)
        print("order:")
        print(order)
        print("------------------------------------")
        print(cfg.in_loops(30), cfg.in_loops(104))
        new_label = max(ir_utils.next_label(), max(func_ir.blocks.keys()) + 1)
        new_labels = dict()
        new_labels['l_range_block'] = new_label
        new_labels['l_header_block'] = new_label + 1
        new_labels['l_body_block'] = new_label + 2
        new_labels['p_header_block'] = new_label + 3
        new_labels['p_body_block'] = new_label + 4
        new_labels['i_header_block'] = new_label + 5
        new_labels['i_body_block'] = new_label + 6
        new_labels['first_loop_body_block'] = new_label + 7
        new_labels['second_loop_body_block'] = new_label + 8
        loop1_body = {}
        loop2_body = {}
        adjacent_loops = self.find_adjacent_loops(cfg)
        loop1 = adjacent_loops[0][0]
        loop2 = adjacent_loops[0][1]
        for x in loop1.body:
            if x != loop1.header:
                loop1_body[x] = blocks[x]
        for x in loop2.body:
            if x != loop2.header:
                loop2_body[x] = blocks[x]
        loop1_labels = sorted(loop1_body.keys())
        loop2_labels = sorted(loop2_body.keys())
        for key in loop1.body:
            del blocks[key]
        for item in loop1.exits:
            del blocks[item]
        for key in loop2.body:
            del blocks[key]
        blocks[0].body[-1].target = new_labels['l_range_block']

        scope = func_ir.blocks[0].scope
        loc = state.func_ir.loc
        loop2_exit = 0
        for item in loop2.exits:
            loop2_exit = item
        new_blocks = mk_fused_loop_nest(loc, scope, 2, 'l_ptr', 'par_ptr', 'ids', 'types', loop2_exit, loop1_body,
                                        loop2_body,
                                        new_labels, loop1_labels, loop2_labels)
        for key, block in new_blocks.items():
            blocks[key] = block
        dprint_func_ir(func_ir, "after maximize fusion down")
        # add rand to the last block

        return mutate  # the pass has not mutated the IR


class Access:
    def __init__(self, var, index):
        self.var = var
        self.index = index

    def __str__(self):
        return f"{self.var}[{self.index}]" if self.index is not None else self.var

    def __repr__(self):
        return self.__str__()


class MyCompiler(CompilerBase):  # custom compiler extends from CompilerBase

    def define_pipelines(self):
        # define a new set of pipelines (just one in this case) and for ease
        # base it on an existing pipeline from the DefaultPassBuilder,
        # namely the "nopython" pipeline
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
        # Add the new pass to run after IRProcessing
        pm.add_pass_after(LoopFusion1, IRProcessing)
        # if self.state.flags.auto_parallel.enabled:
        #     pm.add_pass_after(LoopFusion1, typed_passes.ParforFusionPass)
        # else:
        #     pm.add_pass_after(LoopFusion1, typed_passes.InlineOverloads)
        # pm.add_pass_after(ParFdd, IRProcessing)
        # finalize
        pm.finalize()
        # return as an iterable, any number of pipelines may be defined!
        return [pm]


@register_pass(mutates_CFG=False, analysis_only=False)
class ObserveIRPass(FunctionPass):
    _name = "dead_code_elimination1"

    def __init__(self):
        FunctionPass.__init__(self)

    def find_reads(self, data_value, var, reads):
        # print(data_value)
        if isinstance(var, ir.Expr) and var.op == 'getitem':
            reads.append(var)
        elif isinstance(var, ir.Expr) and (var.op == 'binop' or var.op == 'inplace_binop'):
            self.find_reads(data_value, var.lhs, reads)
            self.find_reads(data_value, var.rhs, reads)
        elif isinstance(var, ir.Assign):
            self.find_reads(data_value, var.value, reads)
        elif isinstance(var, ir.Var):
            # print(var.name)
            if not var.name.startswith('$'):
                reads.append(var)
            if var.name in data_value:
                self.find_reads(data_value, data_value[var.name], reads)
            # reads.append(var)

    def find_all_loop_bodies(self, cfg):
        loop_bodies = set()
        # print(cfg.loops())
        loops = cfg.loops()
        for lp in loops:
            loop = loops[lp]
            for x in loop.body:
                if x not in loops:
                    loop_bodies.add(x)
        return loop_bodies

    def run_pass(self, state):
        # state contains the FunctionIR to be mutated,
        mutate = False

        # # along with environment and typing information
        func_ir = state.func_ir
        ir_blocks = func_ir.blocks
        dprint_func_ir(func_ir, "generated IR")
        loop_bodies = self.find_all_loop_bodies(ir_utils.compute_cfg_from_blocks(func_ir.blocks))
        # print(loop_bodies)
        for block in sorted(loop_bodies):
            reads = {}
            writes = {}
            read_stmnts = {}
            data_values = {}
            stmnts = ir_blocks[block].body
            for st in stmnts:
                if isinstance(st, ir.Assign):
                    data_values[st.target.name] = st.value
            # print(data_values)
            for s, st in enumerate(stmnts):
                if isinstance(st, ir.SetItem):
                    writes[s] = Access(st.target.name, st.index)
                    read_stmnts[s] = []
                    self.find_reads(data_values, st.value, read_stmnts[s])
                    print(read_stmnts[s]) #TODO: convert get item to read access
                    for r in read_stmnts[s]:
                        val = r.value
                        temp_s_reads = []
                        self.find_reads(data_values, r.index, temp_s_reads)
                        if len(temp_s_reads) > 0:
                            if isinstance(temp_s_reads[0], ir.Var):
                                reads[s] = Access(val, temp_s_reads[0])
                            elif isinstance(temp_s_reads[0], ir.Expr) and temp_s_reads[0].op == 'getitem':
                                temp_s_reads2 = []
                                self.find_reads(data_values, temp_s_reads[0].index, temp_s_reads2)
                                if len(temp_s_reads2) > 0:
                                    ind_acc = Access(temp_s_reads[0].value, temp_s_reads2[0])
                                    reads[s] = Access(val,ind_acc)
                        else:
                            reads[s] = Access(val, None  )
            print(writes)
            print(reads)
            # print(data_values)
            # for dv in data_values:
            #     if not dv.startswith('$'):
            #         reads = []
            #         self.find_reads(data_values, data_values[dv],reads)
            #         print(dv, reads)
        # index = func_ir.blocks[44].body[5].value.index
        # data = func_ir.blocks[44].body[5].value.value
        # print(index, data)
        # read = func_ir.blocks[118].body[4].value
        # temp = read
        # print(read)
        # reads = []
        # self.find_reads(data_values, temp, reads)
        # data = reads[0].value
        # index = reads[0].index
        # print(data, index)

        # # create CFG from the IR
        # cfg = ir_utils.compute_cfg_from_blocks(func_ir.blocks)
        # cfg_simplied_blks = ir_utils.simplify_CFG(func_ir.blocks)
        # cfg_simplied = ir_utils.compute_cfg_from_blocks(cfg_simplied_blks)
        # print(func_ir.blocks)
        # print("=====================================")
        # # visualize the CFG
        # cfg.render_dot("cfg.dot").save('ttt.dot')
        # cfg_simplied.render_dot("cfg_simplied.dot").save('ttt_simplied.dot')
        # # convert the dot filt to png
        # #gv.render('dot', 'png', 'cfg.dot')
        # #gv.render('dot', 'bmp', 'ttt.dot')
        # # iterate over each block in the IR
        # for blk in func_ir.blocks.values():
        #     new_body = []
        #     # iterate over each statement in the block
        #     for stmt in blk.body:
        #         print(stmt)
        #         if isinstance(stmt, ir.Assign):
        #             # if the statement is an assignment
        #             # and the target is not used elsewhere
        #             if stmt.target.name not in func_ir._definitions:
        #                 mutate = True  # the pass will mutate the IR
        #                 continue  # skip this statement
        #         new_body.append(stmt)  # keep this statement
        #     blk.body = new_body  # update the block with new statements
        return mutate  # the pass has not mutated the IR


class SPFGenerator(CompilerBase):  # custom compiler extends from CompilerBase

    def define_pipelines(self):
        # define a new set of pipelines (just one in this case) and for ease
        # base it on an existing pipeline from the DefaultPassBuilder,
        # namely the "nopython" pipeline
        pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
        # Add the new pass to run after IRProcessing
        pm.add_pass_after(ObserveIRPass, ReconstructSSA)
        # if self.state.flags.auto_parallel.enabled:
        #     pm.add_pass_after(LoopFusion1, typed_passes.ParforFusionPass)
        # else:
        #     pm.add_pass_after(LoopFusion1, typed_passes.InlineOverloads)
        # pm.add_pass_after(ParFdd, IRProcessing)
        # finalize
        pm.finalize()
        # return as an iterable, any number of pipelines may be defined!
        return [pm]


@njit(float64[:](int32[:], int32[:], float64[:], float64[:], int32, int32[:], int32[:], int32[:], int32[:]),
      pipeline_class=MyCompiler)
def spmv_spmv(Ap, Ai, Ax, B, m, l_ptr, par_ptr, ids, types):
    y = np.zeros(m)
    z = np.zeros(m)
    Ax = Ax + 1
    for i in range(m):
        for j in range(Ap[i], Ap[i + 1]):
            y[i] += Ax[j] * B[Ai[j]]
    for i in range(m):
        for j in range(Ap[i], Ap[i + 1]):
            z[i] += Ax[j] * y[Ai[j]]
    return y


@njit(float64[:](int32[:], int32[:], float64[:], float64[:]), pipeline_class=SPFGenerator)
def test_find_dependencies(Ap, Ai, Ax, B):
    m = len(Ap) - 1
    y = np.zeros(m)
    z = np.zeros(m)
    for i in range(m):
        for j in range(Ap[i], Ap[i + 1]):
            y[i] += Ax[j] * B[Ai[j]]
            # y[i] *= 2
    for i in range(m):
        # y[i] = y[i] + 1
        z[i] = y[i] * 2
    return z


def spmv_spmv_python(Ap, Ai, Ax, B, m):
    y = np.zeros(m)
    z = np.zeros(m)
    Ax = Ax + 1
    for i in range(m):
        for j in range(Ap[i], Ap[i + 1]):
            y[i] += Ax[j] * B[Ai[j]]
    for i in range(m):
        for j in range(Ap[i], Ap[i + 1]):
            z[i] += Ax[j] * y[Ai[j]]
    return y


# @njit(pipeline_class=MyCompiler)
# def test_if(a):
#     b = 0
#     if a == 1:
#         b += 1
#     else:
#         b += 2
#     return b


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

def generate_schedule_arrays(Ap, Ai, Ax):
    n = len(Ap) - 1
    l_ptr = np.zeros(3, dtype=np.int32)
    par_ptr = np.zeros(17, dtype=np.int32)
    ids = np.zeros(len(Ai) * 2, dtype=np.int32)
    types = np.zeros(len(Ai) * 2, dtype=np.int32)
    l_ptr[1] = 8
    l_ptr[2] = 16
    if n % 8 == 0:
        tile_size = n // 8
    else:
        tile_size = n // 8 + 1
    unfused_iterations = []
    cur_index = 0
    for ii in range(0, n, tile_size):
        start = ii
        end = min(ii + tile_size, n)
        fused_iters = []
        unfused_iterations.append([])
        for i in range(start, end):
            ids[cur_index] = i
            types[cur_index] = 0
            cur_index += 1
            if Ai[Ap[i]] >= start and Ai[Ap[i + 1] - 1] < end:
                fused_iters.append(i)
            else:
                unfused_iterations[-1].append(i)
        for fi in fused_iters:
            ids[cur_index] = fi
            types[cur_index] = 1
            cur_index += 1
        par_ptr[ii // tile_size + 1] = cur_index
    print(len(unfused_iterations))
    for i in range(0, 8):
        for j in range(len(unfused_iterations[i])):
            ids[cur_index] = unfused_iterations[i][j]
            types[cur_index] = 1
            cur_index += 1
        par_ptr[i + 9] = cur_index
    return l_ptr, par_ptr, ids, types


print(numba.__version__)

mat = mmread('../fusion/data/tri-banded/tri-banded-16.mtx')
csr = mat.tocsr()
n = csr.shape[0]
IA = csr.indptr.astype(np.int32)
JA = csr.indices.astype(np.int32)
A = csr.data.astype(np.float64)

# n = 80
# A = np.ones(n)
# IA = np.zeros(n + 1, dtype=np.int32)
# JA = np.zeros(n, dtype=np.int32)
x = np.random.random(n)

for i in range(n):
    IA[i] = i
    JA[i] = i
IA[n] = n

l_ptr, par_ptr, ids, types = generate_schedule_arrays(IA, JA, A)
# print(l_ptr, par_ptr, ids, types)
# l_ptr = np.array([0, 2, 2], dtype=np.int32)
# par_ptr = np.array([0, 100, 200], dtype=np.int32)
# ids = np.array([i for i in range(50)] * 2 + [i for i in range(50, 100)] * 2, dtype=np.int32)
# types = np.array([0] * 50 + [1] * 50 + [0] * 50 + [1] * 50, dtype=np.int32)
# y = spmv_spmv(IA, JA, A, x, n, l_ptr, par_ptr, ids, types)
# print(y)
y = test_find_dependencies(IA, JA, A, x)
# y = spmv_spmv_python(IA, JA, A, x, n)
# print(y)
# y = test_if(4)
# print(y)
# y = test_if_condition(1)
# print(y)
