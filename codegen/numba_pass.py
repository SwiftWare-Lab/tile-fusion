
from numba import njit, prange
from numba.core import ir, ir_utils, config, errors
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.typed_passes import PreParforPass, ParforPass, ParforFusionPass, AnnotateTypes, NoPythonBackend, \
    IRLegalization, NopythonTypeInference
from numba.parfors import parfor, array_analysis
from numba.core import typed_passes
from numba import float64, int32
from numba.core.ir_utils import find_topo_order, build_definitions, simplify, dprint_func_ir
from numba.core.ir_utils import *
from numba.core.untyped_passes import IRProcessing, FixupArgs, TranslateByteCode
from numba.parfors import parfor
from numbers import Number
import numpy as np
import numba
from writer import *
import graphviz as gv
from numba.parfors.parfor import simplify_parfor_body_CFG, get_parfor_params, get_parfor_reductions, maximize_fusion, \
    Parfor



# Register this pass with the compiler framework, declare that it will not
# mutate the control flow graph and that it is not an analysis_only pass (it
# potentially mutates the IR).
@register_pass(mutates_CFG=False, analysis_only=False)
class ConstsAddOne(FunctionPass):
    _name = "consts_add_one" # the common name for the pass

    def __init__(self):
        FunctionPass.__init__(self)

    # implement method to do the work, "state" is the internal compiler
    # state from the CompilerBase instance.
    def run_pass(self, state):
        func_ir = state.func_ir # get the FunctionIR object
        mutated = False # used to record whether this pass mutates the IR
        # walk the blocks
        for blk in func_ir.blocks.values():
            tt = blk.find_insts(ir.Branch)
            # find LHS of assignment
            pp = tt
            print(pp)
            # find the assignment nodes in the block and walk them
            for assgn in blk.find_insts(ir.Assign):
                # if an assignment value is a ir.Consts
                #print(assgn.value)
                if isinstance(assgn.value, ir.Const):
                    const_val = assgn.value
                    # if the value of the ir.Const is a Number
                    if isinstance(const_val.value, Number):
                        # then add one!
                        const_val.value += 1
                        mutated |= True
        return mutated  # return True if the IR was mutated, False if not.


@register_pass(mutates_CFG=False, analysis_only=False)
class DeadCodeElimination1(FunctionPass):
    _name = "dead_code_elimination1"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        # state contains the FunctionIR to be mutated,
        mutate = False
        # along with environment and typing information
        func_ir = state.func_ir
        # create CFG from the IR
        cfg = ir_utils.compute_cfg_from_blocks(func_ir.blocks)
        cfg_simplied_blks = ir_utils.simplify_CFG(func_ir.blocks)
        cfg_simplied = ir_utils.compute_cfg_from_blocks(cfg_simplied_blks)
        print(func_ir.blocks)
        print("=====================================")
        # visualize the CFG
        cfg.render_dot("cfg.dot").save('ttt.dot')
        cfg_simplied.render_dot("cfg_simplied.dot").save('ttt_simplied.dot')
        # convert the dot filt to png
        #gv.render('dot', 'png', 'cfg.dot')
        #gv.render('dot', 'bmp', 'ttt.dot')
        # iterate over each block in the IR
        for blk in func_ir.blocks.values():
            new_body = []
            # iterate over each statement in the block
            for stmt in blk.body:
                print(stmt)
                if isinstance(stmt, ir.Assign):
                    # if the statement is an assignment
                    # and the target is not used elsewhere
                    if stmt.target.name not in func_ir._definitions:
                        mutate = True  # the pass will mutate the IR
                        continue  # skip this statement
                new_body.append(stmt)  # keep this statement
            blk.body = new_body  # update the block with new statements
        return mutate  # the pass has not mutated the IR


def _find_mask(typemap, func_ir, arr_def):
    """check if an array is of B[...M...], where M is a
    boolean array, and other indices (if available) are ints.
    If found, return B, M, M's type, and a tuple representing mask indices.
    Otherwise, raise GuardException.
    """
    require(isinstance(arr_def, ir.Expr) and arr_def.op == 'getitem')
    value = arr_def.value
    index = arr_def.index
    value_typ = typemap[value.name]
    index_typ = typemap[index.name]
    ndim = value_typ.ndim
    require(isinstance(value_typ, types.npytypes.Array))
    if (isinstance(index_typ, types.npytypes.Array) and
            isinstance(index_typ.dtype, types.Boolean) and
            ndim == index_typ.ndim):
        return value, index, index_typ.dtype, None
    elif isinstance(index_typ, types.BaseTuple):
        # Handle multi-dimension differently by requiring
        # all indices to be constant except the one for mask.
        seq, op = find_build_sequence(func_ir, index)
        require(op == 'build_tuple' and len(seq) == ndim)
        count_consts = 0
        mask_indices = []
        mask_var = None
        for ind in seq:
            index_typ = typemap[ind.name]
            # Handle boolean mask
            if (isinstance(index_typ, types.npytypes.Array) and
                    isinstance(index_typ.dtype, types.Boolean)):
                mask_var = ind
                mask_typ = index_typ.dtype
                mask_indices.append(None)
            # Handle integer array selector
            elif (isinstance(index_typ, types.npytypes.Array) and
                  isinstance(index_typ.dtype, types.Integer)):
                mask_var = ind
                mask_typ = index_typ.dtype
                mask_indices.append(None)
            # Handle integer index
            elif isinstance(index_typ, types.Integer):
                count_consts += 1
                mask_indices.append(ind)

        require(mask_var and count_consts == ndim - 1)
        return value, mask_var, mask_typ, mask_indices
    raise GuardException




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
        order = find_topo_order(blocks)
        # find adjacent loops
        adj_loops = self.find_adjacent_loops(cfg)
        var_lp1 = blocks[adj_loops[0][0].header].body[0].target
        idx_lp1, idx_lp2 = blocks[42].body[0].target, blocks[80].body[0].target
        body_loop2 = {}
        # copy body of loop2
        body_loop2[80] = blocks[80].copy()
        # modify the jump target to loop 1
        body_loop2[80].body[-1].target = 40
        # replace idx_lp2 in body_loop2 with idx_lp1
        replace_var_names(body_loop2, {idx_lp2.name: idx_lp1.name})
        #tgt = body_loop2[80].body[-2].target
        #val = body_loop2[42].body[2].value

        for i, stmt in enumerate(body_loop2[80].body[1:4]):
            blocks[42].body.insert(4+i, stmt)
        #blocks[42].body[-1].target = 80
        # remove loop 2 header
        del (blocks[80])
        del(blocks[62])
        del(blocks[78])
        blocks[40].terminator.falsebr = 100

        aa = ir.Var(blocks[0].scope, mk_unique_var("aa"), state.func_ir.loc)
        bb = ir.Var(blocks[0].scope, mk_unique_var("bb"), state.func_ir.loc)
        A = ir.Var(blocks[0].scope, mk_unique_var("A"), state.func_ir.loc)
        iii = ir.Var(blocks[0].scope, mk_unique_var("iii"), state.func_ir.loc)
        tt, ttt = mk_array_assign_stmt(A, aa, iii, iii, state.func_ir.loc, func_ir.blocks[0].scope)


        # create a new loop, not used here but can be used, see the writer file where there should be a function to create a loop
        range_label = next_label()
        header_label = next_label()
        typemap = {}
        calltypes = {}
        scope = func_ir.blocks[0].scope
        loc = state.func_ir.loc
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
        # jump to this new inner loop
        # init_block.body.append(ir.Jump(range_label, loc))
        # header_block.body[-1].falsebr = block_label

        prev_header_label = header_label  # to set truebr next loop

        # generate a loop that adds two arrays
        dprint_func_ir(func_ir, "after maximize fusion down")
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
        #pm.add_pass_after(ParFdd, IRProcessing)
        # finalize
        pm.finalize()
        # return as an iterable, any number of pipelines may be defined!
        return [pm]


# this is an example of adjacent loops
@njit(float64[:](float64[:], int32[:], int32[:], float64[:]), pipeline_class=MyCompiler)  # JIT compile using the custom compiler
def adjacent_loop(A, IA, JA, x):
    y = np.zeros(len(IA) - 1)
    for j in range(0, len(IA) - 1, 1):
        y[j] += j
    for k in range(len(IA) - 1):
        y[k] += k
    return y


# generate a random sparse matrix CSR format
print(numba.__version__)
n = 100
A = np.ones(n)
IA = np.zeros(n + 1, dtype=np.int32)
JA = np.zeros(n, dtype=np.int32)
x = np.random.random(n)
for i in range(n):
    IA[i] = i
    JA[i] = i
IA[n] = n
y = adjacent_loop(A, IA, JA, x)
print(y)
