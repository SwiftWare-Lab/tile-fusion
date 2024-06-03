
from numba.core import ir, ir_utils, config
from numba.core.compiler import CompilerBase, DefaultPassBuilder
from numba.core.compiler_machinery import FunctionPass, register_pass
from numba.core.ir_utils import mk_unique_var, next_label, mk_range_block, mk_loop_header, find_topo_order, \
    replace_var_names, dprint_func_ir
from numba.core.untyped_passes import IRProcessing, ReconstructSSA
from numba import njit, float64, int32
import numpy as np
from copy import deepcopy


config.DEBUG_ARRAY_OPT = 3
class Access:
    def __init__(self, var, index):
        self.var = var
        self.index = index

    def __str__(self):
        return f"{self.var}[{self.index}]" if self.index is not None else self.var

    def __repr__(self):
        return self.__str__()

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
        loop_bodies = {}
        # print(cfg.loops())
        loops = cfg.loops()
        lp_exits = set()
        lp_pars = {}
        for lp in loops:
            lp_exits.add(loops[lp].exits.pop())
        for lp in loops:
            loop_bodies[lp] = []
            if lp not in lp_pars:
                covered_loops = set()
                self.find_loop_structure(cfg, lp, covered_loops, lp_pars)
            loop = loops[lp]
            for x in loop.body:
                if x not in loops and x not in lp_exits:
                    loop_bodies[lp].append(x)
        return loop_bodies, lp_pars

    def find_loop_structure(self, cfg, loop, covered_loops, lp_pars):
        in_loops = cfg.in_loops(loop)
        if len(in_loops) == 1:
            pass
        else:
            for il in in_loops:
                il_header = il.header
                if il_header != loop:
                    self.find_loop_structure(cfg, il_header, covered_loops, lp_pars)
            for il in in_loops:
                il_header = il.header
                if il_header != loop:
                    if il_header not in covered_loops:
                        lp_pars[loop] = il_header
                        covered_loops.add(il_header)

    def extract_iter_variables(self, blocks, loop_bodies, lp_pars):
        iter_vars_list = {}
        iter_vars = {}
        for lp, body_labels in loop_bodies.items():
            iter_vars_list[lp] = []
            for b_label in body_labels:
                st = blocks[b_label].body[0]
                iter_vars_list[lp].append(st.target)
        for lp, lp_par in lp_pars.items():
            for var in iter_vars_list[lp]:
                iter_vars_list[lp_par].remove(var)
        for lp, iters in iter_vars_list.items():
            iter_vars[lp] = iters[0]
        return iter_vars

    def create_spf_tree(self, lp_pars, lp_headers):
        spf_tree = {-1: []}
        for lp in lp_headers:
            spf_tree[lp] = []
            if lp not in lp_pars:
                spf_tree[-1].append(lp)
        for lp, par in lp_pars.items():
            spf_tree[par].append(lp)
        for lp in spf_tree:
            spf_tree[lp] = sorted(spf_tree[lp])
        return spf_tree

    def present_accesses(self, writes, reads, spf_tree, iter_vars, iter_set,loop):
        iter_set.append(iter_vars[loop].name)
        for lp in spf_tree[loop]:
            self.present_accesses(writes,reads, spf_tree, iter_vars, iter_set, lp)
        if len(writes[loop].keys()) > 0:
            print("write:", iter_set, "----->", writes[loop])
        if len(reads[loop].keys()) > 0:
            print("write:", iter_set, "----->", reads[loop])
        iter_set.pop()

    def run_pass(self, state):
        # state contains the FunctionIR to be mutated,
        mutate = False

        # # along with environment and typing information
        func_ir = state.func_ir
        ir_blocks = func_ir.blocks
        dprint_func_ir(func_ir, "generated IR")
        loop_bodies, lp_pars = self.find_all_loop_bodies(ir_utils.compute_cfg_from_blocks(func_ir.blocks))
        iter_vars = self.extract_iter_variables(ir_blocks, loop_bodies, lp_pars)
        spf_tree = self.create_spf_tree(lp_pars, loop_bodies.keys())
        print(spf_tree)
        print(iter_vars)
        # self.find_loop_structure(ir_utils.compute_cfg_from_blocks(func_ir.blocks))
        # print(loop_bodies)
        header_bodies = {}
        for lp,par in lp_pars.items():
            for body in loop_bodies[par]:
                if body not in loop_bodies[lp]:
                    header_bodies[par] = body
            if lp not in header_bodies:
                header_bodies[lp] = loop_bodies[lp][0]
        # print(header_bodies)
        # bodies_blocks = set([x for xs in loop_bodies.values() for x in xs])  # flattening loop_bodies_blocks
        reads = {}
        writes = {}
        for header_block, body_block in header_bodies.items():
            reads[header_block] = {}
            writes[header_block] = {}
            read_stmnts = {}
            data_values = {}
            stmnts = ir_blocks[body_block].body
            for st in stmnts:
                if isinstance(st, ir.Assign):
                    data_values[st.target.name] = st.value
            # print(data_values)
            for s, st in enumerate(stmnts):
                if isinstance(st, ir.SetItem): # or (isinstance(st, ir.Assign) and not st.target.name.startswith('$') and st.target.name not in iter_vars.values()):
                    writes[header_block][s] = Access(st.target.name, st.index)
                    read_stmnts[s] = []
                    reads[header_block][s] = []
                    self.find_reads(data_values, st.value, read_stmnts[s])
                    # print(read_stmnts[s]) #TODO: convert get item to read access
                    for r in read_stmnts[s]:
                        val = r.value
                        temp_s_reads = []
                        self.find_reads(data_values, r.index, temp_s_reads)
                        if len(temp_s_reads) > 0:
                            if isinstance(temp_s_reads[0], ir.Var):
                                reads[header_block][s].append(Access(val, temp_s_reads[0]))
                            elif isinstance(temp_s_reads[0], ir.Expr) and temp_s_reads[0].op == 'getitem':
                                temp_s_reads2 = []
                                self.find_reads(data_values, temp_s_reads[0].index, temp_s_reads2)
                                if len(temp_s_reads2) > 0:
                                    ind_acc = Access(temp_s_reads[0].value, temp_s_reads2[0])
                                    reads[header_block][s].append(Access(val, ind_acc))
                        else:
                            reads[header_block][s].append(Access(val, None))
        print(writes)
        print(reads)
        for lp in spf_tree[-1]:
            self.present_accesses(writes, reads, spf_tree, iter_vars, [], lp)

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

# @njit(float64[:](int32[:], int32[:], float64[:], float64[:]), pipeline_class=SPFGenerator)
# def test_find_dependencies(Ap, Ai, Ax, B):
#     m = len(Ap) - 1
#     y = np.zeros(m)
#     z = np.zeros(m)
#     for i in range(m):
#         for j in range(Ap[i], Ap[i + 1]):
#             y[i] += Ax[j] * B[Ai[j]]
#         for k in range(m):
#             for p in range(m):
#                 y[p] += 5
#             # y[i] *= 2
#     for i in range(m):
#         # y[i] = y[i] + 1
#         z[i] = y[i] * 2
#     return z

@njit(float64[:](int32[:], int32[:], float64[:], float64[:], int32, int32[:], int32[:], int32[:], int32[:]),
      pipeline_class=SPFGenerator)
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
