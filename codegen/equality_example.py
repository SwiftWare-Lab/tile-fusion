import islpy as isl

space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT, set=["i1", "i2"],params=["N"])
bset = (isl.BasicSet.universe(space)
        .add_constraint(isl.Constraint.ineq_from_names(space, {"i1": 1}))
        .add_constraint(isl.Constraint.ineq_from_names(space, {"N": 1, "i1": -1}))
        .add_constraint(isl.Constraint.ineq_from_names(space, {"i2": 1}))
        .add_constraint(isl.Constraint.ineq_from_names(space, {"N": 1, "i2": -1})))
# i1 = isl.BasicSet("[N] -> { A[i] : 0 <= i < N }")
# i2 = isl.BasicSet("[N] -> { [i] : 0 <= i < N }")
f = isl.BasicMap("[N] -> {f[[i] -> [j]]: j < N and j >= 0 and i < N and i >= 0}")
# f = isl.BasicMap("{[i] -> [j]: 0 <= i,j < N}")
ds_pred = isl.Set("[e1,e2]: e1 < e2 implies f(e1) < f(e2)")
print(f.space)
#express a relation that has monotonoic property

# dependence_rel = isl.BasicMap("{[i1[i] -> i2[j]]: i1[i] = i2[j]}")
# g_i = isl.BasicMap("{i1[i] -> f_i[i]}")
# print(f_i.())


