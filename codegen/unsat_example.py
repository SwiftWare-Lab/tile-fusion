from z3 import *

N = Const('N', IntSort())

I1 = Int('I1')
I2 = Int('I2')
m = Const('m', IntSort())
k = Const('k', IntSort())
# col = Function('col', IntSort(), IntSort())
row_ptr = Function('row_ptr', IntSort(), IntSort())
dependencies = Exists([m,k], And(I1 < I2, m == k, I1 < N, I1 >= 0, I2 < N, I2 >= 0, row_ptr(I1-1) <= m, row_ptr(I1) > m, row_ptr(I2) <= k, row_ptr(I2+1) > k))
ApMonotonicty = ForAll([I1,I2], Implies(I1 < I2, row_ptr(I1) < row_ptr(I2)))

solve(And(ForAll([I1, I2], Implies(I1 < I2, row_ptr(I1) < row_ptr(I2))), Exists([m,k], And(I1 < I2, m == k, I1 < N, I1 >= 0, I2 < N, I2 >= 0, row_ptr(I1-1) <= m, row_ptr(I1) > m, row_ptr(I2) <= k, row_ptr(I2+1) > k))))