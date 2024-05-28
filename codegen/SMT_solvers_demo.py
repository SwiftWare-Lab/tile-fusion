from z3 import *

Z = IntSort()
I1 = Int('I1')
I2 = Int('I2')
J1 = Int('J1')
J2 = Int('J2')
n = Int('n')
Ap = Function('Ap', Z, Z)
Ai = Function('Ai', Z, Z)
b1 = And(I1 < n, I1 >= 0)
b2 = And(I2 < n, I2 >= 0)
b3 = And(J1 >= Ap(I1), J1 < Ap(I1+1))
b4 = And(J2 >= Ap(I2), J2 < Ap(I2+1))
#domain specific assertions
ApMonotonicty = ForAll([I1,I2], Implies(I1 < I2 , Ap(I1) < Ap(I2)))
AiPeriodicMonotonicty = ForAll([I1,J1,J2], Implies(And(J1 >= Ap(I1), J1 < Ap(I1+1), J2 >= Ap[I1], J2 < Ap[I1+1], J1 < J2) , Ai(J1) < Ai(J2)))
# dep =

# print("num args: ", n.num_args())
# print("children: ", n.children())
# print("1st child:", n.arg(0))
# print("2nd child:", n.arg(1))
# print("operator: ", n.decl())
# print("op name:  ", n.decl().name())
