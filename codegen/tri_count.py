def tri_count(A_csr, n, X):
    row_ptr = A_csr.row_ptr
    col_idx = A_csr.col_idx
    values = A_csr.values
    for i in range(n):
        for j in range(row_ptr[i], row_ptr[i + 1]):
            k = col_idx[j]
            for l in range(row_ptr[k], row_ptr[k + 1]):
                m = col_idx[l]
                X[i, m] += values[j] * values[l]
    sum = 0
    for i in range(n):
        for j in range(n):
            sum += X[i, j]
    sum = sum / 6
    return sum
