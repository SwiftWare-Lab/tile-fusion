
import triton
import triton.language as tl
import torch

DEVICE = torch.device("cuda:0")


@triton.jit
def matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float32)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


@triton.jit
def matmul_a_stationary_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m


    #pid_m, pid_k = tl.program_id(0), tl.program_id(1)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_ak = (pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) % K
    offs_bk = (pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) % K
    offs_bn = tl.arange(0, BLOCK_SIZE_N)
    #offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    #&A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + (m : m+BLOCK_SIZE_M)[:, None]*A.stride(0) + (k : k+BLOCK_SIZE_K)[None, :]*A.stride(1);

    # create a copy of the block of A
    # a[i:i + block_size_m, k:k + block_size_k]
    a_ptrs = a_ptr + (offs_am[:, None]*stride_am + offs_ak [None, :]*stride_ak)
    a_block = tl.load(a_ptrs, mask=offs_ak[None, :] < K, other=0.0)
    #print("a block", a_block)

    b_ptrs = b_ptr + (offs_bk[:, None]*stride_bk + offs_bn[None, :]*stride_bn)
    c_ptrs = c_ptr + (offs_am[:, None]*stride_cm + offs_bn[None, :]*stride_cn)
    for j in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        # load the block of B
        b_block = tl.load(b_ptrs)
        #print("b block", b_block)
        pmul = tl.dot(a_block, b_block)
        # atomic add to the output C
        tl.atomic_add(c_ptrs, pmul)
        #tl.store(c_ptrs, tl.load(c_ptrs) + pmul)
        c_ptrs += BLOCK_SIZE_N * stride_cn
        b_ptrs += BLOCK_SIZE_N * stride_bn



def tiled_matmul_cpu(a, b, block_size_m, block_size_n, block_size_k):
    M, K = a.shape
    K, N = b.shape
    c = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    for i in range(0, M, block_size_m):
        for k in range(0, K, block_size_k):
            tmp_a = a[i:i + block_size_m, k:k + block_size_k]
            for j in range(0, N, block_size_n):
                c[i:i + block_size_m, j:j + block_size_n] += tmp_a @ b[k:k + block_size_k, j:j + block_size_n]
    return c


def matmul_triton(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    META = {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 8}
    # 1D launch kernel where each block gets its own program.
    #grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    grid = (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        META['BLOCK_SIZE_M'], META['BLOCK_SIZE_N'], META['BLOCK_SIZE_K'],  #
        META['GROUP_SIZE_M'],  #
        ACTIVATION=activation  #
    )
    return c


def matmul_triton_a_stationary(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    META = {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 8}
    # 1D launch kernel where each block gets its own program.
    #grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    #grid = (triton.cdiv(M, META['BLOCK_SIZE_M']) , triton.cdiv(K, META['BLOCK_SIZE_K']) )
    grid = (
    triton.cdiv(M, META['BLOCK_SIZE_M'])* triton.cdiv(K, META['BLOCK_SIZE_K']),)
    matmul_a_stationary_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        META['BLOCK_SIZE_M'], META['BLOCK_SIZE_N'], META['BLOCK_SIZE_K'],  #
        META['GROUP_SIZE_M'],  #
        ACTIVATION=activation  #
    )
    return c

# main entry
if __name__ == "__main__":
    # testing
    a = torch.randn(512, 64, device=DEVICE, dtype=torch.float32)
    b = torch.randn(64, 64, device=DEVICE, dtype=torch.float32)
    c = matmul_triton(a, b)

    # Compare with PyTorch
    c_ref = a @ b

    # triton
    print("Triton matmul")
    c_triton = matmul_triton(a, b)

    print("mul cpu")
    c_cpu = tiled_matmul_cpu(a.cpu(), b.cpu(), 32, 32, 32)

    print("matmul a stationary triton")
    c_triton_a_stationary = matmul_triton_a_stationary(a, b)

    # compare results if close
    if torch.allclose(c_triton, c_ref, atol=1e-5):
        print("Are results close ")
    else:
        print("Results are not close")
        print(torch.sum(c_triton - c_ref))

    if torch.allclose(c_cpu, c_ref.cpu(), atol=1e-5):
        print("CPU Are results close ")
    else:
        print("CPU Results are not close")

    if torch.allclose(c_triton_a_stationary, c_ref, atol=1e-5):
        print("A stationary Are results close ")
    else:
        print("Results are not close")
        # print difference
        print(torch.sum(c_triton_a_stationary - c_ref))