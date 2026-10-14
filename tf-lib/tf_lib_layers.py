from __future__ import annotations

from collections.abc import Sequence

import torch

import tf_lib


def _adjoint_spmm(adj: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
    adj_t = adj.to_sparse_coo().transpose(0, 1).coalesce()
    return torch.sparse.mm(adj_t, dense)


class _FusedGeMMSpMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        adj: torch.Tensor,
        feature: torch.Tensor,
        weight: torch.Tensor,
        schedule: Sequence[torch.Tensor],
        num_threads: int,
        variable_tile: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(adj, feature, weight)
        ctx.num_threads = num_threads
        if variable_tile:
            return tf_lib.torch_fused_gemm_spmm_vt(
                adj,
                feature,
                weight,
                list(schedule),
                num_threads,
            )
        return tf_lib.torch_fused_gemm_spmm(
            adj,
            feature,
            weight,
            list(schedule),
            num_threads,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        adj, feature, weight = ctx.saved_tensors
        grad_feature = grad_weight = None
        grad_intermediate = _adjoint_spmm(adj, grad_output)
        if ctx.needs_input_grad[1]:
            grad_feature = grad_intermediate.matmul(weight)
        if ctx.needs_input_grad[2]:
            grad_weight = grad_intermediate.transpose(0, 1).matmul(feature)
        return None, grad_feature, grad_weight, None, None, None


class _CachedSpMMGeMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        af: torch.Tensor,
        weight: torch.Tensor,
        num_threads: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(af, weight)
        return tf_lib.torch_cached_spmm_gemm(af, weight, num_threads)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        af, weight = ctx.saved_tensors
        grad_af = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_af = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.transpose(0, 1).matmul(af)
        return grad_af, grad_weight, None


def fused_gemm_spmm(
    adj: torch.Tensor,
    feature: torch.Tensor,
    weight: torch.Tensor,
    schedule: Sequence[torch.Tensor],
    num_threads: int,
    variable_tile: bool = True,
) -> torch.Tensor:
    return _FusedGeMMSpMMFunction.apply(
        adj,
        feature,
        weight,
        schedule,
        num_threads,
        variable_tile,
    )


def cached_spmm_gemm(
    af: torch.Tensor,
    weight: torch.Tensor,
    num_threads: int,
) -> torch.Tensor:
    return _CachedSpMMGeMMFunction.apply(af, weight, num_threads)


def unfused_gemm_spmm(
    adj: torch.Tensor,
    feature: torch.Tensor,
    weight: torch.Tensor,
    num_threads: int,
) -> torch.Tensor:
    torch.set_num_threads(num_threads)
    return torch.sparse.mm(adj, feature).matmul(weight.transpose(0, 1))


class FusedGCNLayer(torch.nn.Module):
    def __init__(
        self,
        feat_dim: int,
        embed_dim: int,
        adj: torch.Tensor,
        schedule: Sequence[torch.Tensor],
        num_threads: int,
        variable_tile: bool = True,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(embed_dim, feat_dim))
        torch.nn.init.xavier_uniform_(self.weight)
        self.adj = adj
        self.schedule = list(schedule)
        self.num_threads = num_threads
        self.variable_tile = variable_tile

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_gemm_spmm(
            self.adj,
            x,
            self.weight,
            self.schedule,
            self.num_threads,
            self.variable_tile,
        )


class FirstLayerGCNCached(torch.nn.Module):
    def __init__(
        self,
        feat_dim: int,
        embed_dim: int,
        adj: torch.Tensor,
        feature: torch.Tensor,
        num_threads: int,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(embed_dim, feat_dim))
        torch.nn.init.xavier_uniform_(self.weight)
        self.register_buffer("af", torch.sparse.mm(adj, feature))
        self.num_threads = num_threads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cached_spmm_gemm(self.af, self.weight, self.num_threads)


class UnFusedGCNLayer(torch.nn.Module):
    def __init__(
        self,
        feat_dim: int,
        embed_dim: int,
        adj: torch.Tensor,
        num_threads: int,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(embed_dim, feat_dim))
        torch.nn.init.xavier_uniform_(self.weight)
        self.adj = adj
        self.num_threads = num_threads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return unfused_gemm_spmm(self.adj, x, self.weight, self.num_threads)


class FusedGCN(torch.nn.Module):
    def __init__(
        self,
        feat_dim: int,
        embed_dim: int,
        num_classes: int,
        adj: torch.Tensor,
        num_threads: int,
        cache_size: int = 500_000,
    ) -> None:
        super().__init__()
        schedule1 = tf_lib.torch_inspect_vt(
            adj,
            feat_dim,
            embed_dim,
            cache_size,
            num_threads,
        )
        schedule2 = tf_lib.torch_inspect_vt(
            adj,
            embed_dim,
            num_classes,
            cache_size,
            num_threads,
        )
        self.conv1 = FusedGCNLayer(
            feat_dim,
            embed_dim,
            adj,
            schedule1,
            num_threads,
            variable_tile=True,
        )
        self.conv2 = FusedGCNLayer(
            embed_dim,
            num_classes,
            adj,
            schedule2,
            num_threads,
            variable_tile=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = x.relu()
        return self.conv2(x)
