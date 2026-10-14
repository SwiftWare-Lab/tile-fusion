"""Resolve the requested GNN graphs locally and download only missing data."""

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable

import numpy as np
import scipy.io
import scipy.sparse


DEFAULT_DATASETS = [
    "Physics",
    "Cora",
    "CiteSeer",
    "PubMed",
    "CoraFull",
    "Photo",
    "Computers",
    "CS",
    "Reddit",
    "Flickr",
    "ogbn-arxiv",
    "ogbn-products",
    "ogbn-proteins",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
FUSION_DATA = REPO_ROOT / "fusion" / "data"
MATERIALIZED_ROOT = FUSION_DATA / "triton-gnn"


@dataclass(frozen=True)
class DatasetLocation:
    name: str
    path: Path
    origin: str
    ordering: str = "original"


# Full OGB graphs in binding-project are intentionally preferred over the
# training-only matrices under fusion/data/end2end-test-data.
LOCAL_MTX = {
    "Physics": [
        FUSION_DATA / "end2end-test-data/coauthor_physics/coauthor_physics.mtx",
    ],
    "Cora": [FUSION_DATA / "end2end-test-data/cora/cora.mtx"],
    "CiteSeer": [MATERIALIZED_ROOT / "CiteSeer/citeseer.mtx"],
    "PubMed": [FUSION_DATA / "end2end-test-data/pubmed/pubmed.mtx"],
    "CoraFull": [FUSION_DATA / "end2end-test-data/cora_full/cora_full.mtx"],
    "Photo": [MATERIALIZED_ROOT / "Photo/photo.mtx"],
    "Computers": [MATERIALIZED_ROOT / "Computers/computers.mtx"],
    "CS": [FUSION_DATA / "end2end-test-data/coauthor_cs/coauthor_cs.mtx"],
    "Reddit": [FUSION_DATA / "end2end-test-data/reddit2/reddit2.mtx"],
    "Flickr": [FUSION_DATA / "end2end-test-data/flickr/flickr.mtx"],
    "ogbn-arxiv": [
        REPO_ROOT
        / "binding-project/data/end2end-test-data/ogbn-arxiv/ogbn-arxiv.mtx",
        FUSION_DATA / "end2end-test-data/ogbn_arxiv/ogbn_arxiv.mtx",
    ],
    "ogbn-products": [
        REPO_ROOT
        / "binding-project/data/end2end-test-data/ogbn-products/ogbn-products.mtx",
        FUSION_DATA / "end2end-test-data/ogbn_products/ogbn_products.mtx",
    ],
    "ogbn-proteins": [
        REPO_ROOT
        / "binding-project/data/end2end-test-data/ogbn-proteins/ogbn-proteins.mtx",
    ],
}


# Reuse the repository's full-graph RCM caches where they already exist.  The
# three PyG graphs without an existing ordered version are generated beside
# their materialized input with an ``_rcm`` suffix.
LOCAL_RCM_MTX = {
    "Physics": [
        REPO_ROOT
        / "binding-project/data/end2end-test-data/coauthor_physics_ordered/"
        "coauthor_physics_ordered.mtx",
    ],
    "Cora": [
        FUSION_DATA
        / "end2end-test-data-ordered/cora_ordered/cora_ordered.mtx",
    ],
    "CiteSeer": [MATERIALIZED_ROOT / "CiteSeer/citeseer_rcm.mtx"],
    "PubMed": [
        REPO_ROOT
        / "binding-project/data/end2end-test-data/pubmed_ordered/"
        "pubmed_ordered.mtx",
    ],
    "CoraFull": [
        REPO_ROOT
        / "binding-project/data/end2end-test-data/cora_full_ordered/"
        "cora_full_ordered.mtx",
    ],
    "Photo": [MATERIALIZED_ROOT / "Photo/photo_rcm.mtx"],
    "Computers": [MATERIALIZED_ROOT / "Computers/computers_rcm.mtx"],
    "CS": [
        REPO_ROOT
        / "binding-project/data/end2end-test-data/coauthor_cs_ordered/"
        "coauthor_cs_ordered.mtx",
    ],
    "Reddit": [
        REPO_ROOT
        / "binding-project/data/end2end-test-data/reddit2_ordered/"
        "reddit2_ordered.mtx",
    ],
    "Flickr": [
        REPO_ROOT
        / "binding-project/data/end2end-test-data/flickr_ordered/"
        "flickr_ordered.mtx",
    ],
    "ogbn-arxiv": [
        REPO_ROOT
        / "binding-project/data/end2end-test-data/ogbn-arxiv_ordered/"
        "ogbn-arxiv_ordered.mtx",
    ],
    "ogbn-products": [
        REPO_ROOT
        / "binding-project/data/end2end-test-data/ogbn-products_ordered/"
        "ogbn-products_ordered.mtx",
    ],
    "ogbn-proteins": [
        REPO_ROOT
        / "binding-project/data/end2end-test-data/ogbn-proteins_ordered/"
        "ogbn-proteins_ordered.mtx",
    ],
}


def _canonical_name(name: str) -> str:
    aliases = {item.lower().replace("_", "-"): item for item in DEFAULT_DATASETS}
    key = name.lower().replace("_", "-")
    if key not in aliases:
        raise KeyError(
            f"unknown dataset {name!r}; expected one of {', '.join(DEFAULT_DATASETS)}"
        )
    return aliases[key]


def _edge_index_to_csr(edge_index, number_of_nodes: int):
    rows = edge_index[0].detach().cpu().numpy()
    columns = edge_index[1].detach().cpu().numpy()
    matrix = scipy.sparse.coo_matrix(
        (np.ones(rows.size, dtype=np.float32), (rows, columns)),
        shape=(number_of_nodes, number_of_nodes),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def _materialize_pyg(name: str, allow_download: bool) -> Path:
    try:
        from torch_geometric import datasets as pyg_datasets
    except ImportError as error:
        raise RuntimeError(
            "torch-geometric is required to materialize missing datasets"
        ) from error

    output = LOCAL_MTX[name][0]
    output.parent.mkdir(parents=True, exist_ok=True)
    download_root = MATERIALIZED_ROOT / "downloads"

    factories: dict[str, Callable] = {
        "Physics": lambda root: pyg_datasets.Coauthor(root, name="Physics"),
        "Cora": lambda root: pyg_datasets.Planetoid(root, name="Cora"),
        "CiteSeer": lambda root: pyg_datasets.Planetoid(root, name="CiteSeer"),
        "PubMed": lambda root: pyg_datasets.Planetoid(root, name="PubMed"),
        "CoraFull": lambda root: pyg_datasets.CoraFull(root),
        "Photo": lambda root: pyg_datasets.Amazon(root, name="Photo"),
        "Computers": lambda root: pyg_datasets.Amazon(root, name="Computers"),
        "CS": lambda root: pyg_datasets.Coauthor(root, name="CS"),
        "Reddit": lambda root: pyg_datasets.Reddit2(root),
        "Flickr": lambda root: pyg_datasets.Flickr(root),
    }
    existing_roots = {
        "CiteSeer": REPO_ROOT / "tf-lib/data/pyg/citeseer",
    }
    root = download_root / name
    # Reprocess local raw files in our own cache.  This avoids depending on the
    # PyG version that serialized another project's processed/data.pt file.
    if name in existing_roots:
        source_raw = existing_roots[name] / name / "raw"
        target_raw = root / name / "raw"
        if source_raw.is_dir() and not target_raw.is_dir():
            target_raw.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_raw, target_raw)
    processed = list(root.glob("**/processed/data*.pt")) if root.exists() else []
    if not processed and not allow_download:
        raise FileNotFoundError(
            f"{name} is not cached locally and downloading is disabled"
        )
    if name not in factories:
        raise RuntimeError(f"no PyG materializer is registered for {name}")
    dataset = factories[name](str(root))
    graph = dataset[0]
    number_of_nodes = int(graph.num_nodes)
    matrix = _edge_index_to_csr(graph.edge_index, number_of_nodes)
    scipy.io.mmwrite(output, matrix, field="real", symmetry="general")
    return output


def _materialize_ogb(name: str, allow_download: bool) -> Path:
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as error:
        raise RuntimeError("ogb is required to download missing OGB datasets") from error

    output = MATERIALIZED_ROOT / name / f"{name}.mtx"
    output.parent.mkdir(parents=True, exist_ok=True)
    root = MATERIALIZED_ROOT / "downloads/ogb"
    expected = root / name.replace("-", "_")
    if not expected.exists() and not allow_download:
        raise FileNotFoundError(
            f"{name} is not cached locally and downloading is disabled"
        )
    dataset = PygNodePropPredDataset(name=name, root=str(root))
    graph = dataset[0]
    matrix = _edge_index_to_csr(graph.edge_index, int(graph.num_nodes))
    scipy.io.mmwrite(output, matrix, field="real", symmetry="general")
    return output


def resolve_dataset(name: str, allow_download: bool = True) -> DatasetLocation:
    name = _canonical_name(name)
    for candidate in LOCAL_MTX[name]:
        if candidate.is_file():
            origin = "materialized" if MATERIALIZED_ROOT in candidate.parents else "local"
            return DatasetLocation(name, candidate.resolve(), origin)

    if name.startswith("ogbn-"):
        path = _materialize_ogb(name, allow_download)
    else:
        path = _materialize_pyg(name, allow_download)
    return DatasetLocation(name, path.resolve(), "downloaded/materialized")


def resolve_datasets(names, allow_download: bool = True):
    return [resolve_dataset(name, allow_download) for name in names]


def _mtx_metadata(path: Path):
    """Read Matrix Market shape/count/symmetry without loading its entries."""
    with path.open("r") as stream:
        header = stream.readline().strip().lower().split()
        if len(header) != 5 or header[:2] != ["%%matrixmarket", "matrix"]:
            raise ValueError(f"invalid Matrix Market header: {path}")
        for line in stream:
            stripped = line.strip()
            if stripped and not stripped.startswith("%"):
                dimensions = tuple(int(item) for item in stripped.split())
                if len(dimensions) != 3:
                    raise ValueError(f"expected a coordinate matrix: {path}")
                return dimensions + (header[4],)
    raise ValueError(f"missing Matrix Market dimensions: {path}")


def _cached_rcm_path(path: Path) -> Path:
    if path.suffix.lower() != ".mtx":
        raise ValueError(f"expected a .mtx input: {path}")
    return path.with_name(path.stem + "_rcm.mtx")


def _generate_rcm(source: Path, destination: Path) -> None:
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    matrix = load_csr(source)
    permutation = reverse_cuthill_mckee(matrix, symmetric_mode=False)
    reordered = matrix[permutation, :][:, permutation].tocsr()
    reordered.sum_duplicates()
    reordered.eliminate_zeros()
    reordered.sort_indices()

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.stem + ".", suffix=".mtx", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        # Preserve explicit entries even when a PyG graph happens to be
        # structurally symmetric, so cache metadata remains comparable to the
        # original general Matrix Market file.
        scipy.io.mmwrite(temporary, reordered, field="real", symmetry="general")
        try:
            # Keep an RCM file produced concurrently instead of replacing it.
            os.link(temporary, destination)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def resolve_rcm_dataset(
    name: str, allow_download: bool = True
) -> DatasetLocation:
    """Resolve or create A[p, :][:, p] using reverse Cuthill-McKee."""
    original = resolve_dataset(name, allow_download)
    expected = _mtx_metadata(original.path)
    candidates = [_cached_rcm_path(original.path), *LOCAL_RCM_MTX[original.name]]
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        if _mtx_metadata(candidate) != expected:
            raise ValueError(
                f"RCM cache does not match {original.path}: {candidate}"
            )
        return DatasetLocation(original.name, candidate, "rcm-cache", "rcm")

    destination = _cached_rcm_path(original.path)
    print(f"Generating RCM matrix: {destination}")
    _generate_rcm(original.path, destination)
    if _mtx_metadata(destination)[:2] != expected[:2]:
        raise RuntimeError(f"generated RCM matrix has the wrong shape: {destination}")
    return DatasetLocation(
        original.name, destination.resolve(), "rcm-generated", "rcm"
    )


def resolve_rcm_datasets(names, allow_download: bool = True):
    return [resolve_rcm_dataset(name, allow_download) for name in names]


def load_csr(path: Path):
    """Load a square Matrix Market graph as canonical float32 CSR."""
    matrix = scipy.io.mmread(path)
    if not scipy.sparse.issparse(matrix):
        matrix = scipy.sparse.coo_matrix(matrix)
    matrix = matrix.astype(np.float32).tocsr()
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square graph matrix, got {matrix.shape}: {path}")
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix
