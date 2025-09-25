"""
Person 4 - cuGraph
Nvidia Hackathon 2025

Convert CSR → cuGraph edgelist
Run Leiden/Louvain clustering
Map clusters back to obs_names
QC cluster outputs (#clusters, determinism check)
Unit tests: cluster IDs align to cells
"""

# ---------- Convert CSR → cuGraph edgelist ----------
import cugraph
import cudf
from scipy.sparse import csr_matrix

def csr_to_cugraph(csr_mat):
    coo = csr_mat.tocoo()
    edges = cudf.DataFrame({
        "src": coo.row,
        "dst": coo.col,
        "weight": coo.data
    })
    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(edges, source='src', destination='dst',
                         edge_attr='weight', renumber=False)
    return G

# ---------- Run Leiden/Louvain clustering ----------
def run_gpu_clustering(G, method="leiden", resolution=1.0, random_state=0):
    """
    Run community detection on GPU WNN graph
    """
    import cugraph.community as community
    if method.lower() == "leiden":
        cluster_labels = community.leiden(G, resolution=resolution,
                                          random_state=random_state)
    elif method.lower() == "louvain":
        cluster_labels = community.louvain(G)
    else:
        raise ValueError("Method must be 'leiden' or 'louvain'")
    return cluster_labels

# ---------- Map clusters back to obs_names ----------
def map_clusters_to_obs(adata, cluster_df, obs_key="wnn_leiden"):
    """
    Save cluster assignments into adata.obs
    """
    cluster_pd = cluster_df.to_pandas().set_index("vertex")
    adata.obs[obs_key] = cluster_pd['partition'].astype(str)

# ---------- QC cluster outputs (#clusters, determinism check) ----------
def qc_clusters(adata, obs_key="wnn_leiden"):
    n_clusters = adata.obs[obs_key].nunique()
    print(f"Number of clusters: {n_clusters}")
    print("Cluster sizes:")
    print(adata.obs[obs_key].value_counts())

# ---------- Unit tests: cluster IDs align to cells ----------
def test_clusters(adata, cluster_df):
    assert len(cluster_df) == adata.n_obs, "Cluster labels do not match number of cells"


# ===== Person 4: WNN Clustering (Leiden/Louvain) =====
import cudf
import cugraph
from scipy.sparse import csr_matrix

# --- 1 Convert CSR → cuGraph Graph ---
def csr_to_cugraph(csr_mat):
    coo = csr_mat.tocoo()
    edges = cudf.DataFrame({
        "src": coo.row,
        "dst": coo.col,
        "weight": coo.data
    })
    G = cugraph.Graph()
    G.from_cudf_edgelist(edges, source='src', destination='dst',
                         edge_attr='weight', renumber=False)
    return G

# --- 2. Run GPU Clustering ---
def run_gpu_clustering(G, method="leiden", resolution=1.0, random_state=0):
    import cugraph.community as community
    if method.lower() == "leiden":
        cluster_labels = community.leiden(G, resolution=resolution,
                                          random_state=random_state)
    elif method.lower() == "louvain":
        cluster_labels = community.louvain(G)
    else:
        raise ValueError("Method must be 'leiden' or 'louvain'")
    return cluster_labels

# --- 3. Map clusters back to adata.obs ---
def map_clusters_to_obs(adata, cluster_df, obs_key="wnn_leiden"):
    cluster_pd = cluster_df.to_pandas().set_index("vertex")
    adata.obs[obs_key] = cluster_pd['partition'].astype(str)

# --- 4 QC cluster outputs ---
def qc_clusters(adata, obs_key="wnn_leiden"):
    n_clusters = adata.obs[obs_key].nunique()
    print(f"Number of clusters: {n_clusters}")
    print("Cluster sizes:")
    print(adata.obs[obs_key].value_counts())

# --- 5 Run everything ---
# Make sure your fused WNN CSR matrix is stored in adata.obsp["wnn_connectivities"]
G = csr_to_cugraph(adata.obsp["wnn_connectivities"])
clusters = run_gpu_clustering(G, method="leiden", resolution=1.0, random_state=0)
map_clusters_to_obs(adata, clusters, obs_key="wnn_leiden")
qc_clusters(adata, obs_key="wnn_leiden")
test_clusters(adata, clusters)