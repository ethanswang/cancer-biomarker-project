import scanpy as sc

sc.settings.verbosity = 3
adata = sc.read_mtx("/Users/zhiyong/Downloads/synopsis/Bmatrix.mtx")

print("Original matrix shape:", adata.shape)

sc.pp.calculate_qc_metrics(adata, inplace=True)
low_counts = adata.obs['n_genes_by_counts'] < 200
high_counts = adata.obs['total_counts'] > 1e4
adata = adata[~(low_counts | high_counts)]

print("Final matrix shape:", adata.shape)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.neighbors(adata)
sc.tl.umap(adata, min_dist = 0.1)
sc.pl.umap(adata)