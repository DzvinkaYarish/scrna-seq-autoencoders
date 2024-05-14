import scanpy as sc
import os

ROOT = '/gpfs/space/home/dzvenymy/scarches/'
train_a = sc.read(os.path.join(ROOT, 'data/OneK1K/train_merged_samples_with_cell_types.h5ad'))
train_a_hvg = sc.pp.highly_variable_genes(train_a, flavor='seurat_v3',  n_top_genes=10000, inplace=False)

train_a_hvg.to_csv(os.path.join(ROOT, 'data/OneK1K/train_hvg.csv'))

