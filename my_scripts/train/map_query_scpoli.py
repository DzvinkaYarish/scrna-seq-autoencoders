import pandas as pd
import torch
import os
import scanpy as sc
from scarches.models.scpoli import scPoli
from scarches.dataset.trvae.data_handling import remove_sparsity
from sklearn.metrics import classification_report
import json

# from train_scpoli import sample_adata, get_classification_report

import warnings
warnings.filterwarnings('ignore')


def sample_adata(adata, frac=0.3, rs=3):
    indx = adata.obs.reset_index(drop=True).sample(frac=frac, random_state=rs)
    obs = adata.obs.iloc[list(indx.index)]
    x = adata.X[list(indx.index)]
    adata_frac = sc.AnnData(X=x, var=adata.var, obs=obs)
    return adata_frac


def get_classification_report(adata, model, cell_type_key, prefix, config):
    results_dict = model.classify(
        adata.X,
        adata.obs[condition_key],
    )

    clf = classification_report(
        y_true=adata.obs[cell_type_key[0]],
        y_pred=results_dict[cell_type_key[0]]['preds'],
        output_dict=True,
    )

    metrics_to_log = {**{'accuracy': clf['accuracy']}, **clf['macro avg']}
    # wandb.log({prefix + '_' + k: v for k, v in metrics_to_log.items()})

    preds_df = pd.DataFrame({'cell_type_pred': results_dict[cell_type_key[0]]['preds'],
                             'cell_type_gt': adata.obs[cell_type_key[0]]
                             })
    preds_df.to_csv(os.path.join(config['exp_path'], f'{prefix}_preds.csv'), index=False)

ROOT = '/gpfs/space/home/dzvenymy/scarches/'
EXP_NAME = 'map_to_Onek1k_and_Randolp_imerged_cell_types_val_samples_from_Onek1k_'

config = dict (
    exp_path=os.path.join(ROOT, 'experiments/', EXP_NAME),
    data_path=os.path.join(ROOT, 'data/OneK1K_Randolph/train_half_merged_samples_with_cell_types_hvg_from_onek_merged_cell_types.h5ad'),
    query_data_path=os.path.join(ROOT, 'data/OneK1K/val_half_merged_samples_with_cell_types_hvg.h5ad'),
    ref_model_path=os.path.join(ROOT, 'experiments/onek1k_and_randolph_train_merged_cell_types','query_model/'),
    embedding_dim=20,
    latent_dim=30,
    hidden_layer_sizes=[128, 128, 128],
    dr_rate=0.1,
    epochs=80,
    pretraining_epochs=40,
    query_epochs=40,
    query_pretraining_epochs=30,
    batch_size=128,
    eta=10,
    labeled_query_data=False
)

print(config['exp_path'])

if not os.path.exists(config['exp_path']):
    os.makedirs(config['exp_path'])

with open(os.path.join(config['exp_path'], 'config.json'), 'w') as f:
    json.dump(config, f)


condition_key = 'sample'
cell_type_key = ['cell_type']

# Add query data
adata_onek1k = sc.read(config['data_path'])

sample_adata_onek1k = sample_adata(adata_onek1k, 0.001)

del adata_onek1k

sample_adata_onek1k = remove_sparsity(sample_adata_onek1k)

adata_query = sc.read(config['query_data_path'])

adata_query = remove_sparsity(adata_query)

scpoli_model = scPoli.load(config['ref_model_path'], sample_adata_onek1k)


scpoli_query = scPoli.load_query_data(
    adata=adata_query,
    reference_model=scpoli_model,
    labeled_indices=[],
)

scpoli_query.train(
    n_epochs=config['query_epochs'],
    pretraining_epochs=config['query_pretraining_epochs'],
    eta=5
)

scpoli_query.save(os.path.join(config['exp_path'], 'query_model/'))


if config['labeled_query_data']:
    get_classification_report(adata_query, scpoli_query, cell_type_key, 'query', config)