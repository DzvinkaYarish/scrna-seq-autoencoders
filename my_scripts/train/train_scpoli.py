import pandas as pd
import torch
import os
import scanpy as sc
from scarches.models.scpoli import scPoli
from scarches.dataset.trvae.data_handling import remove_sparsity
from sklearn.metrics import classification_report
import wandb

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
    wandb.log({prefix + '_' + k: v for k, v in metrics_to_log.items()})

    preds_df = pd.DataFrame({'cell_type_pred': results_dict[cell_type_key[0]]['preds'],
                             'cell_type_gt': adata.obs[cell_type_key[0]]
                             })
    preds_df.to_csv(os.path.join(config['exp_path'], f'{prefix}_preds.csv'), index=False)



ROOT = '/gpfs/space/home/dzvenymy/scarches/'
EXP_NAME = 'onek1k_and_randolph_train_ignore_r_cell_types'


config = dict (
    exp_path=os.path.join(ROOT, 'experiments/', EXP_NAME),
    data_path=os.path.join(ROOT, 'data/OneK1K_Randolph/train_half_merged_samples_with_cell_types_hvg_from_onek.h5ad'),
    query_data_path=[
        os.path.join(ROOT, 'data/OneK1K/val_half_merged_samples_with_cell_types_hvg.h5ad'),
                     os.path.join(ROOT, 'data/Randolph/train_half_merged_samples_with_cell_types_hvg_from_onek.h5ad'),
                     os.path.join(ROOT, 'data/Perez/val_half_merged_samples_with_cell_types_hvg_from_onek.h5ad')
                     ],
    embedding_dim=20,
    latent_dim=30,
    hidden_layer_sizes=[128, 128, 128],
    dr_rate=0.1,
    epochs=60,
    pretraining_epochs=40,
    query_epochs=40,
    query_pretraining_epochs=30,
    batch_size=128,
    eta=5,
)

run = wandb.init(
  project="scarches",
  name=EXP_NAME,
  notes="",
  tags=[],
  config=config,
)

condition_key = 'sample'
cell_type_key = ['cell_type']

print('Loading data...')

adata_onek1k = sc.read(config['data_path'])

wandb.log({'train_size': len(adata_onek1k)})

adata_onek1k = remove_sparsity(adata_onek1k)

obs = adata_onek1k.obs.reset_index(drop=True)
labeled_indx = obs[obs['batch'] == '1'].index
labeled_indx = list(labeled_indx)

scpoli_model = scPoli(
    adata=adata_onek1k,
    condition_key=condition_key,
    cell_type_keys=cell_type_key,
    embedding_dim=config['embedding_dim'],
    latent_dim=config['latent_dim'],
    hidden_layer_sizes=config['hidden_layer_sizes'],
    dr_rate=config['dr_rate'],
    use_ln=True,
    labeled_indices=labeled_indx,
    # unknown_ct_names=['CD4_T', 'CD8_T', 'NK', 'monocytes',
    #                   'B', 'highly_infected', 'NKT',
    #                   'infected_monocytes', 'NK_high_response', 'neutrophils', 'DC']
)


early_stopping_kwargs = {
    "early_stopping_metric": "val_prototype_loss",
    "mode": "min",
    "threshold": 0,
    "patience": 20,
    "reduce_lr": True,
    "lr_patience": 13,
    "lr_factor": 0.1,
}

print('Training...')
scpoli_model.train(
    n_epochs=config['epochs'],
    pretraining_epochs=config['pretraining_epochs'],
    early_stopping_kwargs=early_stopping_kwargs,
    eta=config['eta'], # weight for prototype loss
    batch_size=config['batch_size'], # default is 128,
    clustering='kmeans', # 'kmeans' or 'louvain
    n_clusters=11,
)


scpoli_model.save(os.path.join(config['exp_path'], 'ref_model/'))

get_classification_report(adata_onek1k, scpoli_model, cell_type_key, 'train', config)

sample_adata_onek1k = sample_adata(adata_onek1k, 0.001)

del adata_onek1k

print('Logging to w&b...')
logs = scpoli_model.trainer.logs
for k,v in logs.items():
    logs[k] = [None] * (config['epochs'] - len(logs[k])) + logs[k]

for i in range(config['epochs']):
    wandb.log({k: v[i] for k, v in logs.items()})



# Add query data
scpoli_model = scPoli.load(os.path.join(config['exp_path'], 'ref_model/'), sample_adata_onek1k)

for i, query_data_path in enumerate(config['query_data_path']):
    adata_query = sc.read(query_data_path)
    wandb.log({f'query_size_{i}': len(adata_query)})

    adata_query = remove_sparsity(adata_query)

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


    logs = scpoli_query.trainer.logs
    for k,v in logs.items():
        logs[k] = [None] * (config['epochs'] - len(logs[k])) + logs[k]

    for k in range(config['epochs']):
        wandb.log({f'query_{i}_' + k: v[i] for k, v in logs.items()})

    scpoli_model = scpoli_query

scpoli_query.save(os.path.join(config['exp_path'], 'query_model/'))


# get_classification_report(adata_query, scpoli_query, cell_type_key, 'query', config)



