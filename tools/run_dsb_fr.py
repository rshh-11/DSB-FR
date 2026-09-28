from pathlib import Path
import argparse,csv,hashlib,json,random,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from multilayer_subspace_pilot import (MultiLayerExtractor,parse_blocks,set_seed,train_paths,test_paths,reference_images,collect_train_features,fit_pca,pca_energy,robust_log_z,image_foreground_mask,image_score,image_score_masked,post_process_map)
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score,average_precision_score
from huggingface_hub import snapshot_download

def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description='Paper-only DSB-FR runner using frozen numerical primitives.')
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--categories',nargs='+',required=True)
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/paper.json')
    p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--device',default='cuda')
    p.add_argument('--model-path',type=Path,help='Local frozen snapshot; must match the documented revision.')
    p.add_argument('--offline',action='store_true')
    p.add_argument('--debug-limit',type=int,help='Per good/bad limit for smoke testing, not benchmark results.')
    p.add_argument('--save-maps',action='store_true')
    a=p.parse_args();cfg=json.loads(a.config.read_text(encoding='utf-8-sig'));seeds=a.seeds or cfg['seeds']
    for cat in a.categories:
        if not train_paths(a.dataset,cat):p.error('Missing train/good images for '+cat)
    a.outdir.mkdir(parents=True,exist_ok=True)
    checkpoint=str(a.model_path) if a.model_path else snapshot_download(repo_id=cfg['model_ckpt'],revision=cfg['model_revision'],local_files_only=a.offline)
    set_seed(seeds[0]);ex=MultiLayerExtractor(checkpoint,cfg['image_res'],parse_blocks(cfg['blocks']),a.device)
    if a.device=='cpu':
        import torch
        import subspacead.core.pca as original_pca
        original_pca.device=torch.device('cpu')
    provenance={'config':cfg,'seeds':seeds,'categories':a.categories,'smoke_test':a.debug_limit is not None,'device':a.device,'local_model_override':a.model_path is not None,'supports':[]}
    rows=[];metrics=[]
    for seed in seeds:
      for cat in a.categories:
        set_seed(seed);paths=train_paths(a.dataset,cat);chosen=paths.copy();random.Random(seed).shuffle(chosen);chosen=chosen[:cfg['k_shot']]
        provenance['supports'].extend({'category':cat,'seed':seed,'path':x.relative_to(a.dataset).as_posix(),'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in chosen)
        refs=reference_images(paths,cat,seed,cfg['k_shot'],cfg['aug_count'],cfg['image_res'])
        feats=collect_train_features(ex,refs,cfg['batch_size']);banks={n:fit_pca(v,cfg['global_pca_ev']) for n,v in feats.items()};energies={n:pca_energy(feats[n],banks[n]) for n in banks}
        selected=test_paths(a.dataset,cat,a.debug_limit);current=[];maps=[]
        for start in range(0,len(selected),cfg['batch_size']):
          batch=selected[start:start+cfg['batch_size']];images=[Image.open(x).convert('RGB') for x in batch];features=ex.extract(images);grid=ex.grid_size
          zs=[robust_log_z(pca_energy(features[n].reshape(-1,features[n].shape[-1]),banks[n]),energies[n]).reshape(len(batch),-1) for n in banks]
          fused=np.mean(np.stack(zs),axis=0).astype(np.float32)
          for i,(path,img) in enumerate(zip(batch,images)):
            mask=image_foreground_mask(img,grid,cfg['image_res']);full=image_score(fused[i],grid,cfg['image_res'],cfg['top_frac']);roi=image_score_masked(fused[i],mask,grid,cfg['image_res'],cfg['top_frac'])
            row={'seed':seed,'category':cat,'path':path.relative_to(a.dataset).as_posix(),'label':int(path.parent.name!='good'),'s_full':full,'s_roi':roi,'s_dsb_fr':0.5*(full+roi)};rows.append(row);current.append(row)
            if a.save_maps:maps.append(post_process_map(fused[i].reshape(grid),cfg['image_res']))
        labels=[x['label'] for x in current]
        for method,column in [('dsb_fusion_full','s_full'),('dsb_fusion_roi','s_roi'),('dsb_fr','s_dsb_fr')]:
          scores=[x[column] for x in current];metrics.append({'seed':seed,'category':cat,'method':method,'image_auroc':roc_auc_score(labels,scores) if len(set(labels))==2 else '', 'image_average_precision':average_precision_score(labels,scores) if 1 in labels else '', 'n_test':len(labels)})
        if a.save_maps:np.savez_compressed(a.outdir/f'{cat}_seed{seed}_maps.npz',maps=np.stack(maps),paths=np.asarray([x['path'] for x in current]))
        write_csv(a.outdir/'image_scores.csv',rows);write_csv(a.outdir/'metrics_by_seed_category.csv',metrics);(a.outdir/'run_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
        print(cat,seed,'DSB-FR AUROC',metrics[-1]['image_auroc'],flush=True)
if __name__=='__main__':main()
