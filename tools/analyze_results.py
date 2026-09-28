from pathlib import Path
from collections import defaultdict
import argparse,csv,json,random
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[1]
def read(name):
    with (ROOT/'results'/name).open(encoding='utf-8-sig',newline='') as h:return list(csv.DictReader(h))
def main():
    p=argparse.ArgumentParser();p.add_argument('--bootstrap',type=int,default=0);p.add_argument('--output',type=Path,default=ROOT/'outputs/result_verification.json');a=p.parse_args()
    pairs=read('paired_image_auroc.csv');groups=defaultdict(list)
    for row in read('image_scores.csv'):groups[(row['dataset'],row['category'],int(row['seed']))].append(row)
    expected={(x['dataset'],x['category'],int(x['seed'])):float(x['dsb_fr']) for x in pairs}
    if set(groups)!=set(expected):raise ValueError('Category/seed coverage differs from frozen paired table.')
    metrics=[];max_err=0.0
    for key,rows in groups.items():
        y=[int(x['label']) for x in rows]
        for col in ['dsb_fusion_full','dsb_fusion_roi','dsb_fr']:
            scores=[float(x[col]) for x in rows];auc=roc_auc_score(y,scores);ap=average_precision_score(y,scores)
            metrics.append(dict(dataset=key[0],category=key[1],seed=key[2],method=col,image_auroc=auc,image_ap=ap))
            if col=='dsb_fr':max_err=max(max_err,abs(auc-expected[key]))
    if max_err>1e-6:raise ValueError('Saved scores disagree with frozen results: '+str(max_err))
    datasets=['MVTec-AD','BTAD','VisA','MPDD'];summary={}
    for ds in datasets:
        summary[ds]={method:float(np.mean([x['image_auroc'] for x in metrics if x['dataset']==ds and x['method']==method])) for method in ['dsb_fusion_full','dsb_fusion_roi','dsb_fr']}
        summary[ds]['subspacead']=float(np.mean([float(x['subspacead']) for x in pairs if x['dataset']==ds]))
    summary['Equal dataset mean']={k:float(np.mean([summary[d][k] for d in datasets])) for k in summary[datasets[0]]}
    mean=summary['Equal dataset mean'];result={'pairs':len(pairs),'query_seed_rows':sum(map(len,groups.values())),'max_auc_error_vs_saved':max_err,'dataset_means':summary,'pipeline_gain_pp':100*(mean['dsb_fr']-mean['subspacead']),'fr_gain_pp':100*(mean['dsb_fr']-mean['dsb_fusion_full'])}
    rank=read('rank_matched_summary.csv');result['rank_matched_gain_pp']=100*float(np.mean([float(x['dual'])-float(x['joint_rank_matched']) for x in rank]))
    if a.bootstrap:
        grouped={d:defaultdict(list) for d in datasets}
        for x in pairs:grouped[x['dataset']][x['category']].append(float(x['dsb_fr'])-float(x['subspacead']))
        rng=random.Random(20260902);draws=[]
        for _ in range(a.bootstrap):
            per_dataset=[]
            for d in datasets:
                cats=list(grouped[d]);vals=[]
                for _ in cats:
                    v=grouped[d][rng.choice(cats)];vals.append(sum(rng.choice(v) for _ in range(3))/3)
                per_dataset.append(sum(vals)/len(vals))
            draws.append(sum(per_dataset)/4)
        result['bootstrap']={'n':a.bootstrap,'seed':20260902,'ci95_pp':(100*np.quantile(draws,[.025,.975])).tolist()}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
