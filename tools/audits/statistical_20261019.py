#!/usr/bin/env python
"""CPU validation declared in docs/STATISTICAL_AUDIT_20261019.md.

Run: OMP_NUM_THREADS=2 .venv/bin/python tools/audits/statistical_20261019.py
STOP: create artifacts/statistical_audit_20261019/STOP; remove it and rerun to resume.
Historical experiment files are read only. No new training criterion is selected.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
from scipy.stats import binomtest, norm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'artifacts/statistical_audit_20261019'
MASTER, REPS, BATCH = 2026101902, 20000, 500
DISTRIBUTIONS = ('normal', 'normal_corr_0.8', 'student_t3', 'lognormal_sd1.5')


def standardized(delta, sd):
    out = np.divide(delta, sd, out=np.zeros_like(delta), where=sd > 0)
    return np.where((sd == 0) & (delta != 0), np.sign(delta)*np.inf, out)


def leave_one_out_z(x):
    """x: (..., rows, endpoints); each row compared with all remaining rows."""
    n = x.shape[-2]
    if n < 3:
        raise ValueError('At least three rows required')
    delta = x-x.mean(axis=-2, keepdims=True)
    ss = (delta**2).sum(axis=-2, keepdims=True)
    variance = np.maximum(0., (ss - n/(n-1)*delta**2)/(n-2))
    return standardized(n/(n-1)*delta, np.sqrt(variance))


def historical_p(x):
    """Observed row first. Matches the historical max-|z|, without rounding."""
    null = x[...,1:,:]
    zc = standardized(x[...,0,:]-null.mean(axis=-2), null.std(axis=-2,ddof=1))
    zn = leave_one_out_z(null)
    maxima = abs(zn).max(axis=-1)
    return (1+(maxima[..., :,None] >= abs(zc)[...,None,:]).sum(axis=-2))/x.shape[-2]


def symmetric_p(x):
    """Observed row first, symmetric studentization; raw and adjusted absolute-z p."""
    z = abs(leave_one_out_z(x))
    raw = (z >= z[...,0:1,:]).mean(axis=-2)
    adjusted = (z.max(axis=-1)[..., :,None] >= z[...,0:1,:]).mean(axis=-2)
    return raw, adjusted


def draw(rng, distribution, count, k, m):
    shape = (count,k+1,m)
    if distribution == 'normal':
        return rng.normal(size=shape)
    if distribution == 'normal_corr_0.8':
        return np.sqrt(.8)*rng.normal(size=(count,k+1,1)) + np.sqrt(.2)*rng.normal(size=shape)
    if distribution == 'student_t3':
        return rng.standard_t(3,size=shape)
    if distribution == 'lognormal_sd1.5':
        return rng.lognormal(sigma=1.5,size=shape)
    raise ValueError(distribution)


def interval(count,n):
    ci = binomtest(count,n).proportion_ci(confidence_level=.95,method='exact')
    p = binomtest(count,n,p=.05,alternative='greater').pvalue
    return dict(rejections=count,n=n,rate=count/n,ci95=[ci.low,ci.high],
                p_excess_over_005=p,flag_excess=bool(p < .01/32))


def references():
    spec = importlib.util.spec_from_file_location('legacy',ROOT/'scripts/diagnostics/study_s_reading.py')
    legacy = importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    def fixture(row):
        return {'summary': {'true/where': {'E1':row[0]},'true/loom':{'E1':row[1]}}}
    x=np.array([[10.,20.],[1.,2.],[2.,4.],[3.,6.],[4.,8.]])
    f=legacy.family(fixture(x[0]),[fixture(v) for v in x[1:]],legacy.PRIMARY)
    assert all(v['p_raw_two_sided']==.4 and v['p_adjusted_maxz']==.2 for v in f.values())
    assert np.all(historical_p(x)==.2)
    rng=np.random.default_rng(MASTER)
    random_fixture=rng.normal(size=(41,2))
    f=legacy.family(fixture(random_fixture[0]),[fixture(v) for v in random_fixture[1:]],legacy.PRIMARY)
    assert np.allclose(historical_p(random_fixture),[v['p_adjusted_maxz'] for v in f.values()],atol=5.1e-5,rtol=0)
    permutation=rng.permutation(41)
    assert np.allclose(leave_one_out_z(random_fixture)[permutation],leave_one_out_z(random_fixture[permutation]))
    raw,adj=symmetric_p(random_fixture);assert np.all(adj>=raw)
    for method in (historical_p,lambda v:symmetric_p(v)[1]):
        assert np.all(method(np.ones((41,8)))==1)
    z=rng.normal(size=(REPS,8))
    invalid=(abs(z)>norm.ppf(.975)).any(axis=1).sum()
    assert invalid/REPS > .10
    x=rng.normal(size=(2000,41,2));x[:,0,:]+=8
    positive={name:float((fn(x)<.05).any(axis=-1).mean()) for name,fn in (
        ('historical',historical_p),('symmetric',lambda v:symmetric_p(v)[1]))}
    assert min(positive.values())>=.95
    return dict(hand_fixture='passed',historical_implementation_agreement='passed',
                permutation_and_ties='passed',unadjusted_negative_control_rate=float(invalid/REPS),
                positive_detection_rate=positive)


def historical_sensitivity():
    data=ROOT/'artifacts/runs/study_s'
    read=lambda name:json.loads((data/f'{name}.json').read_text())
    conn=read('connectome')
    families={'primary':[('E1',t) for t in ('where','loom')],
              'secondary':[(e,t) for t in ('where','loom') for e in ('A_test','A_gen','E4','E5')]}
    result={}
    for kc in (20,40):
        result[str(kc)]={}
        saved=read(f'reading-K{kc}c')
        for rung,prefix,k in [('b','nulllat',40),('c','nullin',kc)]:
            graphs=[conn]+[read(f'{prefix}{i}') for i in range(k)]
            for family,endpoints in families.items():
                x=np.array([[g['summary'][f'true/{t}'][e] for e,t in endpoints] for g in graphs])
                raw,adj=symmetric_p(x)
                result[str(kc)][f'{rung}/{family}']={}
                for j,(e,t) in enumerate(endpoints):
                    old=saved[f'rung_{rung}'][family][f'{e}/{t}']
                    result[str(kc)][f'{rung}/{family}'][f'{e}/{t}']=dict(
                        historical_adjusted=old['p_adjusted_maxz'],symmetric_raw=float(raw[j]),
                        symmetric_adjusted=float(adj[j]),direction=old['direction'])
        result[str(kc)]['two_family_sensitivity']={}
        for family,endpoints in families.items():
            for e,t in endpoints:
                rows=[result[str(kc)][f'{rung}/{family}'][f'{e}/{t}'] for rung in ('b','c')]
                sig=all(row['symmetric_adjusted']<.05 for row in rows) and rows[0]['direction']==rows[1]['direction']
                result[str(kc)]['two_family_sensitivity'][f'{e}/{t}']=dict(
                    meets_rule=bool(sig),direction=rows[0]['direction'])
    return result


def save(result):
    tmp=OUT/'calibration.json.part'
    tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(OUT/'calibration.json')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    signatures={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [Path(__file__).resolve(),ROOT/'docs/STATISTICAL_AUDIT_20261019.md']}
    p=OUT/'calibration.json'
    result=json.loads(p.read_text()) if p.exists() else dict(complete=False,design_sha256=signatures,cells={})
    if result['design_sha256']!=signatures:
        raise ValueError('Design or script changed; preserve the earlier run and use a separate output')
    result['references']=references();save(result)
    print('Fixed references passed',flush=True)
    for di,distribution in enumerate(DISTRIBUTIONS):
        for k in (20,40):
            for m in (2,8):
                key=f'{distribution}/K{k}/m{m}'
                if key in result['cells']:continue
                rng=np.random.default_rng(np.random.SeedSequence([MASTER,di,k,m]))
                counts={'historical':0,'symmetric':0}
                for start in range(0,REPS,BATCH):
                    if (OUT/'STOP').exists():save(result);print('STOP: unfinished cell will restart');return
                    x=draw(rng,distribution,min(BATCH,REPS-start),k,m)
                    for name,pvals in [('historical',historical_p(x)),('symmetric',symmetric_p(x)[1])]:
                        counts[name]+=int((pvals<.05).any(axis=-1).sum())
                result['cells'][key]={name:interval(count,REPS) for name,count in counts.items()}
                save(result);print(key,{name:round(count/REPS,4) for name,count in counts.items()},flush=True)
    result['historical_sensitivity_posthoc']=historical_sensitivity()
    result['complete']=True;save(result)
    print('Complete; historical data files unchanged',flush=True)


if __name__=='__main__':main()
