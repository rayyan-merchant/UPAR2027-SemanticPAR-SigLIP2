"""Generate aggregate frozen-DINO report drafts; final prose is reviewed in Git."""
import csv,json,hashlib,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R/'reports/dino_v3/results';A=R.parent/'dino/v3-frozen-v2-results';S=A/'dinov3_results'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.write_text(x,encoding='utf-8')
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n'
def f(x):return f'{x:.6f}'
summary=read(P/'SUMMARY.json');totals=summary['settings'];metrics=summary['fold_metrics'];curves=read(P/'SOURCE_CURVES_SUPPORT.json');folds=['PA100k','PETA','Market1501']
def m(d,k):return next(x for x in metrics if x['domain']==d and x['setting']==k)
compare=table(['Held-out domain','SigLIP calibrated','DINO calibrated','75/25 calibrated','Ensemble delta'],[[d,f(m(d,'siglip_calibrated')['Challenge_Avg']),f(m(d,'dino_calibrated')['Challenge_Avg']),f(m(d,'ensemble_calibrated')['Challenge_Avg']),f(m(d,'ensemble_calibrated')['Challenge_Avg']-m(d,'siglip_calibrated')['Challenge_Avg'])] for d in folds]+[['Macro',f(totals['siglip_calibrated']['Challenge_Avg']),f(totals['dino_calibrated']['Challenge_Avg']),f(totals['ensemble_calibrated']['Challenge_Avg']),f(totals['ensemble_calibrated']['Challenge_Avg']-totals['siglip_calibrated']['Challenge_Avg'])],['Worst',f(totals['siglip_calibrated']['worst_Challenge_Avg']),f(totals['dino_calibrated']['worst_Challenge_Avg']),f(totals['ensemble_calibrated']['worst_Challenge_Avg']),f(totals['ensemble_calibrated']['worst_Challenge_Avg']-totals['siglip_calibrated']['worst_Challenge_Avg'])]])
rawcal=table(['Setting','PA100K','PETA','Market1501','Macro','Worst'],[[k,*[f(m(d,k)['Challenge_Avg']) for d in folds],f(totals[k]['Challenge_Avg']),f(totals[k]['worst_Challenge_Avg'])] for k in totals])
metricstable=table(['Domain','Setting','mA','Instance-F1','Precision','Recall','Label-F1'],[[d,k,*[f(m(d,k)[q]) for q in ['mA','Inst_F1','Inst_Prec','Inst_Rec','Label_F1']]] for d in folds for k in ['siglip_calibrated','dino_calibrated','ensemble_calibrated']])
source=table(['Held out','Selected epoch / run epochs','DINO source raw','DINO source calibrated','DINO target calibrated','Source-target gap','Ensemble source calibrated','Ensemble source-target gap'],[[d,f"{curves[d]['best_epoch']} / {curves[d]['epochs']}",f(curves[d]['source_selection_raw']['Challenge_Avg']),f(curves[d]['source_selection_calibrated']['Challenge_Avg']),f(m(d,'dino_calibrated')['Challenge_Avg']),f(curves[d]['dino_source_target_calibrated_gap']),f(curves[d]['ensemble_source_selection_calibrated']['Challenge_Avg']),f(curves[d]['ensemble_source_target_calibrated_gap'])] for d in folds])
convergence=table(['Held out','Source train rows','Images consumed','Attempted / successful updates','AMP skips','Head wall seconds','Final unique source images'],[[d,str(sum(curves[d]['source_counts'].values())),str(curves[d]['images_consumed']),f"{curves[d]['attempted_updates']} / {curves[d]['successful_updates']}",curves[d]['amp_skips'],f"{curves[d]['elapsed_seconds']:.2f}",str(sum(curves[d]['unique_seen_final'].values()))] for d in folds])
calibration=table(['Held out','DINO raw','DINO calibrated','Transfer effect','Ensemble raw','Ensemble calibrated','Transfer effect'],[[d,f(m(d,'dino_raw')['Challenge_Avg']),f(m(d,'dino_calibrated')['Challenge_Avg']),f(m(d,'dino_calibrated')['Challenge_Avg']-m(d,'dino_raw')['Challenge_Avg']),f(m(d,'ensemble_raw')['Challenge_Avg']),f(m(d,'ensemble_calibrated')['Challenge_Avg']),f(m(d,'ensemble_calibrated')['Challenge_Avg']-m(d,'ensemble_raw')['Challenge_Avg'])] for d in folds])
correlation=table(['Held out','Probability-residual correlation','Binary-error correlation at0.5','Decision disagreement','Double fault'],[[d,*[f(curves[d]['error_complementarity'][k]) for k in ['raw_probability_residual_correlation','binary_error_correlation_at_half','raw_decision_disagreement','double_fault_at_half']]] for d in folds])
groups=list(csv.DictReader((P/'GROUP_METRICS.csv').open(encoding='utf-8')))
def g(d,s,n,k):return float(next(x for x in groups if x['domain']==d and x['setting']==s and x['group']==n)[k])
groupdelta=table(['Group','PA mA delta','PETA mA delta','Market mA delta','PA label-F1 delta','PETA label-F1 delta','Market label-F1 delta'],[[n,*[f(g(d,'ensemble_calibrated',n,k)-g(d,'siglip_calibrated',n,k)) for k in ['mA','label_F1'] for d in folds]] for n in ['age','upper_colors','lower_colors','accessories']])
sensitivity=table(['Setting','Mean-image-F1 Challenge macro','Mean-image-F1 worst','All-valid safe-zero mA + mean-image-F1 macro'],[[k,f(totals[k]['mean_image_Challenge_Avg']),f(totals[k]['worst_mean_image_Challenge_Avg']),f(totals[k]['all_valid_safe_zero_mean_image_Challenge_Avg'])] for k in ['siglip_calibrated','dino_calibrated','ensemble_calibrated']])
attrs=list(csv.DictReader((P/'ATTRIBUTE_METRICS.csv').open(encoding='utf-8')));lookup={(x['domain'],x['setting'],x['attribute']):x for x in attrs}
changes=[]
for d in folds:
 for name in __import__('upar.attributes',fromlist=['ATTRIBUTES']).ATTRIBUTES:
  old=lookup[d,'siglip_calibrated',name];new=lookup[d,'ensemble_calibrated',name]
  if old['mA'] and new['mA']:changes.append({'domain':d,'attribute':name,'mA_delta':float(new['mA'])-float(old['mA']),'label_F1_delta':float(new['label_F1'])-float(old['label_F1']),'target_positives':int(new['positives']),'source_positives':curves[d]['source_positive_counts'][__import__('upar.attributes',fromlist=['ATTRIBUTES']).ATTRIBUTES.index(name)]})
worst=sorted(changes,key=lambda x:x['mA_delta'])[:8];best=sorted(changes,key=lambda x:x['mA_delta'],reverse=True)[:8]
attributechange=table(['Domain','Attribute','mA delta','Label-F1 delta','Target positives','Source positives'],[[x['domain'],x['attribute'],f(x['mA_delta']),f(x['label_F1_delta']),x['target_positives'],x['source_positives']] for x in best+worst])
write(R/'DINO_LODO_RESULTS.md','''# Frozen DINOv3 LODO: complete, independently verified

Private Kaggle upar2027-dinov3-frozen-threefold version2 COMPLETE. All19Kaggle tests passed. Fresh general-purpose DINOv3 pretrained backbone frozen; exact three existing disjoint source-training/selection/calibration/target partitions, original square-root-domain sampler and source-only class-balanced masked BCE. All three heads ran30epochs maximum with source-only early stopping/checkpoint selection. Target inference followed saved checkpoint/config/calibration/split locks; no target-driven settings were changed. No all-domain model or competition submission was created.

## Held-out scores

'''+rawcal+'\n'+metricstable+'''
Challenge_Avg is the arithmetic domain macro of the per-domain harmonic mean of mA and instance-F1. Primary instance-F1 follows the locked baseline convention: harmonic mean of mean image precision and mean image recall. Both-supported labels enter mA; exact private organizer edge cases remain unresolved. These previously inspected targets are development evaluation, not untouched certification of generalization or a prediction of the official leaderboard score.

## Selection, calibration and gaps

'''+source+'''
Source-selection scores are descriptive on the selection partition used for checkpoint selection. Calibrated source-selection metrics use thresholds fit only on its disjoint source-calibration partition; no target fitting. Ensemble source-selection metrics use the corresponding preserved SigLIP source predictions. Target gaps confound domain shift with class balance, source domain composition and training-set size; they do not isolate a causal mechanism.

'''+calibration+'''
Source-only calibration improved Challenge_Avg for both branches in all three targets, mainly trading precision for recall and label-balanced accuracy. DINO alone remains substantially below SigLIP2. Calibration does not close the entire standalone gap.

## Source convergence and compute

'''+convergence+'''
PA is best at the30epoch limit and still improving; PETA peaks at27,Market at25. Do not claim all three converged or compare raw epoch numbers with the six-epoch fine-tuned SigLIP baseline. PA source size18,668 is much smaller than PETA89,001 or Market87,669. Actual attempted/successful update counts and8total AMP-skipped updates are retained, not assumed zero. Skipped updates still consumed their sampled source images. Source losses were finite. Epoch curves, source scores, draws and unique counts are retained in SOURCE_CURVES_SUPPORT.json and SOURCE_CURVES.csv; a plot is exported separately when available.

All source images were eventually seen at least once in each fold. Measured exposure fractions match the expected square-root sampler: PA-held Market/PETA .51852/.48148; PETA-held Market/PA .26297/.73703; Market-held PA/PETA .75100/.24900. Frequencies/BCE weights were computed from the actual source partition, not sampled or target data. Every one of40source attributes has positive and negative support in all3folds. PA-held lower-orange has only4source positives; this is sparse support, not missing supervision. Full source counts and target TP/TN/FP/FN are in the aggregate CSV/JSON files.

## Attribute behavior and uncertainty

'''+groupdelta+'''
Group deltas compare calibrated ensemble with calibrated SigLIP. Age and accessory mA rise across all three domains; upper/lower clothing colors are mixed, including lower-color mA regressions on PA/PETA. Accessory label-F1 falls onMarket despite higher group mA. There is no uniformly improved attribute group. Largest supported-label changes below include low-support cases and should not be interpreted without the support columns.

'''+attributechange+'''
Full per-attribute and per-group raw/calibrated tables: reports/dino_v3/results/ATTRIBUTE_METRICS.csv and GROUP_METRICS.csv. Zero-target-positive labels and very rare labels have unstable/undefined balanced accuracy; do not turn these observations into hidden-test assumptions.

## Scorer sensitivity

'''+sensitivity+'''
The ensemble improvement remains positive under both instance-F1 conventions and the all-valid safe-zero sensitivity. Numerical promotion criteria were predeclared on the locked primary convention and are not moved. Mean-image sensitivity is diagnostic, not a substitute selected after seeing results.

## Verification boundary

Independent CPU confusion-count calculations reproduce all recorded target metrics. Source-only DINO and ensemble thresholds refit exactly. Source-selection metrics, row identities, all CSV/checkpoint/config/calibration locks, original foundation/code identity, source prevalence/BCE weights, source exclusion and exposure accounting verify. The three small selected heads were downloaded privately toD: and hash-checked. The1.208GBcache remains private onKaggle; its recorded identity/shape/hash were verified, but its payload was not downloaded and independently rehashed locally. No claim of a second full-cache integrity pass is made. Original31baseline report files and accepted ZIP/best/last/calibration hashes remain unchanged.
''')
write(R/'ENSEMBLE_COMPARISON.md','''# Fixed75/25 SigLIP2 + frozen DINOv3: useful but not promoted

'''+compare+'''
Use the predeclared weighted average of raw sigmoid probabilities, not logits or already calibrated outputs. A separate global40threshold vector was fit on each fold's source-calibration data only, then locked before target inference. SigLIP checkpoints/predictions were unchanged. No ensemble-weight or threshold search used targets.

The ensemble improves macro by0.003704 and worst-domain by0.003057. All three domains improve, but macro0.800964 falls0.001336 short of the fixed0.8023criterion. Worst0.780285>=0.7772 and no-regression criteria pass. **The frozen ensemble is not promoted.**

## What changes

Macro mA0.785674->0.786367 (+0.000693),instance-F10.809226->0.816127 (+0.006901),precision0.787004->0.794574 (+0.007570),recall0.833028->0.839026 (+0.005998),label-F10.529495->0.536266. Most aggregate benefit is instance-level precision/recall, not a broad recovery of label-balanced mA.

PA:mA+0.000764,F1+0.010083;PETA:mA-0.000548,F1+0.006840;Market:mA+0.001864,F1+0.003781. PETA's Challenge gain should not be described as a balanced-accuracy gain.

'''+metricstable+'\n'+groupdelta+'''
## Complementarity

'''+correlation+'''
Residual correlation0.747-0.776 and binary-error correlation0.537-0.558 show useful, incomplete agreement. Correlation depends on label prevalence and is descriptive; it is not proof of causality or independent representations. Consistent target gains support complementarity despite the standalone gap, but a single development seed/previously inspected targets do not certify future or official-score gains. Calibration transfer contributes substantially: raw ensemble macro0.790884 versus rawSigLIP0.789605; calibrated ensemble0.800964 versus calibratedSigLIP0.797260.

## Decision

Do not deploy DINO standalone or the current frozen ensemble. The predeclared review-only near-useful conditions are met (macro>=.8003,worst>=.7752,residual correlations<.85), which supports considering exactly one limited final-two-block experiment. It does not override the primary promotion gate or authorize an automatic final all-domain run. Retain original square-root sampling and masked class-balanced BCE; equal-domain sampling remains rejected by its completed controlled comparison. No further ensemble-weight search.

## Inference and packaging boundary

DINO-only warmed T4 projection for29248decode/preprocess/forward/pooling images is~102s; the previous scored SigLIP path was~110s on its official runtime. Adding these heterogeneous numbers is not an ensemble benchmark. Model loading, independent aspect preprocessing,40head logits,raw-probability mixing,newcalibration,serialization and the exact A10 deployment path still need an inclusive <=240s measurement. Never blend the accepted model's threshold-shifted output with raw DINO probabilities. Preserve accepted metadata and calibration; a future ensemble needs a separate wrapper/calibration file. No promoted/all-domain candidate or ZIP exists.
''')
write(R/'FINAL_CANDIDATE_DECISION.md','''# Decision: preserve1003877; frozen DINOv3 is not promoted

'''+compare+'''
The fixed calibrated75/25ensemble improves every domain and satisfies worst/no-regression criteria, but misses the predeclared macro>=0.8023 gate by0.001336. DINOstandalone macro0.754043/worst0.724861 is unsuitable. Do not lower the gate, train an all-domain frozen candidate, package it as validated or upload it. Official accepted1003877 remains0.770450; these development results do not guarantee beating that official score or0.8021.

## One next experiment worth reviewing

Recommend considering exactly one conservative final-two-block DINOv3 fine-tune, with backboneLR~5e-6/headLR~5e-5,original sampler/BCE,unchanged partitions/source-only selection/calibration and fixed75/25probability mixing. Keep the frozen selected heads as fallback. Use a short source-only backward/memory/throughput gate before full allocation. Freeze all earlier blocks. Predeclare the source schedule and unchanged promotion rule before any target evaluation; no full-backbone training or weight/grid search.

Why this is more promising than changing the accepted recipe blindly: the frozen second branch already improves all3development targets, gains survive scorer sensitivities, and residual correlations are below.78. Yet its standalone mA trails badly and source-target gaps persist. Modest feature adaptation is a plausible next lever, not an established causal remedy. The data do not show that SigLIP suffered representation degradation, nor that calibration or missing labels alone explain the gap. Imbalance, sparse PA-held lower-orange support,domain shift,limited frozen-feature transfer and unequal source sizes/updates are confounded. PA head is still best at its30epoch limit; this further limits architecture-level conclusions.

No fine-tune was automatically launched. Planning estimate for the single controlled threefold upper-two-block experiment:~1-2GPU-sessionh,unverified until backward throughput is measured. To remain within the existing cumulative2h diagnostic allocation, cap any additional diagnostic at1.5h and stop if projection cannot fit; about.31h has already been used across DINOv2 cancellation,V3gate,v1failure andv2success. A selected all-domain DINO run would tentatively need another~1-2h plus~0.5-1h exact deployment/package checks; these are estimates,not measured training times. Keep at least6h of the fresh10.66hquota reserved until the actual benchmark establishes a tighter plan.

## Deadline plan

- October21 by18:00PKT: review this report and decide whether to authorize the one limited fine-tune. Use the verified frozen comparison and accepted result for academic reporting today; do not wait for an improved competition model to document current evidence. Academic portal/human/team details and submission acceptance remain unverified.
- October21 by22:00PKT: freeze the final recipe only if its complete threefold comparison satisfies the original promotion rule. Otherwise retain1003877 and stop model experiments.
- October22 by06:00PKT: finish any conditionally promoted all-domain training and source-validation calibration, preserving originalSigLIPweights/calibration.
- October22 06:00-10:00PKT: build offline package with originalmetadata/exact40order/licenses; real-source-image correctness,network-disabled loading and full decode/bothforwardpasses/calibration/serialization benchmark <=240s. Reject a candidate that misses the margin.
- October22 10:00-12:00PKT: deliver concrete candidate/hashes/evidence for explicit user upload approval;preferred upload13:00PKT,hardcutoff17:00PKT. No automatic upload or organizer email.

The frozen diagnostic is complete and reviewed. Follow-up monitoring is disabled after delivery because no job is active and the next experiment requires this review. Original accepted ZIP,best,last and calibration were freshly rehashed unchanged;all31lockedbaseline files verify. No hidden data/identities/test characteristics or leaderboard tuning were used.
''')
# Retain full source trajectories as a portable aggregate CSV.
rows=[]
for d in folds:
 for row in curves[d]['history']:rows.append({'held_out':d,'epoch':row['epoch'],'loss':row['loss'],'source_selection_score':row['source_selection_score'],'epoch_seconds':row['epoch_seconds'],'updates':row['updates'],'amp_skips':row['amp_skips']})
with (P/'SOURCE_CURVES.csv').open('w',newline='',encoding='utf-8') as stream:w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for d in folds:
 for n in ['EVALUATION.json','HISTORY.json','CONFIG.json','CALIBRATION.json','LOCK.json','SOURCE_SUPERVISION.json']:shutil.copyfile(S/d/n,P/(d+'_'+n))
for n in ['FEATURE_RECEIPT.json','INITIALIZATION_RECEIPT.json','FEASIBILITY.json']:shutil.copyfile(S/n,P/n)
st=read(S/'STAGING.json');st.pop('events',None);write(P/'STAGING.json',json.dumps(st,indent=2)+'\n')
for n in ['ENVIRONMENT_BEFORE.json','ENVIRONMENT_AFTER.json','NATIVE_TESTS.xml','hardware.json']:shutil.copyfile(A/n,P/n)
# Compare actual embedded runtime hashes with the submitted source receipt.
receipt=read(R/'reports/dino_v3/FROZEN_SUBMISSION_RECEIPT.json');actual=read(A/'source_hashes.json')
for n,expected in actual.items():assert receipt['source_sha256'][n]==expected,(n,expected)
write(P/'RUNTIME_SOURCE_VERIFICATION.json',json.dumps({'passed':True,'runtime_source_files':len(actual),'source_commit':receipt['source_commit'],'script_sha256':receipt['script_sha256'],'native_sha256':actual['dino/v3_model.py']},indent=2)+'\n')
# Append factual runtime evidence to the already preserved feasibility report.
p=R/'GPU_RUNTIME_REPORT.md';text=p.read_text(encoding='utf-8');text+='''

## Frozen threefold v2 completed

All19Kaggletests passed with Transformers4.57.1 after the baseline reference pin was restored. ENVIRONMENT_BEFORE records5.16.1 in the freshv2base and AFTER4.57.1;the failedv1environment was not logged,so this is strong drift evidence,not a direct measurement ofv1's version. Torch2.11.0cu128/CUDA12.8,Pillow12.3.0,NumPy2.1.3 retained. No preprocessing/model/test/tolerance changes.

Actual archive staging36.855s,wholearchiveSHA/ZIPCRC/exact131076pathunion passed. Two extraction ranges each65538images completed249.932/250.260s (~524images/saggregate). CacheFP16shape131076x6x768,recordedSHA bd7991deafba88f07edd0d53525252b20989a52ac2e355846b4ce8aa62a2285d. Parent entered extraction146.47s,heads410.10s,complete520.14s from scriptstart;runner446.02s. The broader quota increment~.15h also includes setup/export/rounding and should not be confused with runner-only7.43min.

Cached head wallPA24.32s,PETA85.78s,Market78.99s;parallel scheduling finishes headphase~110s. Each30epochs,best30/27/25;attempted4380/20880/20550,successful4380/20876/20546,AMPskips0/4/4. These are cached-head timings,not estimates of backpropagation through two backbone blocks. All actualimages/sourceexposure/sourcecurves retained in reports/dino_v3/results.

Fresh quota19.34hused10.66hremaining. TotaldisplayedDINOspending~.31hfrom10.97initial,wellbelow cumulative2hcap. No GPUjob active,no all-domainrun,newZIP or submission. Trainedselectedheads also loaded strictly and inferred on real source-only images under Python3.11.16/torch2.4.1CPU with socketsdisabled,no timm/Transformers import,40orderedfinite probabilities/calibration/serialization verified. This does not certify full exactdeployment runtime;A10benchmark remains gated on a promoted candidate.
''';write(p,text)
p=R/'DINO_ARCHITECTURE.md';text=p.read_text(encoding='utf-8');text+='\n\nFrozen threefold v2 is complete. GPU FP32/AMP native/timm parity0.0 and19Kaggletests passed;all131076compact descriptors extracted. Three selected trained heads also pass the native Python3.11/torch2.4.1CPU source-image probe without timm/Transformers/network. See DINO_LODO_RESULTS.md,ENSEMBLE_COMPARISON.md andFINAL_CANDIDATE_DECISION.md. Frozenensemble does not meet primarypromotion;no all-domain candidate orupload.\n';write(p,text)
print('Final aggregate reports written; baseline/private arrays/weights unchanged.')
