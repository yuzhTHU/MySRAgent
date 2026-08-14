---
name: discover-symbolic-laws
description: Discover compact symbolic-regression formulas from tabular numerical data by diagnosing scale, domain, nonlinear relevance, singularities, invariants, one-dimensional shape, separability, transformed linearity, residual structure, numerical sensitivity, outliers, and recognizable constants before broad symbolic search. Use when variables are anonymized, direct guesses fail, predictive accuracy has not produced a simple law, or an agent must choose transformations and discovery tools from numerical evidence.
---

# Discover Symbolic Laws

Treat symbolic regression as iterative scientific diagnosis. Make each computation answer one structural question, and distinguish a diagnostic surrogate from a final symbolic law.

## Keep evidence compact

- Compute over all samples inside a tool or script; return summaries only.
- Print shapes, domain flags, robust quantiles, fitted coefficients, stability, top candidates, and residual metrics. Do not print full arrays or feature libraries.
- Inspect at most 5–10 rows selected by an explicit rule such as largest residual or smallest candidate denominator.
- Bound candidate tables at 5–10 entries. Preserve the best compact formula before an expensive branch.
- Use deterministic random seeds and restrict BLAS threads when custom code invokes heavy numerical libraries.

## Run the discovery loop

### 1. Audit marginal distributions

Establish finite coverage, signs, zero proximity, robust quantiles, mean, variance, dynamic range, and whether absolute, relative, or log error is informative.

Start with:

```text
statistics_analysis(n_bins=10, near_zero_threshold=1e-8)
```

Request expression statistics when testing a transformation:

```text
statistics_analysis(variables=["y*x2", "log(y)"])
```

Use a log only on a valid domain. If values cross zero, consider `sign(z)*log1p(abs(z))`, separate sign and magnitude, or an evidence-backed shift.

If `statistics_analysis` is unavailable and `code_executor` is available, run a bounded audit such as:

```python
import numpy as np
# `data` denotes the dictionary supplied to the execution program.
for name, a in data.items():
    a = np.asarray(a, float).ravel()
    finite = np.isfinite(a)
    z = a[finite]
    q = np.quantile(z, [0, .01, .25, .5, .75, .99, 1])
    print(name, "n", len(a), "finite", round(finite.mean(), 4),
          "negative", round(np.mean(z < 0), 4),
          "near_zero", round(np.mean(np.abs(z) <= 1e-8), 4),
          "quantiles", np.round(q, 6).tolist(),
          "mean", float(np.mean(z)), "std", float(np.std(z)))
```

### 2. Analyze relationships and collapse

Call:

```text
relationship_analysis(
  variables=["x1", "x2", "x1/x2"], y="y", n_bins=10,
  validation_fraction=0.2, n_repeats=3, collapse_model="bins"
)
```

Use feature-target Pearson/Spearman values, conditional target bins, held-out collapse, and stability to select a coordinate. Enable `pairwise=true` only when predictor-predictor relationships matter; its output grows quadratically.

Treat high in-sample but weak held-out collapse as overfitting. Try `collapse_model="spline"` for a smooth nonmonotone curve and `"isotonic"` for a monotone probe. These are structural probes, not final formulas.

Pearson and Spearman both being weak does not prove irrelevance. Before removing a variable, inspect nonlinear relevance, nonmonotone shapes, and combinations such as sums, differences, products, ratios, or angular differences.

### 3. Classify plausible structure

Test cheap families before broad search:

- additive or low-degree polynomial;
- multiplicative power law;
- rational, saturating, or singular;
- exponential or logarithmic;
- periodic or angular;
- separable or low-dimensional composite.

Use behavior, not names. Large positive dynamic range suggests `log(y)`. A stable log-log slope suggests a power. Blow-up near small `xi` suggests inverse powers. Saturation suggests a rational or exponential approach to a limit. Repeated zero crossings suggest a periodic coordinate.

After a one-dimensional collapse, inspect values near zero, extrema, poles, saturation levels, and both observed-range endpoints. Prefer `expm1(z)` over `exp(z)-1` near zero.

### 4. Search for invariants and eliminate variables

Test transformations that may become constant or reduce dimension:

```text
y*g(xi), y/g(xi), log(abs(y))-log(abs(g(xi)))
y*product(xi**pi), residual/g(xi), residual*g(xi)
```

Use:

```text
constant_fit(eq="y*x2/x1", use_eq_as_y=true)
relationship_analysis(variables=["x1/(x3*x4)"], y="y*x2")
```

Prefer low robust dispersion and stable constants across subsets. Do not accept an invariant merely because its ordinary mean is large relative to its standard deviation.

### 5. Linearize the selected hypothesis

- Fit a multiplicative law with `power_law_fit(x=["x1", "x2"], y="y")`.
- Request snapping only after raw exponents are stable: `power_law_fit(..., snap_exponents=true, snap_tolerance=0.03)`.
- Inspect both raw and snapped candidates; accept snapping only when the tool reports `snap_accepted=true`.
- Fit a rational grid when shape evidence suggests a pole or saturation:

```text
rational_fit(
  x=["x1"], y="y",
  numerator_degrees=[0,1,2], denominator_degrees=[0,1,2],
  validation_fraction=0.2, top_k=5
)
```

- Reject high-degree candidates with weak held-out improvement, ill conditioning, or dangerous denominator quantiles.
- Fit transformed additive models with `polynomial_fit`, `call_sindy`, or a small custom design matrix.
- For positive variables, try `call_sindy(y="log(y)", x=["log(x1)", "log(x2)", "log(x3)"], poly_degree=1, include_trig=false, threshold=0.01)`.
- Use SINDy for a modest sparse library. Use PySR only after narrowing variables, operators, and target transformations.

Do not infer a law from correlation alone. Screen on a deterministic subset, select structure on held-out data, refit constants on all samples, then run the final evaluator.

### 6. Analyze residual structure

For each serious candidate, call:

```text
evaluate_formula(f="candidate", show_diagnostics=true)
```

Inspect absolute and relative residual quantiles, extreme rows, finite coverage, and residual dependence. If residual structure remains, revise the factorization or discover the residual recursively. Fit a dominant additive term, subtract it, and repeat.

Test exchange symmetry or antisymmetry between similarly distributed variables. Natural coordinates often appear as `xi+xj`, `xi-xj`, `xi*xj`, `xi/xj`, or angular differences.

When errors concentrate near a zero or pole, distinguish a missing singular factor from finite-precision generation or an unstable algebraic form before adding polynomial terms.

### 7. Recover exact structure

- Snap exponents only when subset stability, snap distance, and full-data degradation all support it.
- Fit the final scale with `constant_fit(eq="structural_expression", y="y")`.
- Accept a named constant only when the tool marks the replacement acceptable; compare the simplest and most precise representations.
- Refit the remaining scalar after structural snapping.
- Prefer fewer operations when errors are indistinguishable.

### 8. Validate and submit

Separate discovery metrics from the final metric:

- Use rank, trimmed, signed-log, relative, log, or collapse scores to reveal structure.
- Use the benchmark's original-domain metric and all valid samples to choose the final formula.
- Check extrema, near-zero targets, singular regions, and finite prediction coverage.

Submit only after residuals are structureless at expected numerical precision, or explicitly retain the best compact approximation. Stop dense polynomial, Chebyshev, or genetic-programming branches when expression size grows faster than held-out improvement.

## Custom diagnostic recipes

Use these only when the corresponding specialized tool is absent or insufficient. Use `code_executor` **if available**. Adapt how the execution environment exposes `data`; keep output bounded.

### Analyze one-dimensional shape

Principle: after proposing a coordinate `z`, estimate whether `y=f(z)` generalizes, then inspect monotonicity, turning points, and local log-log elasticity. A flexible curve is evidence of dimensional collapse, not the final answer.

```python
import numpy as np
from scipy.interpolate import UnivariateSpline
rng = np.random.default_rng(0)
z = np.asarray(data["x1"], float).ravel()
y = np.asarray(data["y"], float).ravel()
ok = np.isfinite(z) & np.isfinite(y)
z, y = z[ok], y[ok]
order = rng.permutation(len(y)); cut = int(.8*len(y))
tr, te = order[:cut], order[cut:]
sort = np.argsort(z[tr]); zs, ys = z[tr][sort], y[tr][sort]
zu, inv = np.unique(zs, return_inverse=True)
yu = np.bincount(inv, weights=ys) / np.bincount(inv)
spline = UnivariateSpline(zu, yu, k=min(3, len(zu)-1),
                          s=len(zu)*np.var(yu)*1e-3, ext=3)
pred = spline(z[te])
r2 = 1-np.sum((y[te]-pred)**2)/np.sum((y[te]-np.mean(y[tr]))**2)
grid = np.linspace(np.quantile(z,.01), np.quantile(z,.99), 200)
curve = spline(grid); slope = spline.derivative()(grid)
turns = np.flatnonzero(np.sign(slope[1:]) != np.sign(slope[:-1]))
pos = (grid > 0) & (np.abs(curve) > 0)
elasticity = grid[pos]*slope[pos]/curve[pos]
print("heldout_r2", float(r2), "turning_points", np.round(grid[turns[:8]],6).tolist(),
      "elasticity_q", np.round(np.quantile(elasticity,[.1,.5,.9]),4).tolist())
```

Example output:

```text
heldout_r2 0.9987 turning_points [] elasticity_q [1.96, 2.01, 2.05]
```

Interpretation: strong held-out collapse, no turning point, and elasticity near 2 support `y ∝ z**2`. If elasticity grows roughly linearly with `z`, test an exponential. If it approaches a negative integer in the tail, test a rational denominator power.

### Analyze nonlinear variable relevance

Principle: tree ensembles and held-out permutation importance can reveal U-shaped, thresholded, or interaction-only dependence missed by Pearson and Spearman. Use them only to select variables and transformations.

```python
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
names = [k for k in data if k != "y"]
X = np.column_stack([np.asarray(data[k],float).ravel() for k in names])
y = np.asarray(data["y"],float).ravel()
ok = np.isfinite(X).all(1) & np.isfinite(y); X,y=X[ok],y[ok]
if len(y)>20000:
    idx=np.random.default_rng(0).choice(len(y),20000,replace=False); X,y=X[idx],y[idx]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=0)
model=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=3,
                          random_state=0,n_jobs=1).fit(Xtr,ytr)
imp=permutation_importance(model,Xte,yte,n_repeats=3,random_state=0,n_jobs=1)
ranking=sorted(zip(imp.importances_mean,names),reverse=True)
print("heldout_r2",float(model.score(Xte,yte)),"importance",[(n,round(float(v),4)) for v,n in ranking[:10]])
```

Example output:

```text
heldout_r2 0.97 importance [('x3', 1.41), ('x1', 0.52), ('x2', 0.003)]
```

Interpretation: investigate `x3` and its combinations first; do not submit the forest. Repeat on raw `y`, `sign(y)*log1p(abs(y))`, and a rank target when extremes dominate. A weak held-out model means its importance ranking is not trustworthy.

### Diagnose numerical and domain sensitivity

Principle: a structurally correct formula may look imperfect when data were generated in lower precision or when equivalent algebraic forms suffer cancellation. Quantify this rather than tweaking coefficients.

```python
import numpy as np
x1=np.asarray(data["x1"],float); x2=np.asarray(data["x2"],float)
y=np.asarray(data["y"],float)
f64=x1/(np.exp(x2)-1)
f32=(x1.astype(np.float32)/np.expm1(x2.astype(np.float32))).astype(float)
for name,p in [("float64_exp_minus_1",f64),("float32_expm1",f32)]:
    r=p-y; finite=np.isfinite(p)&np.isfinite(y)
    print(name,"finite",round(float(finite.mean()),5),
          "rmse",float(np.sqrt(np.mean(r[finite]**2))),
          "max_abs",float(np.max(np.abs(r[finite]))))
den=np.expm1(x2)
idx=np.argsort(np.abs(den))[:5]
print("smallest_denominator",[(int(i),float(den[i]),float(y[i])) for i in idx])
```

Example output:

```text
float64_exp_minus_1 finite 1.0 rmse 0.0021 max_abs 0.18
float32_expm1 finite 1.0 rmse 3.2e-07 max_abs 1.1e-05
```

Interpretation: the large improvement under float32-compatible `expm1` suggests generation precision or cancellation, not a missing polynomial correction. Also inspect denominator, radicand, and log-argument quantiles and the fraction of total SSE contributed by the worst samples.

### Analyze robust views and outlier influence

Principle: extreme values can hide a structural dependence. Use robust views for discovery, but always validate the final law on untrimmed original data.

```python
import numpy as np
y=np.asarray(data["y"],float).ravel()
for name,x in data.items():
    if name=="y": continue
    x=np.asarray(x,float).ravel(); ok=np.isfinite(x)&np.isfinite(y)
    xx,yy=x[ok],y[ok]
    lo,hi=np.quantile(yy,[.01,.99]); keep=(yy>=lo)&(yy<=hi)
    signed=np.sign(yy)*np.log1p(np.abs(yy))
    corr=lambda a,b: float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else float("nan")
    print(name,"raw",corr(xx,yy),"trimmed",corr(xx[keep],yy[keep]),
          "signed_log",corr(xx,signed),
          "top1pct_y2_share",float(np.sort(yy*yy)[-max(1,len(yy)//100):].sum()/np.sum(yy*yy)))
```

Example output:

```text
x2 raw 0.03 trimmed 0.81 signed_log 0.76 top1pct_y2_share 0.94
```

Interpretation: `x2` is plausibly relevant even though raw correlation is weak; inspect singular or multiplicative transformations. Do not optimize only the trimmed metric—the extreme region may contain the most informative evidence about a pole.

### Screen singular compensation

```python
import numpy as np
y=np.asarray(data["y"],float)
for name,x in data.items():
    if name=="y": continue
    x=np.asarray(x,float)
    print("SINGULARITY",name)
    for idx in np.array_split(np.argsort(np.abs(x)),10)[:4]:
        print("  median_abs_x",float(np.median(np.abs(x[idx]))),
              "median_abs_y",float(np.median(np.abs(y[idx]))))
    for p in [-3,-2,-1,1,2,3]:
        z=y*np.power(x,p); good=np.isfinite(z)
        scale=np.median(np.abs(z[good]))+1e-300
        print("  p",p,"relative_mad",float(np.median(np.abs(z[good]-np.median(z[good])))/scale))
```

If `y*x` stabilizes as `x→0`, test a `1/x` factor and rediscover the compensated target.

### Analyze binned residuals

```python
import numpy as np
r=y-pred
den=np.maximum(np.abs(y),np.quantile(np.abs(y),.01)+1e-300)
rr=r/den
print("residual_q",np.quantile(r,[0,.01,.5,.99,1]).tolist(),
      "relative_q",np.quantile(rr,[0,.01,.5,.99,1]).tolist())
for name,x in data.items():
    if name=="y": continue
    edges=np.quantile(x,np.linspace(0,1,9))
    bins=np.clip(np.searchsorted(edges[1:-1],x),0,7)
    summary=[]
    for b in range(8):
        z=r[bins==b]
        summary.append((len(z),float(np.mean(z)),float(np.sqrt(np.mean(z*z)))))
    print(name,"bins(n,mean,rmse)",summary)
idx=np.argsort(np.abs(rr))[-5:]
print("worst_indices",idx.tolist())
```

### Fit a small motivated feature library

```python
import numpy as np
y=np.asarray(data["y"],float)
terms={"1":np.ones_like(y),"x1":data["x1"],"1/x2":1/data["x2"],
       "log(x1)":np.log(data["x1"]),"cos(x3-x4)":np.cos(data["x3"]-data["x4"])}
names=list(terms); A=np.column_stack([terms[n] for n in names])
ok=np.isfinite(A).all(1)&np.isfinite(y)
b=np.linalg.lstsq(A[ok],y[ok],rcond=None)[0]
print("condition",float(np.linalg.cond(A[ok])),
      "top_terms",sorted(zip(np.abs(b),names,b),reverse=True)[:10])
```

Build only evidence-motivated terms. Treat a many-term approximation as a diagnostic, not a discovered law.
