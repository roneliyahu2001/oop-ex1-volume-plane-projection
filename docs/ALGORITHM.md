# The projection algorithm

## 1. The problem, made well-defined

The assignment's pseudo-code leaves several decisions open. The table lists them and what we chose.

| Open question | Decision | Reason |
|---|---|---|
| Coordinate system of the points | Voxel indices (`M[x][y][z]` ↔ `(x,y,z)`); world (NRRD `space`) coordinates optional | Matches the problem statement; world mode gives correct aspect on anisotropic scans (MRHead has 1.3 mm along one axis) |
| What is "the value at `p`" when `p` is between voxels? | Closest voxel, `floor(p + 0.5)` per axis | Answer to Q&A #4; deterministic tie-breaking (round half up) |
| Size of `E` (`Dab × Dbc` is not an integer) | `floor(Dab)+1` rows × `floor(Dbc)+1` columns | Includes both the apex row (`y = 0`) and the base row; the pseudo-code steps *before* sampling, so it skips the apex and can miss the base |
| Accumulate `a1 += v` each row? | Compute `a1(y) = a1 + y·v` directly | No accumulated floating-point drift |
| Forward scatter (`x = round(...)`, `E[x][y] = v`) or inverse mapping? | **Inverse mapping**: for every pixel compute the 3D point it shows | Forward scatter can leave holes / overwrite pixels after rounding; the inverse mapping fills every pixel exactly once and vectorises well |
| Which pixels belong to the shape? | Pixel centre within ½ pixel of the row segment | Gives a one-pixel apex for any `Dbc`, and a fully covered base row |
| Outside of `M` / outside the triangle | `0` | As specified (`v = 0` if `p ∉ M`) |
| 4 points | Trapezoid with legs `a1→b`, `a2→c`; a triangle is the special case `a1 = a2` | One code path for both shapes |
| Invalid points | `InvalidPlaneError`; demo renders an error frame | Q&A #3 |

## 2. Geometry

The plane is stored as a quad `(a1, a2, b, c)` (for a triangle `a1 = a2 = a`).

```
          a1 ──────── a2              a (= a1 = a2)
         /              \             /\
   v_l  /   row y: A1 → A2  \ v_r      /  \
       /                    \         /    \
      b ──────────────────── c       b ────── c
```

* `Dab` = leg length, `Dbc = dist(b, c)`.
* `v_l = (b − a1)/|b − a1|`, `v_r = (c − a2)/|c − a2|` (unit steps along the legs).
* Row `y ∈ {0, …, floor(Dab)}`:
  `A1(y) = a1 + y·v_l`, `A2(y) = a2 + y·v_r`, `d12(y) = |A2 − A1|`, `dp(y) = (A2 − A1)/d12`.
* The row segment is centred in `[0, Dbc]`: it starts at column `off(y) = (Dbc − d12)/2`
  (this is `x = Dbc/2 − d12/2 + x1` from the pseudo-code, solved for `x1`).
* Pixel `(x, y)`:
  * `x1 = x − off(y)`
  * inside the shape iff `−½ ≤ x1 ≤ d12 + ½`
  * sample point `P = A1(y) + clamp(x1, 0, d12) · dp(y)`
  * voxel `(i, j, k) = floor(P + ½)`; `E[y, x] = M[i][j][k]` if inside `M` and inside the shape, else `0`.

**World coordinates.** `P` is affine in `A1` and `dp`, so mapping to index space only needs the
per-row vectors to be transformed once: `A1 ← A·A1 + t`, `dp ← A·dp`, with
`(A, t)` the inverse of `world = D·index + origin`. The plane is validated and stepped in world units.

## 3. Validation (raises `InvalidPlaneError`)

* 3 points: `|dist(a,b) − dist(a,c)| < EPS`; `dist(a,b) ≥ EPS`; `dist(b,c) ≥ EPS`; height of `a`
  above line `bc` `≥ EPS` (not collinear).
* 4 points: `|dist(a1,b) − dist(a2,c)| < EPS`; `dist(a1,a2) ≤ dist(b,c) + EPS`; coplanar: the
  distance of `a2` from the plane `(a1, b, c)` is `< EPS` (if `a1, b, c` are collinear, the plane
  `(a1, b, a2)` and the point `c` are used); non-degenerate legs. `dist(a1,a2) < EPS` → triangle.
* All coordinates finite; exactly 3 or 4 points of 3 coordinates.

## 4. Implementation (why it is fast)

`PlaneProjector.project` contains **no Python loop over pixels**; everything is NumPy array work
on arrays the size of one frame:

1. Per-row vectors `A1, dp, d12, off` (shape `rows × 3`): negligible.
2. `x1` grid (`rows × cols`, `float32`) and the inside mask.
3. For each axis `k`: `P_k = A1_k + x1·dp_k`, `floor`, cast to `int32`, bounds check and
   accumulation into a flat index `x + sx·(y + sy·z)`. The bounds check uses one unsigned
   comparison (`uint32(i) < s_k`), which also rejects negative values.
4. Invalid pixels get the index of a **zero sentinel voxel** stored after the volume data, so the
   whole frame is gathered with one `np.take` and needs no masked assignment.

The volume is kept as one flat, x-fastest buffer — exactly the NRRD on-disk order — so loading is
one `np.frombuffer` + copy, and no transposition is needed.

Complexity: `O(rows · cols)` time per frame, independent of the volume size (apart from cache
effects of the random access in the gather); memory `O(rows · cols)` per frame.

Measured on an Apple M4, `500^3` volume: **238–470 fps** depending on plane orientation and
voxel type (see [EXPERIMENT.md](EXPERIMENT.md#2-performance)).

## 5. Pseudo-code of the implementation

```
project(M, plane):
    rows ← floor(Dab) + 1 ;  cols ← floor(Dbc) + 1
    for all y in parallel:
        A1[y] ← a1 + y·v_l ;  A2[y] ← a2 + y·v_r
        d12[y] ← |A2[y] − A1[y]| ;  dp[y] ← (A2[y] − A1[y]) / d12[y]   (0 if d12 = 0)
        off[y] ← (Dbc − d12[y]) / 2
    for all (y, x) in parallel:
        x1 ← x − off[y]
        inside ← −½ ≤ x1 ≤ d12[y] + ½
        P ← A1[y] + clamp(x1, 0, d12[y])·dp[y]
        (i, j, k) ← floor(P + ½)
        E[y, x] ← M[i][j][k]  if inside and (i, j, k) ∈ M  else 0
    return E
```

A literal transcription of the assignment's pseudo-code is kept in `src/volproj/reference.py`;
the tests use it as an oracle, and the experiment compares the two.
