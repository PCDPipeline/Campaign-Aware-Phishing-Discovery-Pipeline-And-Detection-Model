# G1 Graph Construction & Feature Specification 

This document specifies the **G1 homogeneous graph** built from request-level traces.  
It is intended to let others reproduce the **graph structure** and **feature tensors**
from their own input data (without requiring our raw dataset).

---

## 1) Graph Type (PyG `Data`)

We build a **homogeneous graph** where:

- **each node** corresponds to one URL instance identified by `sid`
- **edges** connect pairs of `sid`s that share internal resource identifiers

Final object:

```python
from torch_geometric.data import Data
data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=labels)
```

---

## 2) Required User Inputs (minimum)

Users should start from a request-level table (one row per request), containing at least:

| Column | Meaning |
| --- | --- |
| `sid` | unique URL-instance identifier (node id) |
| `url` | landing URL for the instance |
| `response_url` | response URL for a request |
| `request_url` | requested URL (optional but commonly available) |
| `status` | HTTP status code (optional) |
| `ssdeep` | content fingerprint for response body/resource |
| `file_type` | file type string (e.g., output of `file`, MIME-ish, etc.) |
| `tech` | list of detected technology tokens |
| `registrar_iana_id` | registrar identifier for the registered domain |

---

## 5) URL–URL Edge Construction

### 5.1 Grouping

Build SID groups by shared identifiers:

```python
path_groups[path_id] = {sid_1, sid_2, ...}
ssdeep_groups[ssdeep_id] = {sid_1, sid_2, ...}
```

### 5.2 Group-size cap (recommended)

To avoid overly-dense cliques from very common resources:

```python
keep only groups with len(group) <= MAX_GROUP_SIZE
```

### 5.3 Pairwise edges (bidirectional)

For each group, generate all pairs:

```python
for each unordered pair (sid_a, sid_b):
    add directed edges (a → b) and (b → a)
```

Union all edges from path-groups and ssdeep-groups into:

- `edge_index`: shape `[2, E]`, where nodes are indexed by `sid_to_index[sid]`.

---

## 6) Edge Attributes (`edge_attr`)

For each directed edge `(u, v)` between two SIDs:

Let:

```python
P(u) = set of path_id values seen for sid u
S(u) = set of ssdeep_id values seen for sid u
```

Compute:

```python
common_path_ids   = |P(u) ∩ P(v)|
common_ssdeep_ids = |S(u) ∩ S(v)|
```

Then define the two edge features used in our implementation:

- `common_internal_requests`
- `common_external_requests`

These are derived from whether each sid has internal/external activity (using `external_domain` observations).

**Result**

- `edge_attr`: shape `[E, 2]`

---

## 7) Node Features (X)

Each node is a `sid`. The node feature vector is concatenated from:

### 7.1 Request volume (3 dims)

Per sid:

- total requests  
- internal requests  
- external requests  

### 7.2 Technology multi-hot (`|tech_vocab|` dims)

`tech` is multi-label.  
Build a multi-hot vector over a frozen vocabulary (`tech_vocab`).

### 7.3 Registrar index (1 dim)

Map registrar identifiers to integer indices:

- unknown/missing → `0`
- known registrar IDs → `1..K`

Append as a scalar feature.

### 7.4 File-type encoding (optional)

Normalize file types into coarse classes and encode counts per sid.  
If reproduced, users must use the same frozen ordering as training.

**Final**

- `X`: shape `[N, D]`

---

## 8) Labels (y)

Binary node labels:

- `0` benign  
- `1` phishing  

Tensor:

- `y`: shape `[N]`

---

## 9) Tunable Parameters

Users should set:

- `MIN_PATH_LEN` (default 9): only longer paths become `path_id`
- `MAX_GROUP_SIZE`: cap group size to avoid dense cliques

These are dataset-dependent and can be tuned for scale and noise.
