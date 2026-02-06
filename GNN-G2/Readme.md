# 🧱 Feature Sets Required for G2 Inference 

We provide a trained **G2 Heterogeneous GraphSAGE** checkpoint and a compatible **base hetero-graph schema**.
To run inference on new data, users must construct feature tensors **using the provided vocab ordering**.

## Required Graph Schema (PyG `HeteroData`)

Node types:
- `URL`
- `HAR`
- `Registrar`

Edge types:
- (`Registrar`, `registers`, `URL`)
- (`URL`, `linked_to`, `HAR`)
- (`HAR`, `linked_to`, `URL`)

## Required Node Feature Tensors

Let:
- `N` = number of URL nodes to score
- `M` = number of HAR nodes linked to those URLs

#### 1) URL node features: `data["URL"].x`
Shape:

`D_url` must exactly match the training dimension. `URL.x` is formed by concatenating:

1. `har_count` per URL instance → `[N, 1]`  
2. `external_resource_count` per URL instance → `[N, 1]`  
3. `internal_resource_count` per URL instance → `[N, 1]`  
4. `tech_onehot` using `vocab.json["tech_vocab"]` → `[N, |tech_vocab|]`  
5. `filetype_counts` for internal/external resources using `vocab.json["url_file_types"]`
   → `[N, 2 * |url_file_types|]`

Final:
[N, D_url]


`D_url` must exactly match the training dimension. `URL.x` is formed by concatenating:

1. `har_count` per URL instance → `[N, 1]`  
2. `external_resource_count` per URL instance → `[N, 1]`  
3. `internal_resource_count` per URL instance → `[N, 1]`  
4. `tech_onehot` using `vocab.json["tech_vocab"]` → `[N, |tech_vocab|]`  
5. `filetype_counts` for internal/external resources using `vocab.json["url_file_types"]`
   → `[N, 2 * |url_file_types|]`

Final:


D_url = 3 + |tech_vocab| + 2*|url_file_types|


**Important:** The ordering of one-hot indices must follow `vocab.json` exactly.

---

#### 2) HAR node features: `data["HAR"].x`
Shape:
[M, D_har]


`HAR.x` is formed by:

1. `urls_sharing_har` (number of distinct URLs linked to this HAR node) → `[M, 1]`  
2. `file_type_id` (integer id based on `vocab.json["har_file_types"]`) → `[M, 1]`

Final:
D_har = 2


---

#### 3) Registrar node features: `data["Registrar"].x`
Shape:
[R, D_reg]


If registrar features are not available for user data, inference can still be performed by
injecting a single **dummy registrar node** with all-zero features and connecting it to all new URL nodes
via (`Registrar`, `registers`, `URL`).

---

### Required Edges

Users must provide edges connecting URLs to HAR nodes:

- `data[("URL","linked_to","HAR")].edge_index` with shape `[2, E]`
- `data[("HAR","linked_to","URL")].edge_index` with shape `[2, E]`

Optionally, if your training used edge attributes:
- `edge_attr` of shape `[E, 1]` encoding `external_domain` (0 = internal, 1 = external)

---