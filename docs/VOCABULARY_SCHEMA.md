# Medication vocabulary schema v1.0

`data/vocabulary/medication_vocabulary.csv` follows the v1.0 schema declared
in `src/vocabulary.py`. The schema version describes the file format; it does
not, by itself, certify the medical terms or assign a released vocabulary
version.

| Column | Meaning |
|---|---|
| `schema_version` | Must equal `1.0` for every record. |
| `product_id` | Unique source-backed identifier for this normalized product/term row. |
| `drug_name` | Preferred display name for this product record. |
| `inn` | Source-verified International Nonproprietary Name, where applicable. |
| `ingredient_id` | Stable identifier for a single ingredient; aliases of the same ingredient share it. |
| `brand_names`, `aliases` | Pipe-delimited brand and accepted equivalent spellings, each mapped by this row. |
| `combination_ingredient_ids` | Pipe-delimited stable component IDs for combination products (at least two). |
| `metaphone_key`, `soundex_key` | Generated phonetic retrieval keys; these are not clinical equivalence evidence. |
| `source`, `source_release` | Auditable source and exact release/version used to resolve the row. |

Run `validate_vocabulary_csv(path)` before any pipeline run. The checked-in
development vocabulary has source-mapped INN/RxNorm fields, unique product
identifiers, brand-to-generic mappings, and a two-component combination example.
Empty brand/component cells on single-ingredient rows mean that no brand alias
or combination list is assigned to that row; they are not unresolved IDs. The
vocabulary release alone does not authorize training: corpus access, pinned
revisions, sample feasibility, and evaluation partition checks remain required.

The phonetic keys are candidate-retrieval features only. They must not be used
as evidence that two products are therapeutically equivalent or interchangeable.
