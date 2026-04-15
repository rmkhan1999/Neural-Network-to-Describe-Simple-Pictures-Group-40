# Numbers Dataset

Part of the multimodal dataset generation project for Task 5 (AI & Text Analytics, Spring 2026).

## What this dataset is

Images of colored numbers in two sizes, paired with natural language captions and structured metadata.
Each image shows a single number (1 to 4 digits) rendered in one of four colors on a plain white background.

Example captions:
- `a small red 7`
- `a large blue 1337`
- `a small green 42`

## Pipeline

Each sample is generated in this order:

```
symbolic state → caption → canonical label → rendered image
```

The symbolic state is always the source of truth. Captions and labels are derived from it — never from the image directly.

## Sample structure

Every record in the metadata file contains:

```json
{
  "id": "numbers_train_000001",
  "task": "numbers",
  "image_path": "images/numbers_train_000001.png",
  "symbolic_state": {"size": "small", "color": "red", "digits": "7"},
  "caption": "a small red 7",
  "canonical_label": {"size": "small", "color": "red", "digits": "7", "length": 1},
  "split": "train"
}
```

## Vocabulary

| Attribute    | Allowed values              |
|--------------|-----------------------------|
| Color        | red, blue, green, yellow    |
| Size         | small, large                |
| Digit length | 1, 2, 3, 4 digits           |

## Dataset split

| Split      | Count |
|------------|-------|
| Train      | 800   |
| Validation | 100   |
| Test       | 100   |
| **Total**  | **1000** |

## Files

| File | Description |
|------|-------------|
| `config.py` | All vocabulary constants and settings — edit this to change any values |
| `generate_numbers.py` | Main generation script — creates images and metadata |
| `check_numbers.py` | Validation script — checks metadata, vocabulary, captions and image files |
| `summarise_numbers.py` | Prints frequency distribution across colors, sizes, digit lengths and splits |

## Running the scripts

From the `numbers/` root directory:

```bash
# Generate 1000 samples
python src/generate_numbers.py --num_samples 1000 --seed 42 --output_dir .

# Run validation
python src/check_numbers.py --data_dir .

# Print frequency summary
python src/summarise_numbers.py --data_dir .
```

To generate a small pilot batch first (recommended):

```bash
python src/generate_numbers.py --pilot --seed 42 --output_dir .
```

## Validation checks

The following are verified automatically by `check_numbers.py`:

- All required metadata fields are present
- Vocabulary values match the approved list
- No duplicate IDs
- Caption matches symbolic state exactly
- Canonical label is consistent with symbolic state
- Image files exist and are valid PNGs

## Design notes

- Sampling is balanced across all combinations of color, size and digit length to avoid skew
- A fixed seed (42) is used so the dataset can be fully reproduced
- No leading zeros in any generated number
- Font rendering uses DejaVu Sans Mono Bold (available by default on Colab)
- Image size is fixed at 128×128 pixels with a plain white background
