import os
import pandas as pd

# -------------------------
# Config
# -------------------------
SPLIT_ROOT = "data_preprocessing"   # or "data_preprocessing" if you want metadata after preprocessing
OUTPUT_DIR = "metadata_csv"  # where CSVs will be written
EXTS = (".jpg", ".jpeg", ".png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_metadata_from_relpath(rel_path_parts):
    """
    rel_path_parts: list of path parts relative to the split folder (e.g., ['felidae','Panthera_lineage','lion_head','img1.jpg'])
    """
    family = rel_path_parts[0] if len(rel_path_parts) >= 1 else ""
    subfamily = rel_path_parts[1] if len(rel_path_parts) >= 2 else ""
    species = rel_path_parts[2] if len(rel_path_parts) >= 3 else ""

    filename = rel_path_parts[-1] if len(rel_path_parts) >= 1 else ""
    name_no_ext, ext = os.path.splitext(filename)

    species_keyword_source = species if species else name_no_ext
    species_keyword = species_keyword_source.replace("_", " ").replace("-", " ").strip().title()

    return {
        "family": family,
        "subfamily": subfamily,
        "species": species,
        "filename": filename,
        "name_no_ext": name_no_ext,
        "ext": ext.lower(),
        "species_keyword": species_keyword,
    }

def collect_split_metadata(split_root, split_name):
    """
    Walk split_root/<split_name>/... and collect rows for images.
    """
    rows = []
    base = os.path.join(split_root, split_name)
    if not os.path.isdir(base):
        print(f"[WARN] Split folder not found: {base}")
        return pd.DataFrame(columns=["split","filepath","family","subfamily","species","filename","name_no_ext","ext","species_keyword"])

    for dirpath, _, filenames in os.walk(base):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() not in EXTS:
                continue

            abs_path = os.path.join(dirpath, fn)

            rel_path_from_split = os.path.relpath(abs_path, base)
            rel_parts = rel_path_from_split.split(os.sep)

            meta = extract_metadata_from_relpath(rel_parts)
            rows.append({
                "split": split_name,
                "filepath": os.path.join(split_name, rel_path_from_split).replace("\\", "/"),  # nice portable path
                **meta
            })

    df = pd.DataFrame(rows).sort_values(["family","subfamily","species","filename"]).reset_index(drop=True)
    return df


train_df = collect_split_metadata(SPLIT_ROOT, "train")
val_df = collect_split_metadata(SPLIT_ROOT, "val")
test_df = collect_split_metadata(SPLIT_ROOT, "test")

# Save separately (as requested)
train_csv = os.path.join(OUTPUT_DIR, "metadata_train.csv")
val_csv = os.path.join(OUTPUT_DIR, "metadata_val.csv")
test_csv = os.path.join(OUTPUT_DIR, "metadata_test.csv")

train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)
test_df.to_csv(test_csv, index=False)

print(f"Saved:\n  {train_csv}\n  {val_csv}\n  {test_csv}")
