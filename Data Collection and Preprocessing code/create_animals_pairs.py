import pandas as pd
from itertools import combinations
import os
import re

def normalize_path(p: str) -> str:
    return p.replace("\\", "/")


def origin_id_from_path(filepath: str) -> str:
    """
    Make an 'origin id' by removing a trailing _aug<number> before the extension.
    Examples:
      train/canis/fox/img123.jpg         -> train/canis/fox/img123
    """
    filepath = normalize_path(filepath)
    root, ext = os.path.splitext(filepath)
    # strip one trailing _aug<digits>
    root = re.sub(r"_aug\d+$", "", root)
    return root  # keep full path (without extension) to avoid collisions


def clean_species(species_name):
    return species_name.replace(" Head", "")


def load_taxonomy_from_metadata(df):
    df['species'] = df['species_keyword'].apply(clean_species)
    df = df.drop_duplicates(subset='species')

    taxonomy = {}
    for _, row in df.iterrows():
        species = row['species']
        taxonomy[species] = {
            "subfamily": str(row["subfamily"]),
            "family": str(row["family"]),
        }
    return taxonomy

def get_similarity_score(animal1, animal2, taxonomy):
    animal1 = clean_species(animal1)
    animal2 = clean_species(animal2)

    if animal1 not in taxonomy or animal2 not in taxonomy:
        return None  # Or raise an error if desired

    if animal1 == animal2:
        return 4  # Very High similarity

    tax1 = taxonomy[animal1]
    tax2 = taxonomy[animal2]
    sub1, sub2 = tax1['subfamily'], tax2['subfamily']
    fam1, fam2 = tax1['family'], tax2['family']

    if sub1 == sub2:
        return 3  # High similarity
    if fam1 == fam2:
        return 2  # Medium similarity

    # Cross-family special cases
    pair = {fam1, fam2}
    if pair == {"felidae", "caninae"}:
        return 1  # Medium-Low
    if pair == {"bovidae", "cervidae"}:
        return 1

    return 0.0  # Low

def main():
    metadata = pd.read_csv("metadata_csv\metadata_train.csv")
    # normalize paths and compute origin ids
    metadata["filepath"] = metadata["filepath"].apply(normalize_path)
    metadata["origin_id"] = metadata["filepath"].apply(origin_id_from_path)

    taxonomy = load_taxonomy_from_metadata(metadata)

    pairs = []
    count = 0
    MAX_PAIRS = 100000

    for i, j in combinations(metadata.index, 2):
        row1 = metadata.loc[i]
        row2 = metadata.loc[j]

        # 1) skip exact same file
        if row1["filepath"] == row2["filepath"]:
            continue

        # 2) skip pairs that come from the same origin image
        if row1["origin_id"] == row2["origin_id"]:
            continue

        score = get_similarity_score(row1["species_keyword"], row2["species_keyword"], taxonomy)
        pairs.append((row1["filepath"], row2["filepath"], score))

        # count += 1  #for the train set
        # if count >= MAX_PAIRS:
        #     break

    pairs_df = pd.DataFrame(pairs, columns=["image1", "image2", "similarity_class"])
    os.makedirs("data_preprocessing", exist_ok=True)
    out_csv = "data_preprocessing/train_animal_similarity_pairs.csv"
    pairs_df.to_csv(out_csv, index=False)
    print(f"Saved pairs to: {out_csv}  (total pairs: {len(pairs_df)})")


if __name__ == "__main__":
    main()
