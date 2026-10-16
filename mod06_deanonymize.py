import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
  
    # Exact match on the three quasi-identifiers
    merged = pd.merge(
        anon_df,
        aux_df,
        on=["age", "gender", "zip3"],
        how="inner"
    )

    # Keep only combinations that appear exactly once
    counts = merged.groupby(["age", "gender", "zip3"]).size()
    unique_keys = counts[counts == 1].index

    unique_matches = (
        merged
        .set_index(["age", "gender", "zip3"])
        .loc[unique_keys]
        .reset_index()
    )

    # Return only the required columns
    # (change "name" if the auxiliary column has a different name)
    result = unique_matches[["anon_id", "name"]].rename(
        columns={"name": "matched_name"}
    )
    return result


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if len(anon_df) == 0:
        return 0.0
    return len(matches_df) / len(anon_df)