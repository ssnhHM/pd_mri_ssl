import re

import pandas as pd


FIELD_COLUMNS = {
    "t1": (
        "t1_MagneticFieldStrength_mri",
        "t1_MagneticFieldStrength",
        "MagneticFieldStrength_t1",
        "MagneticFieldStrength_t1_inv",
    ),
    "t2": (
        "t2_MagneticFieldStrength_mri",
        "t2_MagneticFieldStrength",
        "MagneticFieldStrength_t2",
        "MagneticFieldStrength_t2_inv",
    ),
}


def is_3t(value):
    text = str(value).strip().lower().replace("tesla", "t")
    # Use the first recognized field value, as in the original cohort filter.
    match = re.search(r"(?:^|[^0-9])(1\.5|1\.0|1|3\.0|3)(?:[^0-9]|$)", text)
    return match is not None and float(match.group(1)) == 3.0


def filter_3t(df, inputs):
    sources = set()
    for column in inputs:
        name = column.lower()
        if "ratio" in name:
            sources.update(("t1", "t2"))
        elif "t2" in name:
            sources.add("t2")
        elif "t1" in name or "jac" in name:
            sources.add("t1")
        else:
            raise ValueError(f"Cannot infer T1/T2 field-strength source for {column!r}")

    columns = {str(column).lower(): column for column in df.columns}
    keep = pd.Series(True, index=df.index)
    for source in sorted(sources):
        aliases = FIELD_COLUMNS[source]
        column = next((columns[name.lower()] for name in aliases if name.lower() in columns), None)
        if column is None:
            raise ValueError(f"3T filtering requires a {source.upper()} field column: {aliases}")
        keep &= df[column].map(is_3t)
    return df.loc[keep].copy()
