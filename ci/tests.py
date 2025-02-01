import pandas as pd
import pytest
import numpy as np

from .utils import cmor_tables, tables


@pytest.mark.parametrize("dreq_file", tables)
def test_no_duplicates(dreq_file):
    """
    Test that all entries in the data request file are unique.

    Parameters:
    dreq_file (str): Path to the data request file.

    Asserts:
    - All entries in the data request file are unique.
    - All entries, except for the 'priority' column, are unique.
    """
    dreq_df = pd.read_csv(dreq_file)
    assert dreq_df.loc[dreq_df.duplicated()].empty
    assert (
        dreq_df.drop(columns="priority")
        .loc[dreq_df.drop(columns="priority").duplicated()]
        .empty
    )


@pytest.mark.parametrize("dreq_file", tables)
def test_all_in_cmor_tables(dreq_file):
    """
    Test that all entries in the data request file are registered in the CMOR tables.

    Parameters:
    dreq_file (str): Path to the data request file.

    Asserts:
    - All entries in the data request file are registered in the CMOR tables.
    """
    print(f"Checking if all entries exist in cmor table: {dreq_file}")
    cmor_df = pd.read_csv(cmor_tables)
    dreq_df = pd.read_csv(dreq_file).drop(columns="priority")
    # dreq_df.loc[0, "out_name"] = "xxx"
    # merge two dataFrames and add indicator column
    all_df = pd.merge(
        dreq_df, cmor_df, on=dreq_df.columns.to_list(), how="left", indicator="exists"
    )
    all_df["exists"] = np.where(all_df.exists == "both", True, False)
    if not all_df.exists.all():
        print(
            f"!!!! There are some variables from {dreq_file} that do not exist in cmor tables !!!!"
        )
        print(print(all_df[~all_df.exists]))

    assert all_df.exists.all()
    print(f"Seems ok: {dreq_file}")


def test_all_positive_attrs_set():
    """
    Test that all positive attributes are set correctly in the CMOR tables.

    Asserts:
    - All entries with standard names indicating an upward direction have 'positive' set to 'up'.
    - All entries with standard names indicating a downward direction have 'positive' set to 'down'.
    """
    up = ["outgoing", "upward", "upwelling"]
    down = ["incoming", "downward", "downwelling", "sinking"]
    df = pd.read_csv(cmor_tables)

    ups = df.loc[df.standard_name.str.contains("|".join(up), case=False)]
    downs = df.loc[df.standard_name.str.contains("|".join(down), case=False)]

    assert all(ups.positive == "up")
    assert all(downs.positive == "down")
