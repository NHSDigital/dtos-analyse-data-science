import pandas as pd


def add_uptake_columns(df):
    df["date_of_as"] = pd.to_datetime(df["date_of_as"], errors="coerce")
    df["date_of_foa"] = pd.to_datetime(df["date_of_foa"], errors="coerce")

    df["uptake_six_mth_threshold"] = (
        ((df["date_of_as"] - df["date_of_foa"]) <= pd.Timedelta(days=183))
        & df["date_of_as"].notna()
        & df["date_of_foa"].notna()
    )
    df["uptake_six_mth_threshold"] = df["uptake_six_mth_threshold"].astype("boolean")
    df.loc[df["date_of_foa"].isna(), "uptake_six_mth_threshold"] = pd.NA

    return df


def add_round_length_columns(df):
    df = df.sort_values(["nhs_number", "date_of_foa"], ascending=False)
    df["prior_date_of_as"] = df.groupby("nhs_number")["date_of_as"].shift(-1)
    df["prior_date_of_foa"] = df.groupby("nhs_number")["date_of_foa"].shift(-1)
    df["prior_episode_date"] = df["prior_date_of_as"].combine_first(
        df["prior_date_of_foa"]
    )

    df["round_length_three_yr_threshold"] = (
        ((df["date_of_foa"] - df["prior_episode_date"]) <= pd.Timedelta(days=3 * 365))
        & df["date_of_foa"].notna()
        & df["prior_episode_date"].notna()
    )

    df["round_length_three_yr_threshold"] = df[
        "round_length_three_yr_threshold"
    ].astype("boolean")
    df.loc[df["date_of_foa"].isna(), "round_length_three_yr_threshold"] = pd.NA
    df.loc[df["prior_episode_date"].isna(), "round_length_three_yr_threshold"] = pd.NA

    return df
