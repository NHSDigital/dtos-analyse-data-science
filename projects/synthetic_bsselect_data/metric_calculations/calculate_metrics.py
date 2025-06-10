import pandas as pd


def calculate_uptake(df, aggregation, end_date, start_date=None):
    if start_date is None:
        start_date = pd.to_datetime(end_date) - pd.DateOffset(years=1)
    mask = (df["date_of_foa"] >= pd.to_datetime(start_date)) & (
        df["date_of_foa"] <= pd.to_datetime(end_date)
    )
    filtered = df.loc[mask]
    filtered = filtered[filtered["uptake_six_mth_threshold"].notna()]
    if len(filtered) == 0:
        raise ValueError("No data found to perform calculation")
    uptake = filtered.groupby(aggregation)["uptake_six_mth_threshold"].mean() * 100

    return uptake


def calculate_round_length(df, aggregation, end_date, start_date=None):
    if start_date is None:
        start_date = pd.to_datetime(end_date) - pd.DateOffset(years=1)
    mask = (df["date_of_foa"] >= pd.to_datetime(start_date)) & (
        df["date_of_foa"] <= pd.to_datetime(end_date)
    )
    filtered = df.loc[mask]
    filtered = filtered[filtered["round_length_three_yr_threshold"].notna()]
    if len(filtered) == 0:
        raise ValueError("No data found to perform calculation")
    round_length = (
        filtered.groupby(aggregation)["round_length_three_yr_threshold"].mean() * 100
    )

    return round_length
