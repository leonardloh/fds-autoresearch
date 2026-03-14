import pandas as pd
import numpy as np


def load_data():
    iso = pd.read_excel("Dataset/ISO8583_Transaction_Records.xlsx")
    cases = pd.read_excel("Dataset/Case_Creation_Details.xlsx")
    cuh = pd.read_excel("Dataset/Case_Update_History.xlsx")
    return iso, cases, cuh


def create_base_label(iso):
    """Base label: fraud=1 if transaction has a valid CASE_NO (flagged for investigation).

    Label cleaning (flipping confirmed non-fraud to 0) is deferred to
    _clean_labels_as_of() so it can respect point-in-time constraints.
    """
    iso = iso.copy()
    case_no_int = pd.to_numeric(
        iso["CASE_NO"].astype(str).str.strip(), errors="coerce"
    ).fillna(0).astype(int)
    iso["fraud"] = (case_no_int > 0).astype(int)
    return iso


def _get_cleared_cases(cuh, as_of_ts=None):
    """Get case numbers confirmed as non-fraud (status 750).

    If as_of_ts is provided, only use CUH resolutions with timestamp <= as_of_ts
    (point-in-time correctness: training labels must not use future analyst decisions).
    """
    after = cuh[cuh["B4_AFTER_IND"].astype(str).str.strip() == "A"].copy()
    if as_of_ts is not None:
        after_ts = _parse_cre_tms_to_seconds(after["CRE_TMS"])
        after = after[after_ts <= as_of_ts]
    if len(after) == 0:
        return set()
    after_sorted = after.sort_values("CRE_TMS")
    last_per_case = after_sorted.groupby("CASE_NO").last()
    return set(
        int(x) for x in
        last_per_case[last_per_case["CASE_STATUS"] == 750].index.values
    )


def _clean_labels_as_of(df, cuh, as_of_ts=None):
    """Flip confirmed non-fraud (status 750) to fraud=0, respecting as_of_ts.

    as_of_ts=None uses full CUH history (correct for test/evaluation labels).
    as_of_ts=<timestamp> only uses resolutions known by that time (for training).
    """
    df = df.copy()
    cleared_cases = _get_cleared_cases(cuh, as_of_ts=as_of_ts)
    if cleared_cases:
        case_no_int = pd.to_numeric(
            df["CASE_NO"].astype(str).str.strip(), errors="coerce"
        ).fillna(0).astype(int)
        mask = (df["fraud"] == 1) & case_no_int.isin(cleared_cases)
        df.loc[mask, "fraud"] = 0
    return df


def _parse_cre_tms_to_seconds(cre_tms_series):
    """Convert CRE_TMS (YYYYMMDDHHMMSSmmm) to epoch seconds.

    FIX #4: The old code treated CRE_TMS as a plain integer and used
    arithmetic gaps like 10000000 for '1 hour'. That breaks across
    day/month/year boundaries because the DD/MM fields are not base-10
    subdivisions of time. We now parse to proper datetime.
    """
    s = cre_tms_series.astype(str).str.strip().str.zfill(17)
    dt = pd.to_datetime(
        s.str[:14],  # YYYYMMDDHHMMSSs
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )
    # Convert to float seconds since epoch for fast arithmetic
    return (dt - pd.Timestamp("1970-01-01")).dt.total_seconds().fillna(0).values


def engineer_features(df):
    """Engineer features from raw transaction attributes.

    No rule-based system outputs used (removed: all rule hit features).
    No settlement/billing amounts (DE005/DE006 may not be available at auth time).
    Features are inspired by what fraud detection rules check for.
    """
    df = df.copy()

    # === Amount features ===
    df["amount"] = df["DE004"].fillna(0).astype(float)
    df["log_amount"] = np.log1p(df["amount"])
    df["amount_gte_1000"] = (df["amount"] >= 1000).astype(int)
    df["amount_gte_5000"] = (df["amount"] >= 5000).astype(int)
    df["amount_gte_250"] = (df["amount"] >= 250).astype(int)

    # === MCC ===
    df["mcc"] = df["DE018"].fillna(0).astype(int)
    df["mcc_6011"] = (df["mcc"] == 6011).astype(int)
    df["mcc_5411"] = (df["mcc"] == 5411).astype(int)
    df["mcc_5999"] = (df["mcc"] == 5999).astype(int)

    # === Card-present flags ===
    df["is_emv"] = (df["EMV_STAT"].astype(str).str.strip() == "Y").astype(int)
    df["is_contactless"] = (df["EMV_STAT"].astype(str).str.strip() == "C").astype(int)
    df["is_chip"] = (df["CHIP_STAT"].astype(str).str.strip() == "Y").astype(int)
    df["is_fallback"] = (df["FALLBCK_FLG"].astype(str).str.strip() == "Y").astype(int)
    df["is_ecom"] = (df["ECOM_3D_FLG"].astype(str).str.strip() == "Y").astype(int)
    pos_entry = df["DE022"].astype(str).str.strip()
    df["is_magstripe"] = pos_entry.isin(["90", "02"]).astype(int)

    # === CVC2 ===
    df["cvc2_match"] = (df["CVC2_FLG"].astype(str).str.strip() == "M").astype(int)
    df["cvc2_nomatch"] = (df["CVC2_FLG"].astype(str).str.strip() == "N").astype(int)

    # === Currency ===
    df["is_foreign_currency"] = (df["DE049"].astype(str).str.strip() != "458").astype(int)

    # === Hour of day ===
    df["hour"] = pd.to_numeric(
        df["DE012"].astype(str).str.strip().str[:2], errors="coerce"
    ).fillna(0).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # === ECI ===
    eci_str = df["ECI"].astype(str).str.strip()
    df["eci_authenticated"] = (eci_str == "005").astype(int)
    df["eci_nonsecure"] = (eci_str == "008").astype(int)
    df["eci_recurring"] = (eci_str == "002").astype(int)
    df["eci_moto"] = eci_str.isin(["001", "004"]).astype(int)
    df["is_non_3ds"] = ((df["eci_authenticated"] == 0) & (df["is_ecom"] == 0)).astype(int)

    # === Payment network ===
    usr_str = df["USR_ID"].astype(str).str.strip()
    df["is_visa"] = usr_str.str.contains("VIS|VSA", na=False).astype(int)
    df["is_mydebit"] = usr_str.str.contains("MYD", na=False).astype(int)

    # === Misc flags ===
    df["is_non_fiat"] = (df["NON_FIAT_IND"].astype(str).str.strip() == "Y").astype(int)
    df["is_daf"] = (df["DAF_IND"].astype(str).str.strip() == "Y").astype(int)

    # === Timestamp: proper epoch seconds (FIX #4) ===
    df["cre_tms"] = _parse_cre_tms_to_seconds(df["CRE_TMS"])

    # === Categorical string columns (for dummies after split) ===
    df["trx_typ"] = df["TRX_TYP"].astype(str).str.strip()
    df["pos_entry"] = df["DE022"].astype(str).str.strip()
    de003_str = df["DE003"].astype(str).str.zfill(6)
    df["de003_txn_type"] = de003_str.str[:2]

    return df


def add_dummies(df, train_df):
    """Create dummy variables fitted on training data, then aligned to df."""
    df = df.copy()
    for col, prefix in [("trx_typ", "trx"), ("pos_entry", "pos"), ("de003_txn_type", "d3txn")]:
        train_dummies = pd.get_dummies(train_df[col], prefix=prefix)
        dummy_cols = train_dummies.columns.tolist()
        df_dummies = pd.get_dummies(df[col], prefix=prefix)
        df_dummies = df_dummies.reindex(columns=dummy_cols, fill_value=0)
        df = pd.concat([df, df_dummies], axis=1)
    return df


def compute_velocity_features(df):
    """Compute per-card velocity and behavioral features over the full timeline.

    FIX #4: Time windows now use proper seconds (not broken integer arithmetic).
    """
    df = df.copy()
    card_col = "REF_CRD_NO"
    orig_index = df.index.copy()

    vel_cols = [
        "card_txn_count_1h", "card_txn_count_24h", "card_txn_sum_24h",
        "card_txn_count_10min", "card_txn_sum_10min",
        "card_txn_sum_1h",
        "card_max_amt_24h",
        "same_mcc_count_1h", "same_mcc_count_10min",
        "contactless_count_1h", "contactless_sum_1h",
        "distinct_ccy_30min",
        "time_since_last_txn",
        "is_first_txn",
        "new_merchant",
        "distinct_merch_24h",
        "distinct_mcc_24h",
    ]
    for c in vel_cols:
        df[c] = 0.0
    df["is_first_txn"] = 1.0

    # Proper time windows in seconds (FIX #4)
    one_hour = 3600.0
    ten_minutes = 600.0
    thirty_minutes = 1800.0
    twenty_four_hours = 86400.0

    df["_orig_pos"] = np.arange(len(df))
    df = df.sort_values([card_col, "cre_tms"])

    merch_arr = df["MMER_ID"].astype(str).str.strip().values
    mcc_arr = df["mcc"].values
    contactless_arr = df["is_contactless"].values
    ccy_arr = df["DE049"].astype(str).str.strip().values if "DE049" in df.columns else np.full(len(df), "458")

    for card_id in df[card_col].unique():
        mask = df[card_col] == card_id
        card_df = df.loc[mask]
        idxs = card_df.index.tolist()
        times = card_df["cre_tms"].values
        amounts = card_df["amount"].values
        pos_indices = card_df["_orig_pos"].values.astype(int)
        n = len(times)

        cnt_1h = np.zeros(n)
        cnt_24h = np.zeros(n)
        sum_24h = np.zeros(n)
        cnt_10m = np.zeros(n)
        sum_10m = np.zeros(n)
        sum_1h = np.zeros(n)
        max_amt_24h = np.zeros(n)
        same_mcc_1h = np.zeros(n)
        same_mcc_10m = np.zeros(n)
        ctless_cnt_1h = np.zeros(n)
        ctless_sum_1h = np.zeros(n)
        dist_ccy_30m = np.zeros(n)
        time_since = np.zeros(n)
        is_first = np.ones(n)
        new_merch = np.zeros(n)
        dist_merch = np.zeros(n)
        dist_mcc = np.zeros(n)

        seen_merchants = set()
        for i in range(n):
            t = times[i]
            pi = pos_indices[i]
            cur_merch = merch_arr[pi]
            cur_mcc = mcc_arr[pi]

            if cur_merch not in seen_merchants:
                new_merch[i] = 1.0
            seen_merchants.add(cur_merch)

            if i > 0:
                is_first[i] = 0.0
                time_since[i] = t - times[i - 1]

            merch_set_24h = set()
            mcc_set_24h = set()
            ccy_set_30m = set()

            for j in range(i - 1, -1, -1):
                diff = t - times[j]
                if diff > twenty_four_hours:
                    break
                pj = pos_indices[j]

                cnt_24h[i] += 1
                sum_24h[i] += amounts[j]
                if amounts[j] > max_amt_24h[i]:
                    max_amt_24h[i] = amounts[j]
                merch_set_24h.add(merch_arr[pj])
                mcc_set_24h.add(mcc_arr[pj])

                if diff <= one_hour:
                    cnt_1h[i] += 1
                    sum_1h[i] += amounts[j]
                    if mcc_arr[pj] == cur_mcc:
                        same_mcc_1h[i] += 1
                    if contactless_arr[pj]:
                        ctless_cnt_1h[i] += 1
                        ctless_sum_1h[i] += amounts[j]

                if diff <= thirty_minutes:
                    ccy_set_30m.add(ccy_arr[pj])

                if diff <= ten_minutes:
                    cnt_10m[i] += 1
                    sum_10m[i] += amounts[j]
                    if mcc_arr[pj] == cur_mcc:
                        same_mcc_10m[i] += 1

            dist_merch[i] = len(merch_set_24h)
            dist_mcc[i] = len(mcc_set_24h)
            dist_ccy_30m[i] = len(ccy_set_30m)

        for k, idx in enumerate(idxs):
            df.at[idx, "card_txn_count_1h"] = cnt_1h[k]
            df.at[idx, "card_txn_count_24h"] = cnt_24h[k]
            df.at[idx, "card_txn_sum_24h"] = sum_24h[k]
            df.at[idx, "card_txn_count_10min"] = cnt_10m[k]
            df.at[idx, "card_txn_sum_10min"] = sum_10m[k]
            df.at[idx, "card_txn_sum_1h"] = sum_1h[k]
            df.at[idx, "card_max_amt_24h"] = max_amt_24h[k]
            df.at[idx, "same_mcc_count_1h"] = same_mcc_1h[k]
            df.at[idx, "same_mcc_count_10min"] = same_mcc_10m[k]
            df.at[idx, "contactless_count_1h"] = ctless_cnt_1h[k]
            df.at[idx, "contactless_sum_1h"] = ctless_sum_1h[k]
            df.at[idx, "distinct_ccy_30min"] = dist_ccy_30m[k]
            df.at[idx, "time_since_last_txn"] = time_since[k]
            df.at[idx, "is_first_txn"] = is_first[k]
            df.at[idx, "new_merchant"] = new_merch[k]
            df.at[idx, "distinct_merch_24h"] = dist_merch[k]
            df.at[idx, "distinct_mcc_24h"] = dist_mcc[k]

    df = df.sort_values("_orig_pos").drop(columns=["_orig_pos"])
    df.index = orig_index

    df["sum_1h_gte_300"] = (df["card_txn_sum_1h"] >= 300).astype(int)
    df["same_mcc_gte_3_1h"] = (df["same_mcc_count_1h"] >= 3).astype(int)
    df["ctless_cnt_gte_4_1h"] = (df["contactless_count_1h"] >= 4).astype(int)
    df["ctless_sum_gte_250_1h"] = (df["contactless_sum_1h"] >= 250).astype(int)
    df["multi_ccy_30min"] = (df["distinct_ccy_30min"] >= 2).astype(int)
    df["high_charge_500"] = (df["amount"] >= 500).astype(int)

    return df


def _expanding_past_only(df, group_col, value_col, agg_funcs):
    """Compute expanding-window stats using only prior rows (point-in-time safe).

    For each row, statistics are computed from all earlier rows in the same group.
    The current row is excluded via shift(1).
    df must be sorted by cre_tms before calling.
    """
    grouped = df.groupby(group_col)[value_col]
    result = {}
    for name, func in agg_funcs.items():
        if func == "count":
            result[name] = grouped.transform(
                lambda x: x.shift(1).expanding().count()
            ).fillna(0)
        elif func == "mean":
            result[name] = grouped.transform(
                lambda x: x.shift(1).expanding().mean()
            ).fillna(0)
        elif func == "std":
            result[name] = grouped.transform(
                lambda x: x.shift(1).expanding().std()
            ).fillna(0)
        elif func == "max":
            result[name] = grouped.transform(
                lambda x: x.shift(1).expanding().max()
            ).fillna(0)
        elif func == "median":
            result[name] = grouped.transform(
                lambda x: x.shift(1).expanding().median()
            ).fillna(0)
    return result


def compute_aggregation_features(df, train_df):
    """Compute card/merchant/MCC aggregation features with point-in-time correctness.

    Training rows (df is train_df): expanding past-only windows so each transaction
    only sees statistics from temporally earlier transactions — no future leakage.
    Test rows: full train_df statistics (all training data precedes test temporally).
    """
    is_train = (df is train_df)
    df = df.copy()
    card_col = "REF_CRD_NO"

    if is_train:
        # Point-in-time: each row only sees stats from earlier transactions
        sorted_df = df.sort_values("cre_tms")

        # Card-level expanding stats
        card_agg = _expanding_past_only(sorted_df, card_col, "amount", {
            "card_txn_count": "count", "card_txn_mean": "mean",
            "card_txn_std": "std", "card_txn_max": "max",
        })
        for col, vals in card_agg.items():
            sorted_df[col] = vals

        # Merchant-level expanding stats
        sorted_df["_merch_str"] = sorted_df["MMER_ID"].astype(str).str.strip()
        merch_agg = _expanding_past_only(sorted_df, "_merch_str", "amount", {
            "merch_txn_count": "count", "merch_txn_mean": "mean",
        })
        for col, vals in merch_agg.items():
            sorted_df[col] = vals
        sorted_df.drop(columns=["_merch_str"], inplace=True)

        # MCC-level expanding stats
        mcc_agg = _expanding_past_only(sorted_df, "mcc", "amount", {
            "mcc_txn_count": "count", "mcc_txn_mean": "mean",
            "mcc_amount_mean": "mean", "mcc_amount_std": "std",
            "mcc_amount_median": "median",
        })
        for col, vals in mcc_agg.items():
            sorted_df[col] = vals

        # Restore original index order
        df = sorted_df.sort_index()

    else:
        # Test data: all train_df is temporally before test — batch stats are correct
        card_stats = train_df.groupby(card_col)["amount"].agg(
            card_txn_count="count", card_txn_mean="mean",
            card_txn_std="std", card_txn_max="max",
        ).fillna(0)
        df = df.merge(card_stats, left_on=card_col, right_index=True, how="left")
        for c in ["card_txn_count", "card_txn_mean", "card_txn_std", "card_txn_max"]:
            df[c] = df[c].fillna(0)

        m_col = "_merch_str"
        df[m_col] = df["MMER_ID"].astype(str).str.strip()
        train_m = train_df.copy()
        train_m[m_col] = train_m["MMER_ID"].astype(str).str.strip()
        merch_stats = train_m.groupby(m_col)["amount"].agg(
            merch_txn_count="count", merch_txn_mean="mean",
        ).fillna(0)
        df = df.merge(merch_stats, left_on=m_col, right_index=True, how="left")
        for c in ["merch_txn_count", "merch_txn_mean"]:
            df[c] = df[c].fillna(0)
        df.drop(columns=[m_col], inplace=True)

        mcc_stats = train_df.groupby("mcc")["amount"].agg(
            mcc_txn_count="count", mcc_txn_mean="mean",
        ).fillna(0)
        df = df.merge(mcc_stats, left_on="mcc", right_index=True, how="left")
        for c in ["mcc_txn_count", "mcc_txn_mean"]:
            df[c] = df[c].fillna(0)

        mcc_amount_stats = train_df.groupby("mcc")["amount"].agg(
            mcc_amount_mean="mean", mcc_amount_std="std",
            mcc_amount_median="median",
        ).fillna(0)
        df = df.merge(mcc_amount_stats, left_on="mcc", right_index=True, how="left")
        for c in ["mcc_amount_mean", "mcc_amount_std", "mcc_amount_median"]:
            df[c] = df[c].fillna(0)

    # Derived features (same for both paths)
    df["amount_vs_card_mean"] = df["amount"] / (df["card_txn_mean"] + 1)
    df["amount_vs_card_max"] = df["amount"] / (df["card_txn_max"] + 1)
    df["amount_zscore"] = (df["amount"] - df["card_txn_mean"]) / (df["card_txn_std"] + 1e-8)
    df["amount_vs_mcc_mean"] = df["amount"] / (df["mcc_amount_mean"] + 1)
    df["amount_mcc_zscore"] = (df["amount"] - df["mcc_amount_mean"]) / (df["mcc_amount_std"] + 1e-8)
    df["amount_vs_mcc_median"] = df["amount"] / (df["mcc_amount_median"] + 1)
    df["card_amount_cv"] = df["card_txn_std"] / (df["card_txn_mean"] + 1e-8)

    # Interaction features
    df["amount_x_foreign"] = df["amount"] * df["is_foreign_currency"]
    df["amount_x_night"] = df["amount"] * df["is_night"]
    df["amount_x_ecom"] = df["amount"] * df["is_ecom"]
    df["velocity_x_amount"] = df["card_txn_count_24h"] * df["log_amount"]
    df["velocity_10m_x_amount"] = df["card_txn_count_10min"] * df["log_amount"]

    return df


def get_feature_columns(df):
    exclude = {
        "fraud", "CASE_NO", "CRE_TMS", "UPD_TMS", "PGM_ID", "UPD_UID",
        "PRIMAP", "SECMAP", "FI_CDE", "REF_CRD_NO", "CRD_BRN",
        "DE002L", "DE002", "DE003", "DE004", "DE005", "DE006", "DE007",
        "DE011", "DE012", "DE013", "DE014", "DE015", "DE018", "DE022",
        "DE032L", "DE032", "DE033L", "DE033", "DE037", "DE038", "DE039",
        "DE041", "MMER_ID", "DE042", "DE043", "DE048L", "DE048",
        "DE049", "DE050", "DE051", "DE061L", "DE061", "DE121L", "DE121",
        "CIF_NO", "TRANS_AUTH_STAT", "REF_NO", "EMV_STAT", "TRX_TYP",
        "USR_ID", "CHIP_STAT", "FALLBCK_FLG", "CVC2_FLG", "ECOM_3D_FLG",
        "ECI", "TTI", "NON_FIAT_IND", "DAF_IND", "URN",
        "LCL_TM_DE12", "LCL_DT_DE13", "COREBANK_CIF_NO",
        "trx_typ", "pos_entry", "de003_txn_type", "cre_tms",
    }
    return [c for c in df.columns if c not in exclude]


def prepare():
    iso, cases, cuh = load_data()
    iso = create_base_label(iso)
    df = engineer_features(iso)

    df = df.sort_values("cre_tms").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    split_ts = df.iloc[split_idx - 1]["cre_tms"]

    df = compute_velocity_features(df)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Point-in-time label cleaning: training labels only use analyst
    # decisions known at the split timestamp, not future resolutions.
    # Test labels use full history (ground truth for evaluation).
    train_df = _clean_labels_as_of(train_df, cuh, as_of_ts=split_ts)
    test_df = _clean_labels_as_of(test_df, cuh, as_of_ts=None)

    train_df = add_dummies(train_df, train_df)
    test_df = add_dummies(test_df, train_df)

    train_df = compute_aggregation_features(train_df, train_df)
    test_df = compute_aggregation_features(test_df, train_df)

    feature_cols = get_feature_columns(train_df)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["fraud"].values
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["fraud"].values

    return X_train, X_test, y_train, y_test, feature_cols


def prepare_prioritization():
    """Prepare data with case-level metadata for prioritization evaluation."""
    iso, cases, cuh = load_data()
    iso = create_base_label(iso)
    df = engineer_features(iso)

    df = df.sort_values("cre_tms").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    split_ts = df.iloc[split_idx - 1]["cre_tms"]

    df = compute_velocity_features(df)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Point-in-time label cleaning
    train_df = _clean_labels_as_of(train_df, cuh, as_of_ts=split_ts)
    test_df = _clean_labels_as_of(test_df, cuh, as_of_ts=None)

    train_df = add_dummies(train_df, train_df)
    test_df = add_dummies(test_df, train_df)

    train_df = compute_aggregation_features(train_df, train_df)
    test_df = compute_aggregation_features(test_df, train_df)

    feature_cols = get_feature_columns(train_df)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["fraud"].values
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["fraud"].values

    # Case-level info for prioritization eval
    test_case_nos = pd.to_numeric(
        test_df["CASE_NO"].astype(str).str.strip(), errors="coerce"
    ).fillna(0).astype(int).values

    # Confirmed fraud/non-fraud from full CUH history (ground truth for eval)
    after = cuh[cuh["B4_AFTER_IND"].astype(str).str.strip() == "A"]
    last_per_case = after.sort_values("CRE_TMS").groupby("CASE_NO").last()
    confirmed_fraud = set(int(x) for x in
        last_per_case[last_per_case["CASE_STATUS"] == 700].index.values)
    confirmed_nf = set(int(x) for x in
        last_per_case[last_per_case["CASE_STATUS"] == 750].index.values)

    return (X_train, X_test, y_train, y_test, feature_cols,
            test_case_nos, confirmed_fraud, confirmed_nf)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_cols = prepare()
    print(f"Features: {len(feature_cols)}")
    print(f"Train: {X_train.shape}, fraud rate: {y_train.mean():.4f}")
    print(f"Test:  {X_test.shape}, fraud rate: {y_test.mean():.4f}")
    print(f"Train fraud count: {y_train.sum()}")
    print(f"Test fraud count: {y_test.sum()}")
