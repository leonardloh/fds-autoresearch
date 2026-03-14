import pandas as pd
import numpy as np


def load_data():
    iso = pd.read_excel("Dataset/ISO8583_Transaction_Records.xlsx")
    cases = pd.read_excel("Dataset/Case_Creation_Details.xlsx")
    cuh = pd.read_excel("Dataset/Case_Update_History.xlsx")
    return iso, cases, cuh


def create_label(iso, cases, cuh):
    """Label: fraud=1 if transaction has a valid CASE_NO (flagged for investigation).

    Clean labels using Case_Update_History:
    - Status 750 = confirmed non-fraud (analyst reviewed, normal transaction)
    - These cases are flipped to fraud=0.
    """
    iso = iso.copy()
    case_no_int = pd.to_numeric(
        iso["CASE_NO"].astype(str).str.strip(), errors="coerce"
    ).fillna(0).astype(int)
    iso["fraud"] = (case_no_int > 0).astype(int)

    # Clean labels: remove confirmed non-fraud (status 750) from positives
    after = cuh[cuh["B4_AFTER_IND"].astype(str).str.strip() == "A"]
    after_sorted = after.sort_values("CRE_TMS")
    last_per_case = after_sorted.groupby("CASE_NO").last()
    cleared_cases = set(
        last_per_case[last_per_case["CASE_STATUS"] == 750].index.values
    )
    if cleared_cases:
        iso.loc[
            iso["fraud"] == 1,
            "fraud"
        ] = iso.loc[iso["fraud"] == 1, "CASE_NO"].apply(
            lambda x: 0 if int(
                pd.to_numeric(str(x).strip(), errors="coerce") or 0
            ) in cleared_cases else 1
        )

    return iso


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
    # Amount threshold features (inspired by RAD rules checking amount thresholds)
    df["amount_gte_1000"] = (df["amount"] >= 1000).astype(int)
    df["amount_gte_5000"] = (df["amount"] >= 5000).astype(int)
    df["amount_gte_250"] = (df["amount"] >= 250).astype(int)

    # === MCC ===
    df["mcc"] = df["DE018"].fillna(0).astype(int)
    # High-risk MCC flags (inspired by RAD rules targeting specific MCCs)
    df["mcc_6011"] = (df["mcc"] == 6011).astype(int)  # ATM/cash
    df["mcc_5411"] = (df["mcc"] == 5411).astype(int)  # grocery/supermarket
    df["mcc_5999"] = (df["mcc"] == 5999).astype(int)  # misc retail

    # === Card-present flags (pre-authorization attributes) ===
    df["is_emv"] = (df["EMV_STAT"].astype(str).str.strip() == "Y").astype(int)
    df["is_contactless"] = (df["EMV_STAT"].astype(str).str.strip() == "C").astype(int)
    df["is_chip"] = (df["CHIP_STAT"].astype(str).str.strip() == "Y").astype(int)
    df["is_fallback"] = (df["FALLBCK_FLG"].astype(str).str.strip() == "Y").astype(int)
    df["is_ecom"] = (df["ECOM_3D_FLG"].astype(str).str.strip() == "Y").astype(int)
    # Magstripe detection (inspired by TSRAD13: magstripe transaction rule)
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
    # Non-3DS flag (inspired by RAD rules: XYZRAD1A, XYZRAD1B, RAD01)
    df["is_non_3ds"] = ((df["eci_authenticated"] == 0) & (df["is_ecom"] == 0)).astype(int)

    # === Payment network ===
    usr_str = df["USR_ID"].astype(str).str.strip()
    df["is_visa"] = usr_str.str.contains("VIS|VSA", na=False).astype(int)
    df["is_mydebit"] = usr_str.str.contains("MYD", na=False).astype(int)

    # === Misc flags ===
    df["is_non_fiat"] = (df["NON_FIAT_IND"].astype(str).str.strip() == "Y").astype(int)
    df["is_daf"] = (df["DAF_IND"].astype(str).str.strip() == "Y").astype(int)

    # === Timestamp for temporal operations ===
    df["cre_tms"] = pd.to_numeric(df["CRE_TMS"], errors="coerce").fillna(0)

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
        # Get categories from training data only
        train_dummies = pd.get_dummies(train_df[col], prefix=prefix)
        dummy_cols = train_dummies.columns.tolist()

        # Create dummies for df and align to training categories
        df_dummies = pd.get_dummies(df[col], prefix=prefix)
        df_dummies = df_dummies.reindex(columns=dummy_cols, fill_value=0)
        df = pd.concat([df, df_dummies], axis=1)

    return df


def compute_velocity_features(df):
    """Compute per-card velocity and behavioral features over the full timeline.

    Features are engineered from raw transaction data, replicating the underlying
    signals that CC/RAD detection rules check (velocity, volume, same-MCC repeats,
    contactless patterns, currency diversity) WITHOUT using rule outputs.

    Preserves the original dataframe index/order. Only looks backward in time
    (no future leakage). Data must already be sorted by cre_tms.
    """
    df = df.copy()
    card_col = "REF_CRD_NO"
    orig_index = df.index.copy()

    # Initialize all velocity columns
    vel_cols = [
        "card_txn_count_1h", "card_txn_count_24h", "card_txn_sum_24h",
        "card_txn_count_10min", "card_txn_sum_10min",
        "card_txn_sum_1h",       # TSCC9/QACC9: volume in 1h (RM300 threshold)
        "card_max_amt_24h",
        "same_mcc_count_1h",     # TSCC1/QACC1: same MCC repeat in 1h
        "same_mcc_count_10min",  # QACC5/TSCC5: same MCC repeat in 10min
        "contactless_count_1h",  # TSCC6: contactless velocity in 1h
        "contactless_sum_1h",    # TSCC10: contactless volume in 1h
        "distinct_ccy_30min",    # TSCC13/QACC13: different currency in 30min
        "time_since_last_txn",
        "is_first_txn",
        "new_merchant",
        "distinct_merch_24h",
        "distinct_mcc_24h",
    ]
    for c in vel_cols:
        df[c] = 0.0
    df["is_first_txn"] = 1.0

    one_hour = 10000000  # CRE_TMS format: YYYYMMDDHHMMSSmmm
    ten_minutes = 1000000
    thirty_minutes = 3000000
    twenty_four_hours = 240000000

    # Sort by card + time for velocity computation, but track original position
    df["_orig_pos"] = np.arange(len(df))
    df = df.sort_values([card_col, "cre_tms"])

    # Pre-extract arrays for per-txn attributes
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

        # Standard velocity/volume arrays
        cnt_1h = np.zeros(n)
        cnt_24h = np.zeros(n)
        sum_24h = np.zeros(n)
        cnt_10m = np.zeros(n)
        sum_10m = np.zeros(n)
        sum_1h = np.zeros(n)
        max_amt_24h = np.zeros(n)
        # Rule-inspired arrays
        same_mcc_1h = np.zeros(n)
        same_mcc_10m = np.zeros(n)
        ctless_cnt_1h = np.zeros(n)
        ctless_sum_1h = np.zeros(n)
        dist_ccy_30m = np.zeros(n)
        # Behavioral arrays
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
                    # Same MCC as current txn within 1h
                    if mcc_arr[pj] == cur_mcc:
                        same_mcc_1h[i] += 1
                    # Contactless within 1h
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

    # Restore original order
    df = df.sort_values("_orig_pos").drop(columns=["_orig_pos"])
    df.index = orig_index

    # Derived threshold features (inspired by rule thresholds, computed from raw data)
    df["sum_1h_gte_300"] = (df["card_txn_sum_1h"] >= 300).astype(int)
    df["same_mcc_gte_3_1h"] = (df["same_mcc_count_1h"] >= 3).astype(int)
    df["ctless_cnt_gte_4_1h"] = (df["contactless_count_1h"] >= 4).astype(int)
    df["ctless_sum_gte_250_1h"] = (df["contactless_sum_1h"] >= 250).astype(int)
    df["multi_ccy_30min"] = (df["distinct_ccy_30min"] >= 2).astype(int)
    df["high_charge_500"] = (df["amount"] >= 500).astype(int)

    return df


def compute_aggregation_features(df, train_df):
    """Compute card/merchant/MCC aggregation + target encoding from train only."""
    df = df.copy()
    card_col = "REF_CRD_NO"

    # Card-level stats
    card_stats = train_df.groupby(card_col)["amount"].agg(
        card_txn_count="count", card_txn_mean="mean",
        card_txn_std="std", card_txn_max="max",
    ).fillna(0)
    df = df.merge(card_stats, left_on=card_col, right_index=True, how="left")
    for c in ["card_txn_count", "card_txn_mean", "card_txn_std", "card_txn_max"]:
        df[c] = df[c].fillna(0)
    df["amount_vs_card_mean"] = df["amount"] / (df["card_txn_mean"] + 1)
    df["amount_vs_card_max"] = df["amount"] / (df["card_txn_max"] + 1)
    df["amount_zscore"] = (df["amount"] - df["card_txn_mean"]) / (df["card_txn_std"] + 1e-8)

    # Merchant-level stats
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

    # MCC-level stats
    mcc_stats = train_df.groupby("mcc")["amount"].agg(
        mcc_txn_count="count", mcc_txn_mean="mean",
    ).fillna(0)
    df = df.merge(mcc_stats, left_on="mcc", right_index=True, how="left")
    for c in ["mcc_txn_count", "mcc_txn_mean"]:
        df[c] = df[c].fillna(0)

    # === Target-encoded risk (smoothed, train-only) ===
    global_mean = train_df["fraud"].mean()
    smoothing = 10

    for group_col, risk_name in [("mcc", "mcc_risk"), (card_col, "card_risk")]:
        grp = train_df.groupby(group_col)["fraud"].agg(["mean", "count"])
        grp[risk_name] = (
            (grp["count"] * grp["mean"] + smoothing * global_mean)
            / (grp["count"] + smoothing)
        )
        df = df.merge(grp[[risk_name]], left_on=group_col, right_index=True, how="left")
        df[risk_name] = df[risk_name].fillna(global_mean)

    # Merchant risk
    train_m2 = train_df.copy()
    train_m2["_m2"] = train_m2["MMER_ID"].astype(str).str.strip()
    df["_m2"] = df["MMER_ID"].astype(str).str.strip()
    mf = train_m2.groupby("_m2")["fraud"].agg(["mean", "count"])
    mf["merch_risk"] = (
        (mf["count"] * mf["mean"] + smoothing * global_mean)
        / (mf["count"] + smoothing)
    )
    df = df.merge(mf[["merch_risk"]], left_on="_m2", right_index=True, how="left")
    df["merch_risk"] = df["merch_risk"].fillna(global_mean)
    df.drop(columns=["_m2"], inplace=True)

    # === Amount deviation from MCC norm (from training data) ===
    mcc_amount_stats = train_df.groupby("mcc")["amount"].agg(
        mcc_amount_mean="mean", mcc_amount_std="std",
        mcc_amount_median="median",
    ).fillna(0)
    df = df.merge(mcc_amount_stats, left_on="mcc", right_index=True, how="left")
    for c in ["mcc_amount_mean", "mcc_amount_std", "mcc_amount_median"]:
        df[c] = df[c].fillna(0)
    df["amount_vs_mcc_mean"] = df["amount"] / (df["mcc_amount_mean"] + 1)
    df["amount_mcc_zscore"] = (df["amount"] - df["mcc_amount_mean"]) / (df["mcc_amount_std"] + 1e-8)
    df["amount_vs_mcc_median"] = df["amount"] / (df["mcc_amount_median"] + 1)

    # === Card amount variability features ===
    df["card_amount_cv"] = df["card_txn_std"] / (df["card_txn_mean"] + 1e-8)  # coefficient of variation

    # === POS entry risk (from training data) ===
    pos_risk_data = train_df.copy()
    pos_risk_data["_pos"] = pos_risk_data["DE022"].astype(str).str.strip() if "DE022" in pos_risk_data.columns else ""
    df["_pos"] = df["DE022"].astype(str).str.strip() if "DE022" in df.columns else ""
    if "_pos" in pos_risk_data.columns:
        pos_fraud = pos_risk_data.groupby("_pos")["fraud"].agg(["mean", "count"])
        pos_fraud["pos_risk"] = (
            (pos_fraud["count"] * pos_fraud["mean"] + smoothing * global_mean)
            / (pos_fraud["count"] + smoothing)
        )
        df = df.merge(pos_fraud[["pos_risk"]], left_on="_pos", right_index=True, how="left")
        df["pos_risk"] = df["pos_risk"].fillna(global_mean)
    df.drop(columns=["_pos"], errors="ignore", inplace=True)

    # Interaction features
    df["amount_x_foreign"] = df["amount"] * df["is_foreign_currency"]
    df["amount_x_night"] = df["amount"] * df["is_night"]
    df["amount_x_ecom"] = df["amount"] * df["is_ecom"]
    df["amount_x_mcc_risk"] = df["amount"] * df["mcc_risk"]
    df["amount_x_card_risk"] = df["amount"] * df["card_risk"]
    df["log_amount_x_mcc_risk"] = df["log_amount"] * df["mcc_risk"]
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
    iso = create_label(iso, cases, cuh)
    df = engineer_features(iso)

    # Time-based split
    df = df.sort_values("cre_tms").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)

    # Compute velocity on full timeline (so test sees training history)
    # df is already sorted by cre_tms; velocity preserves this order
    df = compute_velocity_features(df)

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Dummies fitted on training data only
    train_df = add_dummies(train_df, train_df)
    test_df = add_dummies(test_df, train_df)

    # Aggregation + target encoding from training data only
    train_df = compute_aggregation_features(train_df, train_df)
    test_df = compute_aggregation_features(test_df, train_df)

    feature_cols = get_feature_columns(train_df)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["fraud"].values
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["fraud"].values

    return X_train, X_test, y_train, y_test, feature_cols


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_cols = prepare()
    print(f"Features: {len(feature_cols)}")
    print(f"Train: {X_train.shape}, fraud rate: {y_train.mean():.4f}")
    print(f"Test:  {X_test.shape}, fraud rate: {y_test.mean():.4f}")
    print(f"Train fraud count: {y_train.sum()}")
    print(f"Test fraud count: {y_test.sum()}")
