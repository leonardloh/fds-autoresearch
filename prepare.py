import pandas as pd
import numpy as np


def load_data():
    iso = pd.read_excel("Dataset/ISO8583_Transaction_Records.xlsx")
    cases = pd.read_excel("Dataset/Case_Creation_Details.xlsx")
    return iso, cases


def create_label(iso, cases):
    """Label: fraud=1 if transaction has a valid CASE_NO (flagged for investigation).

    Using all cased transactions as positive class (4415 txns) since
    confirmed-only (70 txns) is too few for temporal evaluation.
    """
    iso = iso.copy()
    case_no_int = pd.to_numeric(
        iso["CASE_NO"].astype(str).str.strip(), errors="coerce"
    ).fillna(0).astype(int)
    iso["fraud"] = (case_no_int > 0).astype(int)
    return iso


def engineer_features(df):
    """Engineer basic features that don't require train-only computation.

    Response code and auth status REMOVED (leakage: they encode fraud system output).
    Categorical dummies are NOT created here -- done after split to prevent leakage.
    """
    df = df.copy()

    # === Amount features ===
    df["amount"] = df["DE004"].fillna(0).astype(float)
    df["log_amount"] = np.log1p(df["amount"])

    # === MCC ===
    df["mcc"] = df["DE018"].fillna(0).astype(int)

    # === Card-present flags (pre-authorization attributes) ===
    df["is_emv"] = (df["EMV_STAT"].astype(str).str.strip() == "Y").astype(int)
    df["is_contactless"] = (df["EMV_STAT"].astype(str).str.strip() == "C").astype(int)
    df["is_chip"] = (df["CHIP_STAT"].astype(str).str.strip() == "Y").astype(int)
    df["is_fallback"] = (df["FALLBCK_FLG"].astype(str).str.strip() == "Y").astype(int)
    df["is_ecom"] = (df["ECOM_3D_FLG"].astype(str).str.strip() == "Y").astype(int)

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

    # === Payment network ===
    usr_str = df["USR_ID"].astype(str).str.strip()
    df["is_visa"] = usr_str.str.contains("VIS|VSA", na=False).astype(int)
    df["is_mydebit"] = usr_str.str.contains("MYD", na=False).astype(int)

    # === Misc flags ===
    df["is_non_fiat"] = (df["NON_FIAT_IND"].astype(str).str.strip() == "Y").astype(int)
    df["is_daf"] = (df["DAF_IND"].astype(str).str.strip() == "Y").astype(int)

    # === Settlement / billing amount features ===
    df["settle_amount"] = df["DE005"].fillna(0).astype(float)
    df["billing_amount"] = df["DE006"].fillna(0).astype(float)
    df["amount_billing_diff"] = df["amount"] - df["billing_amount"]
    df["amount_billing_ratio"] = df["amount"] / (df["billing_amount"] + 1)

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
    """Compute per-card velocity features over the full timeline.

    Preserves the original dataframe index/order. Only looks backward in time
    (no future leakage). Data must already be sorted by cre_tms.
    """
    df = df.copy()
    card_col = "REF_CRD_NO"
    orig_index = df.index.copy()

    # Initialize velocity columns
    df["card_txn_count_1h"] = 0.0
    df["card_txn_count_24h"] = 0.0
    df["card_txn_sum_24h"] = 0.0

    one_hour = 10000000  # CRE_TMS format: YYYYMMDDHHMMSSmmm
    twenty_four_hours = 240000000

    # Sort by card + time for velocity computation, but track original position
    df["_orig_pos"] = np.arange(len(df))
    df = df.sort_values([card_col, "cre_tms"])

    for card_id in df[card_col].unique():
        mask = df[card_col] == card_id
        card_df = df.loc[mask]
        idxs = card_df.index.tolist()
        times = card_df["cre_tms"].values
        amounts = card_df["amount"].values
        n = len(times)

        cnt_1h = np.zeros(n)
        cnt_24h = np.zeros(n)
        sum_24h = np.zeros(n)

        for i in range(n):
            t = times[i]
            for j in range(i - 1, -1, -1):
                diff = t - times[j]
                if diff > twenty_four_hours:
                    break
                cnt_24h[i] += 1
                sum_24h[i] += amounts[j]
                if diff <= one_hour:
                    cnt_1h[i] += 1

        for k, idx in enumerate(idxs):
            df.at[idx, "card_txn_count_1h"] = cnt_1h[k]
            df.at[idx, "card_txn_count_24h"] = cnt_24h[k]
            df.at[idx, "card_txn_sum_24h"] = sum_24h[k]

    # Restore original order
    df = df.sort_values("_orig_pos").drop(columns=["_orig_pos"])
    df.index = orig_index

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

    # Interaction features
    df["amount_x_foreign"] = df["amount"] * df["is_foreign_currency"]
    df["amount_x_night"] = df["amount"] * df["is_night"]
    df["amount_x_ecom"] = df["amount"] * df["is_ecom"]
    df["amount_x_mcc_risk"] = df["amount"] * df["mcc_risk"]
    df["amount_x_card_risk"] = df["amount"] * df["card_risk"]

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
    iso, cases = load_data()
    iso = create_label(iso, cases)
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
