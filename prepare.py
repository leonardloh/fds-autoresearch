import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data():
    iso = pd.read_excel("Dataset/ISO8583_Transaction_Records.xlsx")
    cases = pd.read_excel("Dataset/Case_Creation_Details.xlsx")
    return iso, cases


def create_label(iso, cases):
    """Label: fraud=1 if transaction has a valid CASE_NO (flagged for investigation).

    While not all flagged transactions are confirmed fraud, the case creation
    process reflects genuine suspicion from the rule-based system. With only
    33 confirmed fraud cases (70 txns), using confirmed-only labels would be
    too few for meaningful modeling. Using all cased transactions (4415 txns)
    as positive class is a pragmatic choice.

    The model's goal: learn transaction characteristics that predict flagging,
    WITHOUT using the hit-rules features that directly cause the flagging.
    """
    iso = iso.copy()
    case_no_int = pd.to_numeric(
        iso["CASE_NO"].astype(str).str.strip(), errors="coerce"
    ).fillna(0).astype(int)
    iso["fraud"] = (case_no_int > 0).astype(int)
    return iso


def engineer_features(df):
    """Engineer features from transaction data only.

    No hit-rules features (would be leakage with confirmed-fraud label).
    Aggregation features computed here on full data, but will be
    recomputed properly in prepare() after split.
    """
    df = df.copy()

    # === Amount features ===
    df["amount"] = df["DE004"].fillna(0).astype(float)
    df["log_amount"] = np.log1p(df["amount"])

    # === Transaction type ===
    df["trx_typ"] = df["TRX_TYP"].astype(str).str.strip()
    trx_dummies = pd.get_dummies(df["trx_typ"], prefix="trx")
    df = pd.concat([df, trx_dummies], axis=1)

    # === POS entry mode ===
    df["pos_entry"] = df["DE022"].astype(str).str.strip()
    pos_dummies = pd.get_dummies(df["pos_entry"], prefix="pos")
    df = pd.concat([df, pos_dummies], axis=1)

    # === MCC ===
    df["mcc"] = df["DE018"].fillna(0).astype(int)

    # === Response code ===
    df["resp_code"] = df["DE039"].astype(str).str.strip()
    df["is_approved"] = (df["resp_code"] == "00").astype(int)
    df["is_declined_fraud"] = (df["resp_code"] == "59").astype(int)  # suspected fraud

    # === Auth status ===
    df["auth_stat"] = df["TRANS_AUTH_STAT"].astype(str).str.strip()
    df["is_auth_approved"] = (df["auth_stat"] == "").astype(int)
    df["is_auth_declined"] = (df["auth_stat"] == "DD").astype(int)

    # === Card-present flags ===
    df["is_emv"] = (df["EMV_STAT"].astype(str).str.strip() == "Y").astype(int)
    df["is_contactless"] = (df["EMV_STAT"].astype(str).str.strip() == "C").astype(int)
    df["is_chip"] = (df["CHIP_STAT"].astype(str).str.strip() == "Y").astype(int)
    df["is_fallback"] = (df["FALLBCK_FLG"].astype(str).str.strip() == "Y").astype(int)
    df["is_ecom"] = (df["ECOM_3D_FLG"].astype(str).str.strip() == "Y").astype(int)

    # === CVC2 ===
    df["cvc2_match"] = (df["CVC2_FLG"].astype(str).str.strip() == "M").astype(int)
    df["cvc2_nomatch"] = (df["CVC2_FLG"].astype(str).str.strip() == "N").astype(int)

    # === Currency ===
    df["currency"] = df["DE049"].astype(str).str.strip()
    df["is_foreign_currency"] = (df["currency"] != "458").astype(int)

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

    # === DE003 subfields ===
    de003_str = df["DE003"].astype(str).str.zfill(6)
    df["de003_txn_type"] = de003_str.str[:2]
    de003_dummies = pd.get_dummies(df["de003_txn_type"], prefix="d3txn")
    df = pd.concat([df, de003_dummies], axis=1)

    # === Timestamp for temporal split ===
    df["cre_tms"] = pd.to_numeric(df["CRE_TMS"], errors="coerce").fillna(0)

    # === Settlement / billing amount features ===
    df["settle_amount"] = df["DE005"].fillna(0).astype(float)
    df["billing_amount"] = df["DE006"].fillna(0).astype(float)
    df["amount_billing_diff"] = df["amount"] - df["billing_amount"]
    df["amount_billing_ratio"] = df["amount"] / (df["billing_amount"] + 1)

    return df


def compute_aggregation_features(df, train_df):
    """Compute card/merchant/MCC aggregation using ONLY training data stats.

    This avoids train-test leakage in aggregation features.
    """
    df = df.copy()
    card_col = "REF_CRD_NO"

    # Card-level stats from training data only
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

    # Merchant-level stats from training data only
    merchant_col = "MMER_ID"
    m_str_col = "_merch_str"
    df[m_str_col] = df[merchant_col].astype(str).str.strip()
    train_m = train_df.copy()
    train_m[m_str_col] = train_m[merchant_col].astype(str).str.strip()
    merch_stats = train_m.groupby(m_str_col)["amount"].agg(
        merch_txn_count="count", merch_txn_mean="mean",
    ).fillna(0)
    df = df.merge(merch_stats, left_on=m_str_col, right_index=True, how="left")
    for c in ["merch_txn_count", "merch_txn_mean"]:
        df[c] = df[c].fillna(0)
    df.drop(columns=[m_str_col], inplace=True)

    # MCC-level stats from training data only
    mcc_stats = train_df.groupby("mcc")["amount"].agg(
        mcc_txn_count="count", mcc_txn_mean="mean",
    ).fillna(0)
    df = df.merge(mcc_stats, left_on="mcc", right_index=True, how="left")
    for c in ["mcc_txn_count", "mcc_txn_mean"]:
        df[c] = df[c].fillna(0)

    # === Target-encoded risk (from training data only, with smoothing) ===
    global_mean = train_df["fraud"].mean()
    smoothing = 10  # regularization parameter

    # MCC risk
    mcc_fraud = train_df.groupby("mcc")["fraud"].agg(["mean", "count"])
    mcc_fraud["mcc_risk"] = (
        (mcc_fraud["count"] * mcc_fraud["mean"] + smoothing * global_mean)
        / (mcc_fraud["count"] + smoothing)
    )
    df = df.merge(mcc_fraud[["mcc_risk"]], left_on="mcc", right_index=True, how="left")
    df["mcc_risk"] = df["mcc_risk"].fillna(global_mean)

    # Merchant risk
    train_m2 = train_df.copy()
    train_m2["_m2"] = train_m2["MMER_ID"].astype(str).str.strip()
    df["_m2"] = df["MMER_ID"].astype(str).str.strip()
    merch_fraud = train_m2.groupby("_m2")["fraud"].agg(["mean", "count"])
    merch_fraud["merch_risk"] = (
        (merch_fraud["count"] * merch_fraud["mean"] + smoothing * global_mean)
        / (merch_fraud["count"] + smoothing)
    )
    df = df.merge(merch_fraud[["merch_risk"]], left_on="_m2", right_index=True, how="left")
    df["merch_risk"] = df["merch_risk"].fillna(global_mean)
    df.drop(columns=["_m2"], inplace=True)

    # Card risk
    card_fraud = train_df.groupby(card_col)["fraud"].agg(["mean", "count"])
    card_fraud["card_risk"] = (
        (card_fraud["count"] * card_fraud["mean"] + smoothing * global_mean)
        / (card_fraud["count"] + smoothing)
    )
    df = df.merge(card_fraud[["card_risk"]], left_on=card_col, right_index=True, how="left")
    df["card_risk"] = df["card_risk"].fillna(global_mean)

    # === Velocity features (per-card transaction counts in time windows) ===
    # Sort by card and time, compute rolling counts
    df = df.sort_values(["REF_CRD_NO", "cre_tms"]).reset_index(drop=True)
    for card_id in df["REF_CRD_NO"].unique():
        mask = df["REF_CRD_NO"] == card_id
        card_times = df.loc[mask, "cre_tms"].values
        card_amounts = df.loc[mask, "amount"].values
        n = len(card_times)

        txn_count_24h = np.zeros(n)
        txn_sum_24h = np.zeros(n)
        txn_count_1h = np.zeros(n)

        # Time windows: CRE_TMS format is YYYYMMDDHHMMSSmmm
        # 1 hour = 10000000 (HHMM * 10^7), 24 hours = 240000000
        one_hour = 10000000
        twenty_four_hours = 240000000

        for i in range(n):
            t = card_times[i]
            for j in range(i - 1, -1, -1):
                diff = t - card_times[j]
                if diff > twenty_four_hours:
                    break
                txn_count_24h[i] += 1
                txn_sum_24h[i] += card_amounts[j]
                if diff <= one_hour:
                    txn_count_1h[i] += 1

        df.loc[mask, "card_txn_count_1h"] = txn_count_1h
        df.loc[mask, "card_txn_count_24h"] = txn_count_24h
        df.loc[mask, "card_txn_sum_24h"] = txn_sum_24h

    for c in ["card_txn_count_1h", "card_txn_count_24h", "card_txn_sum_24h"]:
        df[c] = df[c].fillna(0)

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
        "trx_typ", "pos_entry", "resp_code", "auth_stat", "currency",
        "de003_txn_type", "cre_tms",
    }
    return [c for c in df.columns if c not in exclude]


def prepare():
    iso, cases = load_data()
    iso = create_label(iso, cases)
    df = engineer_features(iso)

    # Time-based split: sort by creation timestamp, 80% train / 20% test
    df = df.sort_values("cre_tms").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Compute aggregation features using ONLY training data
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
