import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data():
    iso = pd.read_excel("Dataset/ISO8583_Transaction_Records.xlsx")
    cases = pd.read_excel("Dataset/Case_Creation_Details.xlsx")
    hit_rules = pd.read_excel("Dataset/Transaction_Hit_Rules.xlsx")
    return iso, cases, hit_rules


def create_label(iso):
    """Label: 1 if transaction has a valid CASE_NO (flagged for fraud), 0 otherwise."""
    case_no_int = pd.to_numeric(iso["CASE_NO"].astype(str).str.strip(), errors="coerce")
    iso = iso.copy()
    iso["fraud"] = (case_no_int > 0).astype(int)
    return iso


def add_hit_rules_features(iso, hit_rules):
    """Aggregate transaction hit rules per DE037 (reference number).

    NOTE: Transaction_Hit_Rules records which rules fired for each transaction.
    This data IS available at decision time since rules fire during authorization.
    RAD rules fire pre-authorization, CC rules fire post-authorization.
    We use only the count/type of rules triggered, not the case outcome.
    """
    hr = hit_rules.copy()
    hr["DE037"] = hr["DE037"].astype(str).str.strip()
    hr["TOT_SCR"] = pd.to_numeric(hr["TOT_SCR"].astype(str).str.strip(), errors="coerce").fillna(0)
    hr["RULE_TYP_str"] = hr["RULE_TYP"].astype(str).str.strip()

    # Aggregate per transaction reference
    hr_agg = hr.groupby("DE037").agg(
        rules_hit_count=("RULE_ID", "count"),
        rules_tot_score=("TOT_SCR", "max"),
        rules_unique_count=("RULE_ID", "nunique"),
    ).reset_index()

    # RAD vs CC rule counts
    hr_rad = hr[hr["RULE_TYP_str"] == "RAD"].groupby("DE037").size().reset_index(name="rad_rule_count")
    hr_cc = hr[hr["RULE_TYP_str"] == "CC"].groupby("DE037").size().reset_index(name="cc_rule_count")

    # Top rule indicators
    top_rules = ["N001", "TA003", "NOT1", "NOT2", "TA004", "000LJW", "C8"]
    for rule_id in top_rules:
        hr_rule = (
            hr[hr["RULE_ID"].astype(str).str.strip() == rule_id]
            .groupby("DE037").size().reset_index(name=f"rule_{rule_id}")
        )
        hr_agg = hr_agg.merge(hr_rule, on="DE037", how="left")
        hr_agg[f"rule_{rule_id}"] = hr_agg[f"rule_{rule_id}"].fillna(0)

    hr_agg = hr_agg.merge(hr_rad, on="DE037", how="left")
    hr_agg = hr_agg.merge(hr_cc, on="DE037", how="left")
    hr_agg["rad_rule_count"] = hr_agg["rad_rule_count"].fillna(0)
    hr_agg["cc_rule_count"] = hr_agg["cc_rule_count"].fillna(0)

    iso = iso.copy()
    iso["DE037_str"] = iso["DE037"].astype(str).str.strip()
    iso = iso.merge(hr_agg, left_on="DE037_str", right_on="DE037", how="left", suffixes=("", "_hr"))

    fill_cols = ["rules_hit_count", "rules_tot_score", "rules_unique_count",
                 "rad_rule_count", "cc_rule_count"]
    fill_cols += [f"rule_{r}" for r in top_rules]
    for c in fill_cols:
        iso[c] = iso[c].fillna(0)
    iso.drop(columns=["DE037_str", "DE037_hr"], errors="ignore", inplace=True)
    return iso


def engineer_features(df):
    """Engineer features from transaction-level data only.

    Avoids target leakage: no features computed using the fraud label.
    Card/merchant/MCC aggregation uses only transaction-level stats (amount, count),
    NOT fraud rates which would leak the target.
    """
    df = df.copy()

    # === Amount features ===
    df["amount"] = df["DE004"].fillna(0).astype(float)
    df["log_amount"] = np.log1p(df["amount"])
    df["amount_sq"] = df["amount"] ** 2

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
    resp_dummies = pd.get_dummies(df["resp_code"], prefix="resp")
    df = pd.concat([df, resp_dummies], axis=1)

    # === Auth status ===
    df["auth_stat"] = df["TRANS_AUTH_STAT"].astype(str).str.strip()
    auth_dummies = pd.get_dummies(df["auth_stat"], prefix="auth")
    df = pd.concat([df, auth_dummies], axis=1)

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

    # === Card-level aggregation (NO target leakage - only amount stats) ===
    card_col = "REF_CRD_NO"
    df["card_txn_count"] = df.groupby(card_col)["amount"].transform("count")
    df["card_txn_mean"] = df.groupby(card_col)["amount"].transform("mean")
    df["card_txn_std"] = df.groupby(card_col)["amount"].transform("std").fillna(0)
    df["card_txn_max"] = df.groupby(card_col)["amount"].transform("max")
    df["card_txn_min"] = df.groupby(card_col)["amount"].transform("min")
    df["amount_vs_card_mean"] = df["amount"] / (df["card_txn_mean"] + 1)
    df["amount_vs_card_max"] = df["amount"] / (df["card_txn_max"] + 1)
    df["amount_zscore"] = (df["amount"] - df["card_txn_mean"]) / (df["card_txn_std"] + 1e-8)

    # === Merchant-level aggregation (NO fraud rate - would leak target) ===
    df["merchant_id"] = df["MMER_ID"].astype(str).str.strip()
    df["merch_txn_count"] = df.groupby("merchant_id")["amount"].transform("count")
    df["merch_txn_mean"] = df.groupby("merchant_id")["amount"].transform("mean")

    # === MCC-level aggregation (NO fraud rate) ===
    df["mcc_txn_count"] = df.groupby("mcc")["amount"].transform("count")
    df["mcc_txn_mean"] = df.groupby("mcc")["amount"].transform("mean")

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

    # === TTI ===
    tti_str = df["TTI"].astype(str).str.strip()
    tti_dummies = pd.get_dummies(tti_str, prefix="tti")
    df = pd.concat([df, tti_dummies], axis=1)

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
    de003_txn_dummies = pd.get_dummies(df["de003_txn_type"], prefix="d3txn")
    df = pd.concat([df, de003_txn_dummies], axis=1)

    # === Interaction features ===
    df["amount_x_foreign"] = df["amount"] * df["is_foreign_currency"]
    df["amount_x_night"] = df["amount"] * df["is_night"]
    df["amount_x_ecom"] = df["amount"] * df["is_ecom"]
    df["rules_x_amount"] = df["rules_hit_count"] * df["log_amount"]

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
        "merchant_id", "de003_txn_type", "DE037_str",
    }
    return [c for c in df.columns if c not in exclude]


def prepare():
    iso, cases, hit_rules = load_data()
    iso = create_label(iso)
    iso = add_hit_rules_features(iso, hit_rules)
    df = engineer_features(iso)
    feature_cols = get_feature_columns(df)

    X = df[feature_cols].values.astype(np.float32)
    y = df["fraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_cols


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_cols = prepare()
    print(f"Features: {len(feature_cols)}")
    print(f"Train: {X_train.shape}, fraud rate: {y_train.mean():.4f}")
    print(f"Test:  {X_test.shape}, fraud rate: {y_test.mean():.4f}")
