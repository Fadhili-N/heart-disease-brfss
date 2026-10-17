"""Feature selection and cleaning for the BRFSS heart-disease task.

Every survey answer is stored as a number, but 7 / 9 / 77 / 99 mean "don't know"
or "refused", not a real value. This module turns those codes into proper NaN so
that imputation happens later, inside the model pipeline, and only on training data.
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# What is deliberately left out, and why. The notebook prints this table.
# ---------------------------------------------------------------------------
EXCLUDED = {
    "_MICHD": "Derived by the CDC from the same two questions as the target; it is the answer.",
    "CVDINFR4": "One of the two target questions.",
    "CVDCRHD4": "One of the two target questions.",
    "CVDSTRK3": "Stroke is a downstream outcome of the same vascular disease, not a risk factor "
                "a screener would know in advance. Tested later as an ablation.",
    "BPMEDS1": "Blood-pressure medication: a treatment that often follows a heart event, and "
               "asked only of people already told they have high BP (59% blank).",
    "CHOLMED3": "Cholesterol medication (statins): commonly started after a heart event.",
    "ASPIRIN, HASYMP*, STRSYMP*, FIRSTAID": "Heart-attack and stroke knowledge/prevention module; "
                                            "closely tied to the outcome and over 90% blank.",
    "_STATE, FMONTH, IDATE, SEQNO, _PSU, DISPCODE": "Interview administration and identifiers.",
    "_LLCPWT, _LLCPWT2, _STSTR, _STRWT, _RAWRAKE, _WT2RAKE": "Survey sampling weights and strata, "
                                                             "not properties of the person.",
    "Optional modules (cancer screening, caregiving, ACEs, COVID, ...)": "Asked of a small subset "
                                                                          "of states or respondents; 50-99% blank by design.",
    "WEIGHT2, HEIGHT3, WTKG3, HTIN4, HTM4": "Replaced by the single derived BMI value.",
    "EXERANY2, _PHYS14D, _MENT14D, _RFHLTH, _RFBMI5, _AGE80 ...": "Re-codings of variables that are already included.",
}

# ---------------------------------------------------------------------------
# Included variables, grouped by how they are cleaned. raw name -> readable name
# ---------------------------------------------------------------------------
# 1 = yes, 2 = no, 7 = don't know, 9 = refused
YES_NO = {
    "VETERAN3": "veteran", "ASTHMA3": "asthma", "CHCCOPD3": "copd", "CHCKDNY2": "kidney_disease",
    "HAVARTH4": "arthritis", "ADDEPEV3": "depression", "CHCSCNC1": "skin_cancer",
    "CHCOCNC1": "other_cancer", "DIFFWALK": "difficulty_walking", "DIFFDRES": "difficulty_dressing",
    "DIFFALON": "difficulty_errands", "DEAF": "deaf", "BLIND": "blind",
    "DECIDE": "difficulty_concentrating", "MEDCOST1": "skipped_care_for_cost", "TOLDHI3": "high_cholesterol",
}
# CDC-derived 1/2 flags with 9 = unknown. value: (readable name, which code means "1")
DERIVED_FLAGS = {
    "_SEX": ("is_female", 2),
    "_URBSTAT": ("rural", 2),
    "_TOTINDA": ("physically_active", 1),
    "_RFDRHV8": ("heavy_drinker", 2),
    "_HLTHPL1": ("has_insurance", 1),
}
# ordered scales; the listed codes are "unknown"
ORDINAL = {
    "_AGEG5YR": ("age_group", [14]),
    "GENHLTH": ("general_health", [7, 9]),
    "_EDUCAG": ("education", [9]),
    "_INCOMG1": ("income", [9]),
}
# unordered categories; the listed codes are "unknown"
NOMINAL = {
    "_RACEGR3": ("race_group", [9]),
    "MARITAL": ("marital_status", [9]),
    "EMPLOY1": ("employment", [9]),
    "RENTHOM1": ("housing", [7, 9]),
    "BPHIGH6": ("blood_pressure_status", [7, 9]),
    "DIABETE4": ("diabetes_status", [7, 9]),
    "_SMOKER3": ("smoking_status", [9]),
    "PERSDOC3": ("personal_doctor", [7, 9]),
}

NOMINAL_FEATURES = [name for name, _ in NOMINAL.values()]
BINARY_FEATURES = list(YES_NO.values()) + [name for name, _ in DERIVED_FLAGS.values()]
NUMERIC_FEATURES = [name for name, _ in ORDINAL.values()] + [
    "checkup_recency", "physical_unwell_days", "mental_unwell_days", "bmi",
]
ENGINEERED = ["n_conditions", "n_limitations", "unwell_days_total"]

CONDITION_COLS = ["asthma", "copd", "kidney_disease", "arthritis", "depression",
                  "skin_cancer", "other_cancer", "high_cholesterol"]
LIMITATION_COLS = ["difficulty_walking", "difficulty_dressing", "difficulty_errands",
                   "deaf", "blind", "difficulty_concentrating"]


def clean_features(df: pd.DataFrame, engineer: bool = True) -> pd.DataFrame:
    """Return a numeric feature table with unknown answers as NaN.

    No imputation, scaling or encoding happens here: those are fitted on the
    training split only, inside the model pipeline.
    """
    out = pd.DataFrame(index=df.index)

    for raw, name in YES_NO.items():
        out[name] = df[raw].map({1: 1.0, 2: 0.0})

    for raw, (name, one_code) in DERIVED_FLAGS.items():
        out[name] = (df[raw] == one_code).astype(float).where(df[raw].isin([1, 2]))

    for raw, (name, unknown) in ORDINAL.items():
        out[name] = df[raw].where(~df[raw].isin(unknown))

    for raw, (name, unknown) in NOMINAL.items():
        out[name] = df[raw].where(~df[raw].isin(unknown))

    # time since last check-up: 1 = within a year ... 4 = 5+ years, 8 = never
    out["checkup_recency"] = df["CHECKUP1"].replace({8: 5}).where(df["CHECKUP1"].isin([1, 2, 3, 4, 8]))
    # days unwell in the last 30: 88 means none, 77 / 99 mean unknown
    for raw, name in [("PHYSHLTH", "physical_unwell_days"), ("MENTHLTH", "mental_unwell_days")]:
        out[name] = df[raw].replace({88: 0}).where(df[raw].between(0, 30) | (df[raw] == 88))
    # BMI is stored as an integer with two implied decimals
    out["bmi"] = df["_BMI5"] / 100.0

    if engineer:
        out["n_conditions"] = (
            (out[CONDITION_COLS] == 1).sum(axis=1)
            + (df["DIABETE4"] == 1).astype(int)
            + (df["BPHIGH6"] == 1).astype(int)
        )
        out["n_limitations"] = (out[LIMITATION_COLS] == 1).sum(axis=1)
        out["unwell_days_total"] = out["physical_unwell_days"] + out["mental_unwell_days"]

    return out
