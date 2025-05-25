import os
import pandas as pd
import re
from tqdm import tqdm
import numpy as np
from datetime import timedelta

def extract_demographics(cohort, hosp_path):
    """Extract demographic features including age and gender."""
    print("Extracting demographic features...")

    # Load patients table
    patients = pd.read_csv(
        os.path.join(hosp_path, "_patients.csv"),
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
    )

    # Load admissions for timestamp data
    admissions = pd.read_csv(
        os.path.join(hosp_path, "_admissions.csv"),
        usecols=["subject_id", "hadm_id", "admittime"],
    )
    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    admissions["admittime_year"] = admissions["admittime"].dt.year

    # Convert gender to numeric (1 for M, 0 for F)
    patients["gender_numeric"] = patients["gender"].map({"M": 1, "F": 0})

    # Merge patients with cohort
    features = cohort.merge(patients, on="subject_id", how="left")

    # Merge with admissions to get admission time
    features = features.merge(
        admissions[["subject_id", "hadm_id", "admittime", "admittime_year"]],
        on=["subject_id", "hadm_id"],
        how="left",
    )

    # Calculate age at admission time
    features["age"] = features["anchor_age"] + (
        features["admittime_year"] - features["anchor_year"]
    )

    # Cap ages above 89 at 90 (for de-identification)
    features.loc[features["age"] > 89, "age"] = 90

    # Keep track of feature extraction progress
    print(
        f"Extracted demographic features for {features['subject_id'].nunique()} unique subjects"
    )

    return features


def extract_prior_diagnoses(features, hosp_path):
    """Extract information about prior diagnoses with ICD chapter categorization."""
    # Load admissions and diagnoses tables
    admissions = pd.read_csv(
        os.path.join(hosp_path, "_admissions.csv"),
        usecols=["subject_id", "hadm_id", "admittime"],
    )
    admissions["admittime"] = pd.to_datetime(admissions["admittime"])

    diagnoses = pd.read_csv(
        os.path.join(hosp_path, "_diagnoses_icd.csv"),
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
    )

    # Create lookup dictionary for ICU admission times
    icu_admit_dict = features.set_index("subject_id")["intime"].to_dict()

    # ICD code categorization functions
    def map_icd9_to_chapter(icd_code_str):
        """Map ICD-9 code to clinical chapter"""
        if pd.isna(icd_code_str):
            return "unknown"
        icd_code_str = str(icd_code_str).upper()

        # Handle V and E codes
        if any(prefix in icd_code_str for prefix in ["V", "E"]):
            return "other_icd9"

        # Process numeric codes
        try:
            numeric_part = int(re.match(r"([0-9]+)", icd_code_str).group(1))
            if 1 <= numeric_part <= 139:
                return "infectious_parasitic"
            if 140 <= numeric_part <= 239:
                return "neoplasms"
            if 240 <= numeric_part <= 279:
                return "endocrine_metabolic"
            if 280 <= numeric_part <= 289:
                return "blood_disorders"
            if 290 <= numeric_part <= 319:
                return "mental_disorders"
            if 320 <= numeric_part <= 389:
                return "nervous_sensory"
            if 390 <= numeric_part <= 459:
                return "circulatory"
            if 460 <= numeric_part <= 519:
                return "respiratory"
            if 520 <= numeric_part <= 579:
                return "digestive"
            if 580 <= numeric_part <= 629:
                return "genitourinary"
            if 630 <= numeric_part <= 679:
                return "pregnancy_childbirth"
            if 680 <= numeric_part <= 709:
                return "skin_subcutaneous"
            if 710 <= numeric_part <= 739:
                return "musculoskeletal_connective"
            if 740 <= numeric_part <= 759:
                return "congenital_anomalies"
            if 760 <= numeric_part <= 779:
                return "perinatal_conditions"
            if 780 <= numeric_part <= 799:
                return "symptoms_signs_illdefined"
            if 800 <= numeric_part <= 999:
                return "injury_poisoning"
            return "other_icd9"
        except:
            return "other_icd9"

    def map_icd10_to_chapter(icd_code_str):
        """Map ICD-10 code to clinical chapter"""
        if pd.isna(icd_code_str):
            return "unknown"
        icd_code_str = str(icd_code_str).upper()

        if "A" <= icd_code_str[0] <= "B":
            return "infectious_parasitic"
        if icd_code_str[0] == "C" or (
            icd_code_str[0] == "D" and "0" <= icd_code_str[1] <= "4"
        ):
            return "neoplasms"
        if icd_code_str[0] == "D" and "5" <= icd_code_str[1] <= "9":
            return "blood_disorders"
        if icd_code_str[0] == "E":
            return "endocrine_metabolic"
        if icd_code_str[0] == "F":
            return "mental_disorders"
        if icd_code_str[0] == "G":
            return "nervous_sensory"
        if icd_code_str[0] == "H" and "0" <= icd_code_str[1] <= "5":
            return "nervous_sensory"  # eye
        if icd_code_str[0] == "H" and "6" <= icd_code_str[1] <= "9":
            return "nervous_sensory"  # ear
        if icd_code_str[0] == "I":
            return "circulatory"
        if icd_code_str[0] == "J":
            return "respiratory"
        if icd_code_str[0] == "K":
            return "digestive"
        if icd_code_str[0] == "L":
            return "skin_subcutaneous"
        if icd_code_str[0] == "M":
            return "musculoskeletal_connective"
        if icd_code_str[0] == "N":
            return "genitourinary"
        if icd_code_str[0] == "O":
            return "pregnancy_childbirth"
        if icd_code_str[0] == "P":
            return "perinatal_conditions"
        if icd_code_str[0] == "Q":
            return "congenital_anomalies"
        if icd_code_str[0] == "R":
            return "symptoms_signs_illdefined"
        if "S" <= icd_code_str[0] <= "T":
            return "injury_poisoning"
        if "V" <= icd_code_str[0] <= "Y":
            return "external_causes"
        if icd_code_str[0] == "Z":
            return "factors_health_status"
        return "other_icd10"

    # Create a dataframe with unique subject IDs
    subject_ids = features["subject_id"].unique()
    prior_dx_df = pd.DataFrame({"subject_id": subject_ids})
    prior_dx_df["has_prior_diagnoses"] = 0  # Default value

    # Initialize columns for ICD chapters
    icd_chapters = [
        "infectious_parasitic",
        "neoplasms",
        "endocrine_metabolic",
        "blood_disorders",
        "mental_disorders",
        "nervous_sensory",
        "circulatory",
        "respiratory",
        "digestive",
        "genitourinary",
        "pregnancy_childbirth",
        "skin_subcutaneous",
        "musculoskeletal_connective",
        "congenital_anomalies",
        "perinatal_conditions",
        "symptoms_signs_illdefined",
        "injury_poisoning",
        "external_causes",
        "factors_health_status",
        "other_icd9",
        "other_icd10",
        "unknown",
    ]

    for chapter in icd_chapters:
        prior_dx_df[f"prev_dx_{chapter}_count"] = 0

    # Store all prior diagnoses
    all_prior_dx = []

    # Process each patient
    for subject_id in tqdm(subject_ids, desc="Processing prior diagnoses"):
        # Get all admissions for this patient
        pt_admissions = admissions[admissions["subject_id"] == subject_id]

        if not pt_admissions.empty and subject_id in icu_admit_dict:
            # Get the ICU admission time for this patient
            icu_admit = icu_admit_dict[subject_id]

            # Check for prior admissions
            prior_admits = pt_admissions[pt_admissions["admittime"] < icu_admit]

            if len(prior_admits) > 0:
                # Check if prior admissions have diagnoses
                prior_hadm_ids = prior_admits["hadm_id"].tolist()
                prior_diagnoses = diagnoses[
                    diagnoses["hadm_id"].isin(prior_hadm_ids)
                ].copy()

                if not prior_diagnoses.empty:
                    # Update has_prior_diagnoses value for this subject
                    prior_dx_df.loc[
                        prior_dx_df["subject_id"] == subject_id, "has_prior_diagnoses"
                    ] = 1

                    # Categorize diagnoses by ICD chapters
                    prior_diagnoses["icd_chapter"] = prior_diagnoses.apply(
                        lambda r: (
                            map_icd9_to_chapter(r["icd_code"])
                            if r["icd_version"] == 9
                            else (
                                map_icd10_to_chapter(r["icd_code"])
                                if r["icd_version"] == 10
                                else "unknown"
                            )
                        ),
                        axis=1,
                    )

                    # Count diagnoses by chapter
                    chapter_counts = (
                        prior_diagnoses["icd_chapter"].value_counts().to_dict()
                    )

                    # Update counts in the dataframe
                    for chapter, count in chapter_counts.items():
                        col_name = f"prev_dx_{chapter}_count"
                        if col_name in prior_dx_df.columns:
                            prior_dx_df.loc[
                                prior_dx_df["subject_id"] == subject_id, col_name
                            ] = count

                    # Add total count
                    prior_dx_df.loc[
                        prior_dx_df["subject_id"] == subject_id, "prev_dx_count_total"
                    ] = len(prior_diagnoses)
                else:
                    prior_dx_df.loc[
                        prior_dx_df["subject_id"] == subject_id, "prev_dx_count_total"
                    ] = 0
            else:
                prior_dx_df.loc[
                    prior_dx_df["subject_id"] == subject_id, "prev_dx_count_total"
                ] = 0
        else:
            prior_dx_df.loc[
                prior_dx_df["subject_id"] == subject_id, "prev_dx_count_total"
            ] = 0

    print(f"Extracted prior diagnosis information for {len(prior_dx_df)} patients")
    return prior_dx_df


def extract_metastatic_cancer_flag(cohort, hosp_path):
    """Extract metastatic cancer flag based on ICD diagnosis codes."""
    print("Extracting metastatic cancer flag...")

    # Load diagnoses table
    diagnoses = pd.read_csv(
        os.path.join(hosp_path, "_diagnoses_icd.csv"),
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
    )

    # Get admissions to match hadm_id with cohort
    admissions = pd.read_csv(
        os.path.join(hosp_path, "_admissions.csv"), usecols=["subject_id", "hadm_id"]
    )

    # Define ICD codes for metastatic cancer
    # ICD-9 codes for metastatic cancer
    icd9_metastatic = [
        "196",
        "197",
        "198",
        "199.0",
        "199.1",  # Main metastatic codes
    ]
    # Include codes that start with these patterns
    icd9_patterns = ["196.", "197.", "198."]

    # ICD-10 codes for metastatic cancer
    icd10_metastatic = ["C77", "C78", "C79", "C7B", "C80"]

    # Merge cohort with admissions to get hadm_id
    cohort_adm = cohort.merge(admissions, on="subject_id", how="left")

    # Get all diagnoses for cohort patients
    cohort_dx = diagnoses[diagnoses["subject_id"].isin(cohort["subject_id"])]

    # Flag metastatic cancer diagnoses
    def is_metastatic(row):
        code = str(row["icd_code"]).strip().upper()
        version = row["icd_version"]

        if version == 9:
            # Check exact codes
            if code in icd9_metastatic:
                return True
            # Check pattern starts
            for pattern in icd9_patterns:
                if code.startswith(pattern):
                    return True
        elif version == 10:
            # Check ICD-10 codes (using startswith for category)
            for metastatic_code in icd10_metastatic:
                if code.startswith(metastatic_code):
                    return True
        return False

    # Apply the metastatic function
    cohort_dx["is_metastatic"] = cohort_dx.apply(is_metastatic, axis=1)

    # Group by subject_id and get if any diagnosis is metastatic
    metastatic_by_subject = (
        cohort_dx.groupby("subject_id")["is_metastatic"].any().reset_index()
    )
    metastatic_by_subject.rename(
        columns={"is_metastatic": "has_metastatic_cancer"}, inplace=True
    )

    # Convert boolean to int for consistency
    metastatic_by_subject["has_metastatic_cancer"] = metastatic_by_subject[
        "has_metastatic_cancer"
    ].astype(int)

    # Ensure all cohort subjects are included, even those without metastatic cancer
    result = pd.DataFrame({"subject_id": cohort["subject_id"].unique()})
    result = result.merge(metastatic_by_subject, on="subject_id", how="left")
    result["has_metastatic_cancer"] = (
        result["has_metastatic_cancer"].fillna(0).astype(int)
    )

    metastatic_count = result["has_metastatic_cancer"].sum()
    print(
        f"Identified {metastatic_count} patients with metastatic cancer ({metastatic_count/len(result):.1%} of cohort)"
    )

    return result


