import re

import pandas as pd

COURSE_NAMES = {
    "LOMA281": "LOMA 281 - Meeting Customer Needs with Insurance and Annuities - 3rd Edition",
    "LOMA291": "LOMA 291 - Improving the Bottom Line - Insurance Company Operations - 2nd Edition",
}

MODULE_NAMES = {
    ("LOMA281", "1"): "Module 1 - Risk and Insurance",
    ("LOMA281", "2"): "Module 2 - Individual Insurance Products",
    ("LOMA281", "3"): "Module 3 - Benefits, Provisions, and Ownership Rights",
    ("LOMA281", "4"): "Module 4 - Group Products",
    ("LOMA291", "1"): "Module 1 - Managing the Company to Meet Stakeholder Needs",
    ("LOMA291", "2"): "Module 2 - Serving the Customer throughout the Policy Lifecycle",
    ("LOMA291", "3"): "Module 3 - Key Support Functions for Insurer Success",
    ("LOMA291", "4"): "Module 4 - Functions and Goals of Financial Management",
}

LESSON_NAMES = {
    ("LOMA281", "1", "1"): "Lesson 1 - Risky Business",
    ("LOMA281", "1", "2"): "Lesson 2 - Organization and Regulation of Insurance Companies",
    ("LOMA281", "1", "3"): "Lesson 3 - Life Insurance Policies as Contracts",
    ("LOMA281", "1", "4"): "Lesson 4 - The Value Exchange in the Insurance Transaction",
    ("LOMA281", "2", "1"): "Lesson 1 - Term Life Insurance",
    ("LOMA281", "2", "2"): "Lesson 2 - Cash Value Life Insurance",
    ("LOMA281", "2", "3"): "Lesson 3 - Annuities",
    ("LOMA281", "2", "4"): "Lesson 4 - Health Insurance",
    ("LOMA281", "3", "1"): "Lesson 1 - Supplemental Benefits",
    ("LOMA281", "3", "2"): "Lesson 2 - Life Insurance Policy Provisions",
    ("LOMA281", "3", "3"): "Lesson 3 - Life Insurance Policy Ownership Rights",
    ("LOMA281", "4", "1"): "Lesson 1 - Group Insurance",
    ("LOMA281", "4", "2"): "Lesson 2 - Group Life Insurance",
    ("LOMA281", "4", "3"): "Lesson 3 - Group Retirement Plans",
    ("LOMA291", "1", "1"): "Lesson 1 - Many Stakeholders, Many Demands",
    ("LOMA291", "1", "2"): "Lesson 2 - The Great Organizational Pyramid",
    ("LOMA291", "1", "3"): "Lesson 3 - Risk, Return, and Risk Management",
    ("LOMA291", "2", "1"): "Lesson 1 - Distribution",
    ("LOMA291", "2", "2"): "Lesson 2 - New Business and Underwriting",
    ("LOMA291", "2", "3"): "Lesson 3 - Customer Service",
    ("LOMA291", "2", "4"): "Lesson 4 - Claims Administration",
    ("LOMA291", "3", "1"): "Lesson 1 - Marketing",
    ("LOMA291", "3", "2"): "Lesson 2 - Product Development",
    ("LOMA291", "3", "3"): "Lesson 3 - Legal and Compliance Functions",
    ("LOMA291", "4", "1"): "Lesson 1 - Financial Functions in an Insurance Company",
    ("LOMA291", "4", "2"): "Lesson 2 - Goals for Financial Management",
}

SOURCE_NAMES = {
    "LOMA281_M1L1": "LOMA281_M1L1_Knowledge File_Risk and Insurance",
    "LOMA281_M1L2": "LOMA281_M1L2_Knowledge File_Organization and Regulation of Insurance Companies",
    "LOMA281_M1L3": "LOMA281_M1L3_Knowledge File_Life Insurance Policies as Contracts",
    "LOMA281_M1L4": "LOMA281_M1L4_Knowledge File_The Value Exchange in the Insurance Transaction",
    "LOMA281_M2L1": "LOMA281_M2L1_Knowledge File_Term Life Insurance",
    "LOMA281_M2L2": "LOMA281_M2L1_Knowledge File_Term Life Insurance",
    "LOMA281_M2L3": "LOMA281_M2L3_Knowledge File_Annuities",
    "LOMA281_M2L4": "LOMA281_M2L4_Knowledge File_Health Insurance",
    "LOMA281_M3L1": "LOMA281_M3L1_Knowledge File_Supplemental Benefits",
    "LOMA281_M3L2": "LOMA281_M3L2_Knowledge File_Life Insurance Policy Provisions",
    "LOMA281_M3L3": "LOMA281_M3L3_Knowledge File_Life Insurance Policy Ownership Rights",
    "LOMA281_M4L1": "LOMA281_M4L1_Knowledge File_Group Insurance",
    "LOMA281_M4L2": "LOMA281_M4L2_Knowledge File_Group Life Insurance",
    "LOMA281_M4L3": "LOMA281_M4L3_Knowledge File_Group Retirement Plans",
    "LOMA291_M1L1": "LOMA291_M1L1_Knowledge File_Many Stakeholders, Many Demands",
    "LOMA291_M1L2": "LOMA291_M1L2_Knowledge File_Insurance Company Management",
    "LOMA291_M1L3": "LOMA291_M1L3_Knowledge File_Risk, Return, and Risk Management",
    "LOMA291_M2L1": "LOMA291_M2L1_Knowledge File_Product Distribution",
    "LOMA291_M2L2": "LOMA291_M2L2_Knowledge File_New Business and Underwriting",
    "LOMA291_M2L3": "LOMA291_M2L3_Knowledge File_Annuities",
    "LOMA291_M2L4": "LOMA291_M2L4_Knowledge File_Claim Administration",
    "LOMA291_M3L1": "LOMA291_M3L1_Knowledge File_Marketing",
    "LOMA291_M3L2": "LOMA291_M3L2_Knowledge File_Product Development",
    "LOMA291_M3L3": "LOMA291_M3L3_Knowledge File_The Legal and Compliance Function",
    "LOMA291_M4L1": "LOMA291_M4L1_Knowledge File_Financial Functions",
    "LOMA291_M4L2": "LOMA291_M4L2_Knowledge File_Goals of Financial Management",
}


def parse_source(source_str):
    """Parse source like LOMA281_M1L1 -> (course, module, lesson, key)"""
    if pd.isna(source_str) or not isinstance(source_str, str):
        return None, None, None, None

    match = re.match(r'(LOMA\d+)_M(\d+)L(\d+)', source_str.strip(), re.IGNORECASE)
    if match:
        course = match.group(1).upper()
        module = match.group(2)
        lesson = match.group(3)
        key = f"{course}_M{module}L{lesson}"
        return course, module, lesson, key

    return None, None, None, None


FILE_PATH = r"D:\Deverlopment\huudan.com\PythonProject\data\metadata_node.xlsx"

df = pd.read_excel(FILE_PATH, dtype=str)

for idx, row in df.iterrows():
    source = row.get("Source", "")
    course, module, lesson, key = parse_source(source)

    if key:
        df.at[idx, "Source"] = SOURCE_NAMES.get(key, source)

    if course:
        df.at[idx, "Course"] = COURSE_NAMES.get(course, course)

    if course and module:
        df.at[idx, "Module"] = MODULE_NAMES.get((course, module), f"Module {module}")

    if course and module and lesson:
        df.at[idx, "Lesson"] = LESSON_NAMES.get((course, module, lesson), f"Lesson {lesson}")

df.to_excel(FILE_PATH, index=False)
print(f"Done! Updated {len(df)} rows in {FILE_PATH}")
