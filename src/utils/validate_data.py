import great_expectations as gx
import pandas as pd
from typing import Tuple, List


def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.

    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    """
    print("🔍 Starting data validation with Great Expectations...")

    # TotalCharges arrives as a raw string column with blanks for brand-new customers
    # (tenure == 0). Coerce a copy just for the numeric checks below, without mutating
    # the caller's DataFrame (preprocess_data handles the "real" coercion later).
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("telco_pandas_datasource")
    data_asset = data_source.add_dataframe_asset(name="telco_data")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("telco_batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = context.suites.add(gx.ExpectationSuite(name="telco_suite"))

    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    print("   📋 Validating schema and required columns...")
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="customerID"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="customerID"))

    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="gender"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="Partner"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="Dependents"))

    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="PhoneService"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="InternetService"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="Contract"))

    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="tenure"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="MonthlyCharges"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="TotalCharges"))

    # === BUSINESS LOGIC VALIDATION ===
    print("   💼 Validating business logic constraints...")
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(column="gender", value_set=["Male", "Female"])
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(column="Partner", value_set=["Yes", "No"])
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(column="Dependents", value_set=["Yes", "No"])
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(column="PhoneService", value_set=["Yes", "No"])
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="Contract", value_set=["Month-to-month", "One year", "Two year"]
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="InternetService", value_set=["DSL", "Fiber optic", "No"]
        )
    )

    # === NUMERIC RANGE VALIDATION ===
    print("   📊 Validating numeric ranges and business constraints...")
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="tenure", min_value=0))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="MonthlyCharges", min_value=0))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="TotalCharges", min_value=0))

    # === STATISTICAL VALIDATION ===
    print("   📈 Validating statistical properties...")
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="tenure", min_value=0, max_value=120)
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="MonthlyCharges", min_value=0, max_value=200)
    )
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="tenure"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="MonthlyCharges"))

    # === DATA CONSISTENCY CHECKS ===
    print("   🔗 Validating data consistency...")
    suite.add_expectation(
        gx.expectations.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="TotalCharges",
            column_B="MonthlyCharges",
            or_equal=True,
            mostly=0.95,  # Allow 5% exceptions for edge cases
        )
    )

    # === RUN VALIDATION SUITE ===
    print("   ⚙️  Running complete validation suite...")
    result = batch.validate(suite).to_json_dict()

    # === PROCESS RESULTS ===
    failed_expectations = [
        r["expectation_config"]["type"] for r in result["results"] if not r["success"]
    ]

    total_checks = len(result["results"])
    failed_checks = len(failed_expectations)
    passed_checks = total_checks - failed_checks

    if result["success"]:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")

    return result["success"], failed_expectations
