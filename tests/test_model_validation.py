import pandas as pd
from src.predict import load_model, predict_fahrenheit

TEST_DATA_PATH = "data/test_cases.csv"


def test_model_predictions_against_expected_ranges():
    model = load_model()
    test_cases = pd.read_csv(TEST_DATA_PATH)

    results = []

    for _, row in test_cases.iterrows():
        actual = predict_fahrenheit(model, row["input_celsius"])
        expected = row["expected_fahrenheit"]
        tolerance = row["tolerance"]
        difference = abs(actual - expected)

        status = "PASS" if difference <= tolerance else "FAIL"

        results.append(
            {
                "test_type": row["test_type"],
                "input_celsius": row["input_celsius"],
                "expected_fahrenheit": expected,
                "actual_fahrenheit": round(actual, 3),
                "difference": round(difference, 3),
                "tolerance": tolerance,
                "confidence_score": row["confidence_score"],
                "failure_severity": row["failure_severity"],
                "status": status,
            }
        )

    report = pd.DataFrame(results)

    print("\n\nMODEL VALIDATION REPORT")
    print(report.to_string(index=False))

    failures = report[report["status"] == "FAIL"]

    if not failures.empty:
        print("\n\nFAILURE SUMMARY")
        print(
            failures[
                [
                    "test_type",
                    "input_celsius",
                    "expected_fahrenheit",
                    "actual_fahrenheit",
                    "difference",
                    "tolerance",
                    "confidence_score",
                    "failure_severity",
                ]
            ].to_string(index=False)
        )

    blocking_failures = failures[
        failures["failure_severity"].isin(["critical", "major"])
    ]

    assert blocking_failures.empty, (
        "\nBlocking model validation failures found:\n"
        + blocking_failures.to_string(index=False)
    )