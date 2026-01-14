"""Golden constraint comparison for evaluation results.

Compares extracted metrics against expected constraints.
"""


def compare_metrics(
    metrics: dict[str, int | float | bool],
    expected: dict[str, int | float | bool],
) -> list[str]:
    """Compare metrics against expected constraints.

    Constraint rules:
    - If key starts with 'min_': metric >= value
    - If key starts with 'max_': metric <= value
    - Otherwise: metric == value

    Args:
        metrics: Extracted metrics from run.
        expected: Expected metric constraints.

    Returns:
        List of failure messages. Empty if all constraints pass.
    """
    failures: list[str] = []

    for key, expected_value in expected.items():
        # Strip prefix to get metric name
        if key.startswith("min_"):
            metric_name = key[4:]  # Remove 'min_' prefix
            comparison = "min"
        elif key.startswith("max_"):
            metric_name = key[4:]  # Remove 'max_' prefix
            comparison = "max"
        else:
            metric_name = key
            comparison = "exact"

        # Get actual metric value
        if metric_name not in metrics:
            failures.append(f"Missing metric: {metric_name}")
            continue

        actual_value = metrics[metric_name]

        # Handle boolean comparisons
        if isinstance(expected_value, bool):
            if actual_value != expected_value:
                failures.append(
                    f"Constraint '{key}' failed: expected {expected_value}, got {actual_value}"
                )
            continue

        # Handle numeric comparisons
        if comparison == "min":
            if not isinstance(actual_value, (int, float)):
                failures.append(f"Constraint '{key}' failed: metric '{metric_name}' is not numeric")
            elif actual_value < expected_value:
                failures.append(
                    f"Constraint '{key}' failed: expected >= {expected_value}, got {actual_value}"
                )
        elif comparison == "max":
            if not isinstance(actual_value, (int, float)):
                failures.append(f"Constraint '{key}' failed: metric '{metric_name}' is not numeric")
            elif actual_value > expected_value:
                failures.append(
                    f"Constraint '{key}' failed: expected <= {expected_value}, got {actual_value}"
                )
        else:  # exact
            if actual_value != expected_value:
                failures.append(
                    f"Constraint '{key}' failed: expected {expected_value}, got {actual_value}"
                )

    return failures
