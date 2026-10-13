"""复现冻结前的 15 项纯合成分析器检查，不读取实测、不运行求解。"""

import hashlib
import json
from pathlib import Path
import runpy

ANALYZER = Path(__file__).with_name("v42-newton-analyze.py")
EXPECTED_SHA256 = "ed490c72c9c05c6ece262b2b35942783cb09ec66e97901bf0356a0788d3a2a4a"


def main():
    assert hashlib.sha256(ANALYZER.read_bytes()).hexdigest() == EXPECTED_SHA256
    api = runpy.run_path(str(ANALYZER))
    checks = 0

    def event(kind, direction, event_id, **fields):
        value = dict(schema=1, event=kind, direction_id=direction, event_id=event_id,
                     outer_iteration=direction, phase="predictor", stage="initial",
                     finite_fields=True, candidate_pending=False)
        for key in api["DIRECTION_COUNTERS"] + api["DIAGNOSTIC_COUNTERS"]:
            value[key] = 0
        for key in api["PRODUCTION_COUNTERS"]:
            value[key] = 0
            value[key + "_valid"] = True
        value.update(diagnostic_cache_peak_elements=100,
                     diagnostic_seconds_completed_before_event=0.1 * event_id,
                     diagnostic_seconds_completed_before_event_valid=True)
        value.update(fields)
        return value

    def checked(kind, direction, event_id):
        value = event(kind, direction, event_id, accepted=False)
        prefixes = api["PREFIXES"] + api["DIFFERENCE_PREFIXES"] if kind == "candidate" else ("true",)
        for prefix in prefixes:
            for suffix in api["NORM_SUFFIXES"]:
                key = prefix + "_" + suffix
                value[key], value[key + "_valid"] = 2.0, True
        keys = ["production_measure"] + (["givens_weighted_l2", "remaining"] if kind == "candidate" else [])
        for key in keys:
            value[key], value[key + "_valid"] = 2.0, True
        if kind == "candidate":
            for prefix in api["DIRECTION_DIFFERENCE_PREFIXES"]:
                for suffix in ("l2", "inf"):
                    key = prefix + "_" + suffix
                    value[key], value[key + "_valid"] = 0.0, True
            value.update(ld_sum_finite=True, once_cast_finite=True, once_cast_underflows=0)
        return value

    events = [event("direction_begin", 1, 1), checked("initial_check", 1, 2),
              event("direction_return", 1, 3, failure_kind="none"), event("direction_begin", 2, 4),
              event("arnoldi_start", 2, 5, direction_krylov_steps=1),
              event("direction_failure", 2, 6, direction_krylov_steps=1, failure_kind="time_limit",
                    candidate_pending=True, candidates_started=1)]
    result = api["analyze_record"](dict(events=events, report={"status": "time_limit"},
                                       diagnostic_parse_errors=[], variant="diagnostic_on"))
    assert result["analysis_complete"]
    checks += 1
    assert [result["selections"][key]["direction_id"] for key in
            ("last_failed_or_interrupted", "preceding_returned", "first_krylov")] == [2, 1, 2]
    checks += 1
    failed = result["directions"][1]
    assert failed["termination"] == "time_limit" and failed["partial_candidate_state"]["started_minus_completed"] == 1
    checks += 1
    assert failed["costs"]["global_cache_peak_endpoints"] == {"begin": 100, "terminal": 100}
    checks += 1
    assert "diagnostic_cache_peak_elements" not in failed["costs"]["diagnostic_counter_delta"]
    checks += 1
    fallback = api["analyze_direction"](1, list(enumerate([
        event("direction_begin", 1, 1), event("direction_failure", 1, 2,
        failure_kind="time_limit", direction_krylov_steps=1)])))
    assert fallback["triggered_krylov"]
    checks += 1
    partial = api["analyze_direction"](1, list(enumerate([event("direction_begin", 1, 1),
        {"event": "diagnostic_error", "event_id": 2, "direction_id": 1, "finite_fields": False}])))
    assert partial["termination"] == "diagnostic_disabled" and partial["outcome"] == "missing_terminal" and partial["issues"]
    checks += 1
    assert api["selections"]([partial], True)["last_failed_or_interrupted"]["direction_id"] is None
    checks += 1
    missing = api["analyze_direction"](1, [(0, event("direction_begin", 1, 1))])
    assert api["selections"]([missing], True)["last_failed_or_interrupted"]["direction_id"] == 1
    checks += 1
    assert api["selections"]([missing], False)["last_failed_or_interrupted"]["direction_id"] is None
    checks += 1
    candidate = api["inspect_check"](checked("candidate", 1, 1), 0)
    assert not candidate["missing_metrics"] and not candidate["invalid_metrics"] and len(candidate["metrics"]) == 81
    checks += 1
    assert candidate["ratios"]["true_over_image_weighted_l2"] == 1 and candidate["accepted_by_production_measure"] is False
    checks += 1
    assert api["ratio"](1, 0) is None and api["ratio"](True, 1) is None
    checks += 1
    invalid = checked("candidate", 1, 1)
    invalid["true_weighted_l2_valid"] = False
    assert "true_weighted_l2" in api["inspect_check"](invalid, 0)["invalid_metrics"]
    checks += 1
    json.dumps(result, allow_nan=False)
    json.dumps(partial, allow_nan=False)
    checks += 1
    print(json.dumps({"passed": True, "checks": checks, "analyzer_sha256": EXPECTED_SHA256}))


if __name__ == "__main__":
    main()
