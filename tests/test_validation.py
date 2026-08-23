from loan_status_prediction.validation import summarize_cv_scores


def test_summarize_cv_scores_ignores_non_test_scores():
    summary = summarize_cv_scores(
        {
            "fit_time": [1.0, 2.0],
            "test_roc_auc": [0.8, 0.9],
            "test_f1": [0.5, 0.7],
        },
        "demo_model",
    )

    assert summary["model"] == "demo_model"
    assert summary["roc_auc_mean"] == 0.85
    assert summary["f1_mean"] == 0.6
    assert "fit_time_mean" not in summary
