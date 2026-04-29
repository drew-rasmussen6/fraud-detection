from __future__ import annotations

import pandas as pd
import pytest

from features import build_model_frame
from analyze_fraud import score_transactions, summarize_results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def accounts():
    return pd.DataFrame([
        {"account_id": 1, "prior_chargebacks": 0},
        {"account_id": 2, "prior_chargebacks": 1},
        {"account_id": 3, "prior_chargebacks": 3},
    ])


@pytest.fixture
def transactions():
    return pd.DataFrame([
        # clean domestic, low everything
        {
            "transaction_id": 101, "account_id": 1,
            "amount_usd": 50.0, "device_risk_score": 10,
            "is_international": 0, "velocity_24h": 1, "failed_logins_24h": 0,
        },
        # high-risk: international, high device, high velocity, prior chargebacks
        {
            "transaction_id": 102, "account_id": 3,
            "amount_usd": 1500.0, "device_risk_score": 85,
            "is_international": 1, "velocity_24h": 8, "failed_logins_24h": 6,
        },
        # medium-risk
        {
            "transaction_id": 103, "account_id": 2,
            "amount_usd": 600.0, "device_risk_score": 50,
            "is_international": 0, "velocity_24h": 4, "failed_logins_24h": 1,
        },
    ])


@pytest.fixture
def chargebacks():
    return pd.DataFrame([
        {"transaction_id": 102, "loss_amount_usd": 1500.0},
    ])


# ---------------------------------------------------------------------------
# build_model_frame (features.py)
# ---------------------------------------------------------------------------

class TestBuildModelFrame:

    def test_merges_account_fields(self, transactions, accounts):
        df = build_model_frame(transactions, accounts)
        assert "prior_chargebacks" in df.columns

    def test_row_count_preserved(self, transactions, accounts):
        df = build_model_frame(transactions, accounts)
        assert len(df) == len(transactions)

    def test_is_large_amount_flag_set_for_high_amounts(self, transactions, accounts):
        df = build_model_frame(transactions, accounts)
        assert df.loc[df["transaction_id"] == 102, "is_large_amount"].values[0] == 1

    def test_is_large_amount_flag_unset_for_low_amounts(self, transactions, accounts):
        df = build_model_frame(transactions, accounts)
        assert df.loc[df["transaction_id"] == 101, "is_large_amount"].values[0] == 0

    def test_login_pressure_none_for_zero_failures(self, transactions, accounts):
        df = build_model_frame(transactions, accounts)
        assert df.loc[df["transaction_id"] == 101, "login_pressure"].values[0] == "none"

    def test_login_pressure_high_for_many_failures(self, transactions, accounts):
        df = build_model_frame(transactions, accounts)
        assert df.loc[df["transaction_id"] == 102, "login_pressure"].values[0] == "high"

    def test_login_pressure_low_for_moderate_failures(self, transactions, accounts):
        df = build_model_frame(transactions, accounts)
        assert df.loc[df["transaction_id"] == 103, "login_pressure"].values[0] == "low"

    def test_unknown_account_produces_nan_not_error(self, transactions):
        orphan = pd.DataFrame([{"account_id": 999}])
        df = build_model_frame(transactions, orphan)
        assert len(df) == len(transactions)


# ---------------------------------------------------------------------------
# score_transactions (analyze_fraud.py)
# ---------------------------------------------------------------------------

class TestScoreTransactions:

    def test_risk_score_column_present(self, transactions, accounts):
        scored = score_transactions(transactions, accounts)
        assert "risk_score" in scored.columns

    def test_risk_label_column_present(self, transactions, accounts):
        scored = score_transactions(transactions, accounts)
        assert "risk_label" in scored.columns

    def test_scores_within_bounds(self, transactions, accounts):
        scored = score_transactions(transactions, accounts)
        assert scored["risk_score"].between(0, 100).all()

    def test_high_risk_transaction_labelled_high(self, transactions, accounts):
        scored = score_transactions(transactions, accounts)
        row = scored.loc[scored["transaction_id"] == 102]
        assert row["risk_label"].values[0] == "high"

    def test_low_risk_transaction_labelled_low(self, transactions, accounts):
        scored = score_transactions(transactions, accounts)
        row = scored.loc[scored["transaction_id"] == 101]
        assert row["risk_label"].values[0] == "low"

    def test_high_risk_scores_higher_than_low_risk(self, transactions, accounts):
        scored = score_transactions(transactions, accounts)
        high_score = scored.loc[scored["transaction_id"] == 102, "risk_score"].values[0]
        low_score = scored.loc[scored["transaction_id"] == 101, "risk_score"].values[0]
        assert high_score > low_score

    def test_all_labels_are_valid(self, transactions, accounts):
        scored = score_transactions(transactions, accounts)
        assert scored["risk_label"].isin({"low", "medium", "high"}).all()

    def test_row_count_preserved(self, transactions, accounts):
        scored = score_transactions(transactions, accounts)
        assert len(scored) == len(transactions)


# ---------------------------------------------------------------------------
# summarize_results (analyze_fraud.py)
# ---------------------------------------------------------------------------

class TestSummarizeResults:

    @pytest.fixture
    def scored(self, transactions, accounts):
        return score_transactions(transactions, accounts)

    def test_output_columns_present(self, scored, chargebacks):
        summary = summarize_results(scored, chargebacks)
        for col in ("risk_label", "transactions", "total_amount_usd", "chargebacks", "chargeback_rate"):
            assert col in summary.columns, f"missing column: {col}"

    def test_transaction_counts_sum_to_total(self, scored, chargebacks):
        summary = summarize_results(scored, chargebacks)
        assert summary["transactions"].sum() == len(scored)

    def test_chargeback_rate_between_0_and_1(self, scored, chargebacks):
        summary = summarize_results(scored, chargebacks)
        assert summary["chargeback_rate"].between(0, 1).all()

    def test_known_chargeback_appears_in_summary(self, scored, chargebacks):
        summary = summarize_results(scored, chargebacks)
        assert summary["chargebacks"].sum() == len(chargebacks)

    def test_no_chargebacks_produces_zero_rate(self, scored):
        summary = summarize_results(scored, pd.DataFrame(columns=["transaction_id"]))
        assert (summary["chargeback_rate"] == 0).all()

    def test_high_risk_bucket_has_highest_chargeback_rate(self, scored, chargebacks):
        summary = summarize_results(scored, chargebacks)
        high_rate = summary.loc[summary["risk_label"] == "high", "chargeback_rate"].values[0]
        other_rates = summary.loc[summary["risk_label"] != "high", "chargeback_rate"]
        assert (high_rate >= other_rates).all()
