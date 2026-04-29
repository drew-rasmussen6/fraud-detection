from risk_rules import label_risk, score_transaction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def base_tx(**overrides):
    """Low-risk baseline transaction; override individual fields per test."""
    tx = {
        "device_risk_score": 10,
        "is_international": 0,
        "amount_usd": 100,
        "velocity_24h": 1,
        "failed_logins_24h": 0,
        "prior_chargebacks": 0,
    }
    tx.update(overrides)
    return tx


# ---------------------------------------------------------------------------
# label_risk thresholds
# ---------------------------------------------------------------------------

def test_label_risk_low():
    assert label_risk(0) == "low"
    assert label_risk(29) == "low"

def test_label_risk_medium():
    assert label_risk(30) == "medium"
    assert label_risk(59) == "medium"

def test_label_risk_high():
    assert label_risk(60) == "high"
    assert label_risk(100) == "high"


# ---------------------------------------------------------------------------
# device_risk_score
# ---------------------------------------------------------------------------

def test_high_device_risk_increases_score():
    low = score_transaction(base_tx(device_risk_score=10))
    high = score_transaction(base_tx(device_risk_score=75))
    assert high > low

def test_very_high_device_risk_adds_more_than_medium():
    mid = score_transaction(base_tx(device_risk_score=50))
    high = score_transaction(base_tx(device_risk_score=75))
    assert high > mid

def test_low_device_risk_adds_nothing():
    score = score_transaction(base_tx(device_risk_score=10))
    assert score == 0


# ---------------------------------------------------------------------------
# is_international
# ---------------------------------------------------------------------------

def test_international_increases_score():
    domestic = score_transaction(base_tx(is_international=0))
    international = score_transaction(base_tx(is_international=1))
    assert international > domestic

def test_international_adds_15():
    assert score_transaction(base_tx(is_international=1)) == 15


# ---------------------------------------------------------------------------
# amount_usd
# ---------------------------------------------------------------------------

def test_large_amount_adds_risk():
    assert score_transaction(base_tx(amount_usd=1200)) >= 25

def test_medium_amount_adds_less_than_large():
    medium = score_transaction(base_tx(amount_usd=600))
    large = score_transaction(base_tx(amount_usd=1200))
    assert large > medium

def test_small_amount_adds_nothing():
    assert score_transaction(base_tx(amount_usd=50)) == 0


# ---------------------------------------------------------------------------
# velocity_24h
# ---------------------------------------------------------------------------

def test_high_velocity_increases_score():
    low_v = score_transaction(base_tx(velocity_24h=1))
    high_v = score_transaction(base_tx(velocity_24h=8))
    assert high_v > low_v

def test_very_high_velocity_adds_more_than_medium():
    mid = score_transaction(base_tx(velocity_24h=4))
    high = score_transaction(base_tx(velocity_24h=8))
    assert high > mid

def test_low_velocity_adds_nothing():
    assert score_transaction(base_tx(velocity_24h=1)) == 0


# ---------------------------------------------------------------------------
# failed_logins_24h
# ---------------------------------------------------------------------------

def test_many_failed_logins_increases_score():
    clean = score_transaction(base_tx(failed_logins_24h=0))
    sus = score_transaction(base_tx(failed_logins_24h=6))
    assert sus > clean

def test_moderate_failed_logins_adds_less_than_many():
    moderate = score_transaction(base_tx(failed_logins_24h=3))
    many = score_transaction(base_tx(failed_logins_24h=6))
    assert many > moderate


# ---------------------------------------------------------------------------
# prior_chargebacks
# ---------------------------------------------------------------------------

def test_prior_chargebacks_increase_score():
    clean = score_transaction(base_tx(prior_chargebacks=0))
    one = score_transaction(base_tx(prior_chargebacks=1))
    two = score_transaction(base_tx(prior_chargebacks=2))
    assert one > clean
    assert two > one

def test_two_plus_chargebacks_adds_20():
    assert score_transaction(base_tx(prior_chargebacks=3)) == 20

def test_one_chargeback_adds_5():
    assert score_transaction(base_tx(prior_chargebacks=1)) == 5


# ---------------------------------------------------------------------------
# Score bounds
# ---------------------------------------------------------------------------

def test_score_never_exceeds_100():
    tx = base_tx(
        device_risk_score=90,
        is_international=1,
        amount_usd=2000,
        velocity_24h=10,
        failed_logins_24h=10,
        prior_chargebacks=5,
    )
    assert score_transaction(tx) == 100

def test_score_never_below_zero():
    assert score_transaction(base_tx()) >= 0


# ---------------------------------------------------------------------------
# Combined high-risk profile
# ---------------------------------------------------------------------------

def test_combined_high_risk_profile_scores_high():
    tx = base_tx(
        device_risk_score=80,
        is_international=1,
        amount_usd=1500,
        velocity_24h=7,
        failed_logins_24h=6,
        prior_chargebacks=2,
    )
    assert label_risk(score_transaction(tx)) == "high"
