"""
Tests for services/ml_service.py — proxy engagement labels and the
user rating store used to train the Random Forest engagement model.
"""

import pandas as pd
import pytest

from services import ml_service


def _features(avg_quality=0.5, avg_energy=0.1, slide_std=0.5, blur_mean=750):
    return pd.Series({
        "avg_quality": avg_quality,
        "avg_energy": avg_energy,
        "slide_std": slide_std,
        "blur_mean": blur_mean,
    })


def test_proxy_engagement_score_is_within_valid_range():
    score = ml_service._proxy_engagement_score(_features())
    assert 1.0 <= score <= 5.0


def test_proxy_engagement_score_rewards_higher_quality():
    low = ml_service._proxy_engagement_score(_features(avg_quality=0.1))
    high = ml_service._proxy_engagement_score(_features(avg_quality=0.9))
    assert high > low


def test_save_rating_rejects_out_of_range(tmp_path, monkeypatch):
    monkeypatch.setattr(ml_service, "RATINGS_CSV", str(tmp_path / "ratings.csv"))

    with pytest.raises(ValueError):
        ml_service.save_rating("reel-1", 6)
    with pytest.raises(ValueError):
        ml_service.save_rating("reel-1", 0)


def test_save_rating_roundtrip_and_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(ml_service, "RATINGS_CSV", str(tmp_path / "ratings.csv"))

    ml_service.save_rating("reel-1", 3)
    ml_service.save_rating("reel-1", 5)  # re-rating the same reel should overwrite, not duplicate

    ratings = ml_service.load_ratings()
    assert len(ratings) == 1
    assert int(ratings.iloc[0]["rating"]) == 5


def test_get_rating_map(tmp_path, monkeypatch):
    monkeypatch.setattr(ml_service, "RATINGS_CSV", str(tmp_path / "ratings.csv"))

    ml_service.save_rating("reel-1", 4)

    assert ml_service.get_rating_map() == {"reel-1": 4}


def test_load_ratings_missing_file_returns_empty_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(ml_service, "RATINGS_CSV", str(tmp_path / "missing.csv"))

    ratings = ml_service.load_ratings()
    assert ratings.empty
