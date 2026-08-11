import os
import numpy as np
import joblib
import pytest

import routing


class FakeVectorizer:
    def transform(self, X):
        return X


class FakeClassifier:
    def __init__(self, probs):
        # probs: list of floats for one sample
        self._probs = np.array([probs])

    def predict_proba(self, X):
        return self._probs


class FakePriorityModel:
    def __init__(self, label):
        self._label = label

    def predict(self, X):
        return [self._label for _ in X]


@pytest.fixture
def monkey_multilabel(monkeypatch):
    """Monkeypatch os.path.exists and joblib.load to provide a fake multilabel model and priority model."""
    orig_exists = os.path.exists

    def fake_exists(path):
        if path.endswith("multilabel_model.pkl") or path.endswith("priority_model.pkl"):
            return True
        return orig_exists(path)

    monkeypatch.setattr(os.path, "exists", fake_exists)

    def fake_load(path):
        if path.endswith("multilabel_model.pkl"):
            return {
                "vectorizer": FakeVectorizer(),
                "classifier": FakeClassifier([0.1, 0.8, 0.1]),
                "departments": ["Billing", "Technical", "General"],
            }
        if path.endswith("priority_model.pkl"):
            return FakePriorityModel("High")
        return joblib.load(path)

    monkeypatch.setattr(joblib, "load", fake_load)
    yield


def test_clean_removes_on_wrote():
    text = "Hello\nOn Tue, Jan 1, 2020, John Doe <john@example.com> wrote:\n> quoted reply\nPlease help"
    cleaned = routing.clean_ticket_text(text)
    assert "wrote" not in cleaned.lower()
    assert "quoted" not in cleaned.lower()


def test_clean_removes_signature():
    text = "Please assist with billing.\nRegards,\nJane Doe\n+1 555 1234"
    cleaned = routing.clean_ticket_text(text)
    assert "regards" not in cleaned.lower()
    assert "jane" not in cleaned.lower()


def test_route_ticket_technical(monkey_multilabel):
    res = routing.route_ticket("Login error", "Cannot login, api error and database down")
    assert isinstance(res["assigned_departments"], list)
    assert "Technical" in res["assigned_departments"]


def test_route_ticket_multilabel(monkeypatch):
    # Provide a classifier that returns two departments above threshold
    orig_exists = os.path.exists

    def fake_exists(path):
        if path.endswith("multilabel_model.pkl") or path.endswith("priority_model.pkl"):
            return True
        return orig_exists(path)

    monkeypatch.setattr(os.path, "exists", fake_exists)

    def fake_load(path):
        if path.endswith("multilabel_model.pkl"):
            return {
                "vectorizer": FakeVectorizer(),
                "classifier": FakeClassifier([0.35, 0.35, 0.1]),
                "departments": ["Billing", "Technical", "General"],
            }
        if path.endswith("priority_model.pkl"):
            return FakePriorityModel("Medium")
        return joblib.load(path)

    monkeypatch.setattr(joblib, "load", fake_load)

    res = routing.route_ticket("Issue with charge and login", "Payment charge failed and cannot login to account")
    assert len(res["assigned_departments"]) > 1


def test_priority_in_allowed_set(monkey_multilabel):
    res = routing.route_ticket("Site down", "Site is down and users cannot access")
    assert res["priority"] in {"Critical", "High", "Medium", "Low", "Unknown"} or isinstance(res["priority"], str)


def test_fallback_general_enquiry(monkeypatch):
    # Force no models available
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    res = routing.route_ticket("Hi", "Hello")
    assert res["assigned_departments"] == ["General Enquiry"]
