"""Routing utilities for incoming tickets.

Provides `route_ticket(subject, body) -> dict` which:
- cleans text using `clean_ticket_text` from `02_text_cleaning.py`
- loads `department_model.pkl`, `priority_model.pkl`, and
  `multilabel_model.pkl` (when available)
- computes department probabilities and assigns queues by threshold
- predicts priority via the priority model

The module is defensive: if models are missing it falls back
to reasonable defaults and reports the routing reason.
"""
from typing import List, Dict
import joblib
import os
import numpy as np

# Reuse cleaning from 02_text_cleaning.py (module name starts with digit,
# so import via importlib by filename to avoid invalid identifier issues)
import importlib.util
import types

def _load_text_cleaner():
    """Dynamically load `02_text_cleaning.py` and return clean_ticket_text, ensure_stopwords.

    Falls back to simple passthrough if file or functions are missing.
    """
    fname = os.path.join(os.path.dirname(__file__), "02_text_cleaning.py")
    if not os.path.exists(fname):
        # fallback no-op cleaner
        def clean_ticket_text(x):
            return x or ""

        def ensure_stopwords():
            return

        return clean_ticket_text, ensure_stopwords

    spec = importlib.util.spec_from_file_location("text_cleaning", fname)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore
    except Exception:
        def clean_ticket_text(x):
            return x or ""

        def ensure_stopwords():
            return

        return clean_ticket_text, ensure_stopwords

    clean_ticket_text = getattr(module, "clean_ticket_text", lambda x: x or "")
    ensure_stopwords = getattr(module, "ensure_stopwords", lambda: None)
    return clean_ticket_text, ensure_stopwords


clean_ticket_text, ensure_stopwords = _load_text_cleaner()


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Compute softmax over last axis for numeric stability."""
    exps = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def _to_probs(model, X):
    """Return probability matrix for model on feature matrix X.

    Supports `predict_proba` directly, or `decision_function` with
    logistic/sigmoid or softmax conversion.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        dec = model.decision_function(X)
        dec = np.atleast_2d(dec)
        # If binary and shape (n_samples, ) or (n_samples, 1), use sigmoid
        if dec.shape[1] == 1:
            probs_pos = 1 / (1 + np.exp(-dec.ravel()))
            return np.vstack([1 - probs_pos, probs_pos]).T
        # multiclass/multilabel: apply softmax per row
        return _softmax(dec)
    raise RuntimeError("Model does not support probability or decision outputs")


def route_ticket(subject: str, body: str) -> Dict[str, object]:
    """Route a ticket: return cleaned text, assigned departments, priority, and reason.

    - Combines subject+body, cleans text using `clean_ticket_text`.
    - Loads available models from the working directory.
    - Uses `multilabel_model.pkl` when possible to get per-department probs.
      Falls back to `department_model.pkl` single-label model if needed.
    - Assigns departments with prob >= 0.3; if none, picks top department
      unless top probability < 0.15, in which case assigns `General Enquiry`.
    - Predicts priority using `priority_model.pkl` when available.
    """
    # Ensure stopwords available for cleaner
    try:
        ensure_stopwords()
    except Exception:
        pass

    combined = " ".join([str(subject or ""), str(body or "")])
    clean = clean_ticket_text(combined)

    # Load multilabel model if present
    multilabel_path = "multilabel_model.pkl"
    department_path = "department_model.pkl"
    priority_path = "priority_model.pkl"

    probs = None
    departments = None
    routing_reason = "no_model_available"

    if os.path.exists(multilabel_path):
        try:
            ml = joblib.load(multilabel_path)
            vec = ml["vectorizer"]
            clf = ml["classifier"]
            departments = ml.get("departments")
            X = vec.transform([clean])
            probs = _to_probs(clf, X)
            # probs shape (1, n_depts)
            probs = probs[0]
            routing_reason = "multilabel_model"
        except Exception:
            probs = None

    # Fallback to single-label department model
    if probs is None and os.path.exists(department_path):
        try:
            pipe = joblib.load(department_path)
            # Expect a sklearn Pipeline with vectorizer + estimator
            if hasattr(pipe, "predict_proba") or hasattr(pipe, "decision_function"):
                # Get class labels from pipeline
                if hasattr(pipe, "classes_"):
                    departments = list(pipe.classes_)
                    X = [clean]
                    if hasattr(pipe, "predict_proba"):
                        probs_mat = pipe.predict_proba(X)
                    else:
                        dec = pipe.decision_function(X)
                        dec = np.atleast_2d(dec)
                        probs_mat = _softmax(dec)
                    probs = probs_mat[0]
                    routing_reason = "singlelabel_model"
        except Exception:
            probs = None

    assigned = []
    if probs is not None and departments is not None:
        # Map departments to probabilities
        dept_probs = list(zip(departments, map(float, probs)))
        # Threshold assignment
        assigned = [d for d, p in dept_probs if p >= 0.3]
        if assigned:
            routing_reason = "threshold_0.3"
        else:
            # pick top department
            top_idx = int(np.argmax(probs))
            top_dept = departments[top_idx]
            top_prob = float(probs[top_idx])
            if top_prob < 0.15:
                assigned = ["General Enquiry"]
                routing_reason = "fallback_general_enquiry"
            else:
                assigned = [top_dept]
                routing_reason = "fallback_top_prob"
    else:
        # No probabilities available at all
        assigned = ["General Enquiry"]
        routing_reason = "no_department_model"

    # Priority prediction
    priority = "Unknown"
    if os.path.exists(priority_path):
        try:
            pr_pipe = joblib.load(priority_path)
            # If priority model expects raw text, call predict
            priority = str(pr_pipe.predict([clean])[0])
            routing_reason = routing_reason + ";priority_model_used"
        except Exception:
            # fall back to simple keyword-based mapping
            if any(k in clean for k in ["urgent", "immediately", "asap", "down"]):
                priority = "High"
            else:
                priority = "Medium"
    else:
        # no priority model available
        if any(k in clean for k in ["urgent", "immediately", "asap", "down"]):
            priority = "High"
            routing_reason = routing_reason + ";priority_rule"
        else:
            priority = "Low"

    return {
        "clean_text": clean,
        "assigned_departments": assigned,
        "priority": priority,
        "routing_reason": routing_reason,
    }


if __name__ == "__main__":
    # Example ticket
    subj = "Website down - can't access account"
    body = (
        "Our site is down since 10:00 UTC. Customers cannot log in. Please fix immediately. "
        "Contact: john@example.com"
    )
    result = route_ticket(subj, body)
    print(result)
