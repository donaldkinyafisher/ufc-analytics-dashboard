"""Event predictions route, through the real feature pipeline and a tiny model."""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from src.api.v1.database import get_db
from src.api.v1.main import app
from src.api.v1.models import Event, Fight, Fighter
from src.api.v1.services import predictions_services


@pytest.fixture
def client(session_factory):
    def override_get_db():
        with session_factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def card(db, raw, tiny_model, monkeypatch):
    """An upcoming card on top of the conftest history; the tiny model favours the taller fighter."""
    monkeypatch.setattr(predictions_services, "best_model_name", lambda: tiny_model)
    alpha = db.query(Fighter).filter_by(name="Alpha").one()  # 70 in, 3 earlier fights
    bravo = db.query(Fighter).filter_by(name="Bravo").one()  # 68 in, 2 earlier fights
    delta = Fighter(name="Delta", height_in=76, reach_in=78, stance="Orthodox")  # debut
    event = Event(name="UFC 1000", event_date=date(2030, 2, 1), status="upcoming")
    empty = Event(name="UFC 1001", event_date=date(2030, 3, 1), status="upcoming")
    db.add_all([
        delta, event, empty,
        Fight(event=event, red_fighter=alpha, blue_fighter=bravo, bout_order=2,
              weight_class="Lightweight", bout_type="Lightweight Bout"),
        Fight(event=event, red_fighter=bravo, blue_fighter=delta, bout_order=1, is_title_bout=True,
              weight_class="Lightweight", bout_type="UFC Lightweight Title Bout"),
    ])
    db.commit()
    return {"event_id": event.id, "empty_event_id": empty.id}


def test_predicts_every_fight_in_card_order(client, card):
    response = client.get(f"/api/v1/predictions/events/{card['event_id']}")

    assert response.status_code == 200
    body = response.json()
    assert body["event"]["name"] == "UFC 1000"
    assert body["model_name"] == "tiny"
    main, co_main = body["fights"]
    assert (main["bout_order"], co_main["bout_order"]) == (1, 2)
    assert main["bout_type"] == "UFC Lightweight Title Bout"
    assert main["is_title_bout"] is True
    assert (main["red_fighter"]["name"], main["blue_fighter"]["name"]) == ("Bravo", "Delta")
    assert (main["red_ufc_fights"], main["blue_ufc_fights"]) == (2, 0)
    assert (co_main["red_ufc_fights"], co_main["blue_ufc_fights"]) == (3, 2)
    # Taller fighter wins.
    assert co_main["predicted_winner"]["name"] == "Alpha"
    assert co_main["red_win_probability"] + co_main["blue_win_probability"] == pytest.approx(1)


def test_fight_with_a_debutant_has_no_prediction(client, card):
    main = client.get(f"/api/v1/predictions/events/{card['event_id']}").json()["fights"][0]

    assert main["blue_fighter"]["name"] == "Delta"  # no UFC fights stored
    assert main["blue_ufc_fights"] == 0
    assert (main["predicted_winner"], main["red_win_probability"], main["blue_win_probability"]) == (None, None, None)


def test_event_without_a_card_returns_no_fights(client, card):
    body = client.get(f"/api/v1/predictions/events/{card['empty_event_id']}").json()

    assert body["fights"] == []
    assert body["model_name"] == "tiny"


def test_unknown_event_is_404(client, card):
    response = client.get("/api/v1/predictions/events/9999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Event not found"


def test_untrained_model_is_404(client, card):
    response = client.get(f"/api/v1/predictions/events/{card['event_id']}", params={"model_name": "xgboost"})

    assert response.status_code == 404
    assert "Trained model 'xgboost' was not found" in response.json()["detail"]


def test_unsupported_model_name_is_422(client, card):
    response = client.get(f"/api/v1/predictions/events/{card['event_id']}", params={"model_name": "tiny"})

    assert response.status_code == 422


def test_no_trained_models_is_404(client, card, monkeypatch):
    monkeypatch.setattr(predictions_services, "best_model_name", lambda: None)

    response = client.get(f"/api/v1/predictions/events/{card['event_id']}")

    assert response.status_code == 404
    assert "No trained models" in response.json()["detail"]
