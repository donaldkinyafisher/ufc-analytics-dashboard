"""Win predictions for every fight on an event's card.

Features come from src.ml.dataset, the same code used to build the training
set, so each fighter's career stats cover their full stored history.
Fights with a UFC debutant (no fights stored for that fighter) get no
prediction. Predictions are computed on request and not stored.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from src.analytics.extract import load_raw_tables
from src.api.v1.models import Event, Fight
from src.api.v1.services.ml_services import best_model_name
from src.ml.dataset import fight_features
from src.ml.predictor import predict_fights


class NoTrainedModelError(LookupError):
    pass


def predict_event(db: Session, event: Event, model_name: str | None = None) -> dict:
    """Predictions for the event's card, in card order (1 = main event).

    model_name defaults to the best trained model. Raises FileNotFoundError
    for a model that isn't trained and NoTrainedModelError if none is.
    """
    model_name = model_name or best_model_name()
    if model_name is None:
        raise NoTrainedModelError("No trained models; start a training job first.")

    fights = db.scalars(
        select(Fight)
        .where(Fight.event_id == event.id)
        .options(selectinload(Fight.red_fighter), selectinload(Fight.blue_fighter))
        .order_by(Fight.bout_order.asc().nulls_last(), Fight.id)
    ).all()
    response = {"event": event, "model_name": model_name, "fights": []}
    if not fights:
        return response

    features = fight_features(load_raw_tables(db.get_bind(), include_upcoming=True))
    card = features[features["event_id"] == event.id]
    predictions = predict_fights(card, model_name).set_index("fight_id")
    history = card.set_index("fight_id")[["red_ufc_fights", "blue_ufc_fights"]]

    for fight in fights:
        red_ufc_fights = int(history.at[fight.id, "red_ufc_fights"])
        blue_ufc_fights = int(history.at[fight.id, "blue_ufc_fights"])
        entry = {
            "fight_id": fight.id,
            "bout_order": fight.bout_order,
            "weight_class": fight.weight_class,
            "bout_type": fight.bout_type,
            "is_title_bout": fight.is_title_bout,
            "red_fighter": fight.red_fighter,
            "blue_fighter": fight.blue_fighter,
            "red_ufc_fights": red_ufc_fights,
            "blue_ufc_fights": blue_ufc_fights,
            "predicted_winner": None,
            "red_win_probability": None,
            "blue_win_probability": None,
        }
        if red_ufc_fights > 0 and blue_ufc_fights > 0:
            prediction = predictions.loc[fight.id]
            red_won = prediction["predicted_winner"] == "red"
            entry["predicted_winner"] = fight.red_fighter if red_won else fight.blue_fighter
            entry["red_win_probability"] = float(prediction["red_win_probability"])
            entry["blue_win_probability"] = float(prediction["blue_win_probability"])
        response["fights"].append(entry)
    return response
