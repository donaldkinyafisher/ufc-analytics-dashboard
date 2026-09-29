from datetime import date

from pydantic import BaseModel, Field


class BaseFight(BaseModel):
    red_fighter:str = Field()
    blue_fighter: str = Field()
    weight_class: str | None = None
    location: str| None = None
    source_url: str | None = None
    fight_date: date | None = None

class UpcomingFight(BaseFight):
    pass
    
class PastFight(BaseFight):
    pass 

    