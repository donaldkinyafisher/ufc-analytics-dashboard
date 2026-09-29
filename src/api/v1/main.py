from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.v1.database import Base, SessionLocal, engine
from src.api.v1.routers import events_router, fighters_router, fights_router, ingestion_router
from src.api.v1.services.ingestion_services import fail_stale_jobs

# Start with synchrnours API
Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # A job left queued/running by a previous process can never finish and
    # would block new jobs, so mark it failed on startup.
    with SessionLocal() as db:
        fail_stale_jobs(db)
        db.commit()
    yield


app = FastAPI(title="UFC Analytics API", version="1.0.0", lifespan=lifespan)


app.include_router(fighters_router.router, prefix="/api/v1/fighters", tags=["fighters"])
app.include_router(events_router.router, prefix="/api/v1/events", tags=["events"])
app.include_router(fights_router.router, prefix="/api/v1/fights", tags=["fights"])
app.include_router(ingestion_router.router, prefix="/api/v1/ingestion", tags=["ingestion"])

@app.get("/", include_in_schema=False, name="home")
def home():
    return {"message": "Backend is running!"}

@app.get("/health", name='health')
async def health():
    return {"status": "ok"}

## Exception Handling
