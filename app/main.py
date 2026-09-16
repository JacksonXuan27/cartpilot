from fastapi import FastAPI


app = FastAPI(title="CartPilot")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"service": "cartpilot", "status": "ok"}

