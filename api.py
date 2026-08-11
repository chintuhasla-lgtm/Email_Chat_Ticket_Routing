"""FastAPI wrapper exposing /route endpoint to route tickets.

Run with: `uvicorn api:app --reload`
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from routing import route_ticket


class RouteRequest(BaseModel):
    subject: str = ""
    body: str = ""


app = FastAPI()


@app.post("/route")
def route(req: RouteRequest):
    try:
        res = route_ticket(req.subject, req.body)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("Run with: uvicorn api:app --reload")
