from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return HTMLResponse(content="""
    <html>
        <head><title>App Test</title></head>
        <body>
            <h1>✅ Server is Working!</h1>
            <p>If you can see this, the server is accessible.</p>
            <p>The Chainlit app white screen is likely a WebSocket or CORS issue.</p>
        </body>
    </html>
    """)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
