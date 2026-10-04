"""Per-user Hub proxy boundary; standalone mode remains local development only."""
import secrets
from urllib.parse import urlsplit

from starlette.responses import JSONResponse


class BackendAccessMiddleware:
    def __init__(self, app, *, hub_mode=False, proxy_token=""):
        self.app = app
        self.hub_mode = hub_mode
        self.proxy_token = proxy_token

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        supplied = headers.get(b"x-mona-proxy-token", b"")
        denied = self.hub_mode and (
            not self.proxy_token or not secrets.compare_digest(supplied, self.proxy_token.encode())
        )
        # Hub's authenticated proxy is the trust boundary. Local development
        # additionally rejects browser cross-origin access, including WebSockets.
        origin = headers.get(b"origin")
        if not self.hub_mode and origin:
            try:
                origin_host = urlsplit(origin.decode("latin1")).netloc
                denied = origin_host != headers.get(b"host", b"").decode("latin1")
            except ValueError:
                denied = True
        if denied:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            else:
                await JSONResponse({"error": "Backend access denied"}, status_code=403)(scope, receive, send)
            return
        await self.app(scope, receive, send)
