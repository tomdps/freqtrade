"""Offline security regressions for dependencies used by the exchange client."""

from http.cookies import SimpleCookie

import pytest
from aiohttp import CookieJar
from yarl import URL


@pytest.mark.asyncio
async def test_secure_shared_cookie_stays_off_unencrypted_requests():
    """A cached HTTPS cookie must still respect Secure on the next HTTP request."""
    jar = CookieJar()
    jar.update_cookies(SimpleCookie("secure-cookie=synthetic; Secure; plain-cookie=synthetic;"))

    for scheme in ("http", "https", "http"):
        cookies = jar.filter_cookies(URL(f"{scheme}://example.com/"))
        assert "plain-cookie" in cookies
        assert ("secure-cookie" in cookies) == (scheme == "https")
