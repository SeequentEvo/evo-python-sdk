import unittest

from evo.oauth.authorizer import AccessTokenAuthorizer


class TestAccessTokenAuthorizer(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.authorizer = AccessTokenAuthorizer(access_token="abc-123")

    async def test_get_default_headers(self) -> None:
        headers = await self.authorizer.get_default_headers()
        assert headers == {"Authorization": "Bearer abc-123"}

    async def test_refresh_token(self) -> None:
        assert not await self.authorizer.refresh_token()
