from evo.oauth.authorizer import AccessTokenAuthorizer


class TestAccessTokenAuthorizer:
    def test_get_default_headers(self) -> None:
        authorizer = AccessTokenAuthorizer(access_token="abc-123")
        headers = authorizer.get_default_headers()
        assert headers == {"Authorization": "Bearer abc-123"}

    def test_refresh_token(self) -> None:
        authorizer = AccessTokenAuthorizer(access_token="abc-123")
        assert not authorizer.refresh_token()
