#  Copyright © 2025 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import pytest

from evo.common.utils import urls


#  fmt: off
@pytest.mark.parametrize(
    "input,expected",
    [

        ("https://example.com/foo/bar",         "https://example.com"),
        ("https://example.com/foo/bar?a=b&c=d", "https://example.com"),
        ("http://example.com/foo/bar",          "http://example.com"),
        ("http://example.com/foo/bar?a=b&c=d",  "http://example.com"),
        ("https://example.com:5000/foo/bar",    "https://example.com:5000"),
        ("http://example.com:5000/foo/bar",     "http://example.com:5000"),
        
    ],
)
# fmt: on
def test_url_parsed_with_insecure_allowed(input: str, expected: str):

    assert urls.load_base_url(input, allow_insecure=True) == expected


def test_disallowed_http_if_not_allow_insecure():
    assert urls.load_base_url("https://example.com:9000/foo", allow_insecure=False) == "https://example.com:9000"
    with pytest.raises(ValueError):
        urls.load_base_url("http://example.com:9000/foo", allow_insecure=False)
