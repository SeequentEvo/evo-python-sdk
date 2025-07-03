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

import json
from typing import Any

from parameterized import parameterized

from evo.common import DependencyStatus, HealthCheckType, RequestMethod, ServiceHealth, ServiceStatus
from evo.common.exceptions import ServiceHealthCheckFailed
from evo.common.test_tools import TestWithConnector
from evo.common.utils import get_service_health, get_service_status

from ...data import load_test_data


def using_test_data(full: bool, with_dependencies: bool, strict: bool) -> Any:
    test_data = load_test_data("health_check.json")
    p_input = []
    for scenario in test_data:
        if full:
            content = {
                "status": scenario["status"],
                "version": scenario["version"],
            }
            if with_dependencies and "dependencies" in scenario:
                content["dependencies"] = scenario["dependencies"]
        else:
            content = scenario["status"]

        if strict:
            status_code = 200 if scenario["status"] == "pass" else 503
        else:
            status_code = 200 if scenario["status"] in {"pass", "degraded"} else 503

        p_input.append((scenario["version"], content, status_code))
    return parameterized.expand(p_input)


class TestHealthCheck(TestWithConnector):
    @using_test_data(full=False, with_dependencies=False, strict=False)
    async def test_get_service_status(self, _label: str, content: str, status_code: int) -> None:
        with self.transport.set_http_response(status_code, content):
            status = await get_service_status(self.connector, "test", check_type=HealthCheckType.BASIC)

        self.assert_request_made(RequestMethod.GET, "/test/health_check")
        self.assertEqual(ServiceStatus(content), status)

    @using_test_data(full=False, with_dependencies=True, strict=False)
    async def test_get_service_status_with_dependencies(self, _label: str, content: str, status_code: int) -> None:
        with self.transport.set_http_response(status_code, content):
            status = await get_service_status(self.connector, "test", check_type=HealthCheckType.FULL)

        self.assert_request_made(RequestMethod.GET, "/test/health_check?check_dependencies=True")
        self.assertEqual(ServiceStatus(content), status)

    @using_test_data(full=False, with_dependencies=True, strict=True)
    async def test_get_service_status_with_dependencies_strict(
        self, _label: str, content: str, status_code: int
    ) -> None:
        with self.transport.set_http_response(status_code, content):
            status = await get_service_status(self.connector, "test", check_type=HealthCheckType.STRICT)

        self.assert_request_made(RequestMethod.GET, "/test/health_check?check_dependencies=True&strict=True")
        self.assertEqual(ServiceStatus(content), status)

    def _check_health_check_response(self, content: dict, status_code: int, response: ServiceHealth) -> None:
        self.assertEqual("test", response.service)
        self.assertEqual(status_code, response.status_code)
        self.assertEqual(ServiceStatus(content["status"]), response.status)
        self.assertEqual(content["version"], response.version)

        if "dependencies" in content:
            self.assertIsInstance(response.dependencies, dict)
            for dependency_name, dependency_status in content["dependencies"].items():
                self.assertIn(dependency_name, response.dependencies)
                self.assertEqual(DependencyStatus(dependency_status), response.dependencies[dependency_name])

        if status_code == 200:
            response.raise_for_status()
        else:
            with self.assertRaises(ServiceHealthCheckFailed):
                response.raise_for_status()

    @using_test_data(full=True, with_dependencies=False, strict=False)
    async def test_get_service_health(self, _label: str, content: dict, status_code: int) -> None:
        with self.transport.set_http_response(status_code, json.dumps(content)):
            response = await get_service_health(self.connector, "test", check_type=HealthCheckType.BASIC)

        self.assert_request_made(RequestMethod.GET, "/test/health_check?full=True")
        self._check_health_check_response(content, status_code, response)

    @using_test_data(full=True, with_dependencies=True, strict=False)
    async def test_get_service_health_with_dependencies(self, _label: str, content: dict, status_code: int) -> None:
        with self.transport.set_http_response(status_code, json.dumps(content)):
            response = await get_service_health(self.connector, "test", check_type=HealthCheckType.FULL)

        self.assert_request_made(RequestMethod.GET, "/test/health_check?full=True&check_dependencies=True")
        self._check_health_check_response(content, status_code, response)

    @using_test_data(full=True, with_dependencies=True, strict=True)
    async def test_get_service_health_with_dependencies_strict(
        self, _label: str, content: dict, status_code: int
    ) -> None:
        with self.transport.set_http_response(status_code, json.dumps(content)):
            response = await get_service_health(self.connector, "test", check_type=HealthCheckType.STRICT)

        self.assert_request_made(RequestMethod.GET, "/test/health_check?full=True&check_dependencies=True&strict=True")
        self._check_health_check_response(content, status_code, response)
