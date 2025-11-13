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
"""
Task API
=============


    The Task API provides the ability to execute computation tasks in Evo that require variable, on-demand
    processing power. Building the Task API into your application can enable fast processing of long
    running, or resource intensive operations, without depending on the end user's physical hardware.

    Tasks are created, triggering a job to be executed asynchronously within a specific topic for your organization,
    and can be monitored throughout their execution lifecycle.

    For more information on using the Task API, see [Overview](/docs/guides/tasks), or the API
    references here.

This code is generated from the OpenAPI specification for Task API.
API version: 0.0.2
"""

# Import endpoint apis.
from .api import JobsApi, TasksApi

__all__ = [
    "JobsApi",
    "TasksApi",
]
