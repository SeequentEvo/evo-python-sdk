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
#
# Installs a version of uv matching the tracked version in UV_VERSION file.
# Will replace existing versions already installed. Expected to be run from repo root.

$TARGET_VERSION = Get-Content -Path .\UV_VERSION

Write-Host "Installing uv version $TARGET_VERSION"

powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/$TARGET_VERSION/install.ps1 | iex"
