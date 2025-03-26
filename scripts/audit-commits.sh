#!/bin/bash
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
# Checks a commit to ensure Seequent or Bentley emails were not used

EMAIL=$(git config user.email)

if [[ $(echo $EMAIL | grep -i "@seequent.com") || $(echo $EMAIL | grep -i "@bentley.com") ]]; then
  echo "Please enable 'no-reply' commit settings in your GitHub account"
  echo "See https://docs.github.com/articles/setting-your-email-in-git"
  exit 1
fi
