/*
 * Copyright © 2026 Bentley Systems, Incorporated
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

function render({ model, el }) {
    const root = document.createElement("div");
    root.className = "evo-row";

    const logo = document.createElement("img");
    logo.className = "evo-logo";
    logo.alt = "Evo";
    logo.src = model.get("logo");

    const button = document.createElement("button");
    button.className = "evo-button";
    button.type = "button";

    const spinner = document.createElement("img");
    spinner.className = "evo-spinner";
    spinner.alt = "";

    const message = document.createElement("span");
    message.className = "evo-message";

    root.append(logo, button, spinner, message);
    el.appendChild(root);

    const renderButtonText = () => {
        button.textContent = model.get("button_text");
    };

    const renderDisabled = () => {
        button.disabled = model.get("disabled");
    };

    const renderLoading = () => {
        const loading = model.get("loading");
        spinner.src = loading ? model.get("spinner") : "";
        spinner.style.display = loading ? "inline-block" : "none";
    };

    const renderMessage = () => {
        message.textContent = model.get("message");
    };

    const onClick = () => {
        model.send({ type: "click" });
    };

    button.addEventListener("click", onClick);
    model.on("change:button_text", renderButtonText);
    model.on("change:disabled", renderDisabled);
    model.on("change:loading", renderLoading);
    model.on("change:message", renderMessage);

    renderButtonText();
    renderDisabled();
    renderLoading();
    renderMessage();

    return () => {
        button.removeEventListener("click", onClick);
        model.off("change:button_text", renderButtonText);
        model.off("change:disabled", renderDisabled);
        model.off("change:loading", renderLoading);
        model.off("change:message", renderMessage);
    };
}

export default { render };
