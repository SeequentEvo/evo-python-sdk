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

function createDomElement(tag, properties = {}) {
    const element = document.createElement(tag);
    Object.assign(element, properties);
    return element;
}

function addListeners(model, listeners) {
    const entries = listeners.flatMap((listenerMap) => Object.entries(listenerMap));
    for (const [event, handler] of entries) {
        model.on(event, handler);
    }
    return () => {
        for (const [event, handler] of entries) {
            model.off(event, handler);
        }
    };
}

function render({ model, el }) {
    const root = createDomElement("div", { className: "evo-row" });

    const label = createDomElement("span", { className: "evo-label" });

    const percent = createDomElement("span", { className: "evo-percent" });

    const progress = createDomElement("div", { className: "evo-progress", role: "progressbar" });

    const bar = createDomElement("div", { className: "evo-progress-bar" });
    progress.appendChild(bar);

    const message = createDomElement("span", { className: "evo-message" });

    root.append(label, percent, progress, message);
    el.appendChild(root);

    const renderLabel = () => {
        label.textContent = model.get("label");
    };

    const renderValue = () => {
        const value = Math.min(Math.max(model.get("value"), 0), 1);
        bar.style.width = `${value * 100}%`;
        percent.textContent = `${(value * 100).toFixed(1).padStart(5)}%`;
        progress.ariaValueNow = String(Math.round(value * 100));
    };

    const renderMessage = () => {
        message.textContent = model.get("message");
    };

    const removeListeners = addListeners(model, [
        {
            "change:label": renderLabel,
            "change:value": renderValue,
            "change:message": renderMessage,
        },
    ]);

    renderLabel();
    renderValue();
    renderMessage();

    return () => {
        removeListeners();
    };
}

export default { render };
