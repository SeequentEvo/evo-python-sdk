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

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(new URL("../../src/evo/notebooks/assets/service_manager.js", import.meta.url), "utf8");
const { default: { render } } = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

test("renders service manager state and sends clicks to the widget", () => {
    const elements = [];
    const handlers = new Map();
    const messages = [];
    const values = {
        logo: "logo-data",
        spinner: "spinner-data",
        button_text: "Sign In",
        disabled: false,
        loading: false,
        message: "Ready",
    };
    const originalDocument = globalThis.document;
    globalThis.document = {
        createElement: (tag) => {
            const element = {
                tag,
                style: {},
                listeners: new Map(),
                append(...children) {
                    this.children = children;
                },
                addEventListener(event, callback) {
                    this.listeners.set(event, callback);
                },
                removeEventListener(event, callback) {
                    assert.equal(this.listeners.get(event), callback);
                    this.listeners.delete(event);
                },
            };
            elements.push(element);
            return element;
        },
    };

    const model = {
        get: (key) => values[key],
        send: (message) => messages.push(message),
        on: (event, callback) => handlers.set(event, callback),
        off: (event, callback) => {
            assert.equal(handlers.get(event), callback);
            handlers.delete(event);
        },
    };
    let root;

    try {
        const cleanup = render({ model, el: { appendChild: (element) => { root = element; } } });
        const [, logo, button, spinner, message] = elements;

        assert.equal(root.className, "evo-row");
        assert.equal(logo.src, "logo-data");
        assert.equal(button.textContent, "Sign In");
        assert.equal(button.disabled, false);
        assert.equal(spinner.style.display, "none");
        assert.equal(message.textContent, "Ready");

        button.listeners.get("click")();
        assert.deepEqual(messages, [{ type: "click" }]);

        values.button_text = "Refresh Evo Services";
        handlers.get("change:button_text")();
        values.loading = true;
        handlers.get("change:loading")();
        values.message = "Loaded";
        handlers.get("change:message")();
        assert.equal(button.textContent, "Refresh Evo Services");
        assert.equal(spinner.src, "spinner-data");
        assert.equal(spinner.style.display, "inline-block");
        assert.equal(message.textContent, "Loaded");

        cleanup();
        assert.equal(button.listeners.size, 0);
        assert.equal(handlers.size, 0);
    } finally {
        if (originalDocument === undefined) {
            delete globalThis.document;
        } else {
            globalThis.document = originalDocument;
        }
    }
});
