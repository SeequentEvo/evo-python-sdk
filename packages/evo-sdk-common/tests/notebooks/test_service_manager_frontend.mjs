import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(new URL("../../src/evo/notebooks/assets/service_manager.js", import.meta.url), "utf8");
const { default: { render } } = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

test("browser token retries reread storage and clean up channels and listeners", () => {
    for (const useBroadcastChannel of [true, false]) {
        const channels = [];
        const messages = [];
        const handlers = new Map();
        const values = {
            browser_token_required: true,
            browser_token_channel: "first",
        };
        let storedToken = null;
        globalThis.document = {
            createElement: () => ({
                style: {},
                append() {},
                addEventListener() {},
                removeEventListener() {},
            }),
        };
        globalThis.localStorage = { getItem: () => storedToken };
        globalThis.BroadcastChannel = useBroadcastChannel ? class {
            constructor(name) {
                this.name = name;
                this.closed = false;
                channels.push(this);
            }
            postMessage(message) {
                assert.equal(this.closed, false);
                messages.push(message);
            }
            close() {
                this.closed = true;
            }
        } : undefined;
        const model = {
            get: (key) => values[key],
            send: (message) => messages.push(message),
            on: (event, callback) => handlers.set(event, callback),
            off: (event, callback) => {
                assert.equal(handlers.get(event), callback);
                handlers.delete(event);
            },
        };
        const cleanup = render({ model, el: { appendChild() {} } });
        try {
            assert.match(messages[0].error, /Sign in first/);
            values.browser_token_channel = "";
            handlers.get("change:browser_token_channel")();
            assert.equal(messages.length, 1);
            storedToken = JSON.stringify({ access_token: "retry-token" });
            values.browser_token_channel = "retry";
            handlers.get("change:browser_token_channel")();
            assert.deepEqual(messages[1], { type: "browser_token", token: "retry-token" });
            if (useBroadcastChannel) {
                assert.equal(channels[0].closed, true);
                assert.equal(channels[1].name, "evo-browser-token-retry");
            }
        } finally {
            cleanup();
            delete globalThis.document;
            delete globalThis.localStorage;
            delete globalThis.BroadcastChannel;
        }
        assert.equal(handlers.size, 0);
        assert.ok(channels.every((channel) => channel.closed));
    }
});
