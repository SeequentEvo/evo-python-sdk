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

    const channelId = model.get("browser_token_channel");
    const tokenChannel = channelId && typeof BroadcastChannel !== "undefined"
        ? new BroadcastChannel(`evo-browser-token-${channelId}`) : null;
    const sendBrowserToken = (message) => {
        if (tokenChannel) {
            tokenChannel.postMessage(message);
        } else {
            model.send(message);
        }
    };

    if (model.get("browser_token_required")) {
        try {
            const stored = localStorage.getItem("accessToken");
            if (!stored) {
                throw new Error("No access token found in browser local storage. Sign in first.");
            }
            const token = JSON.parse(stored)?.access_token;
            if (typeof token !== "string" || !token) {
                throw new Error("The browser access token document is invalid.");
            }
            sendBrowserToken({ type: "browser_token", token });
        } catch (error) {
            sendBrowserToken({ type: "browser_token", error: error instanceof SyntaxError
                ? "The browser access token document is invalid."
                : error instanceof Error ? error.message : "Cannot read the browser access token." });
        }
    }

    return () => {
        button.removeEventListener("click", onClick);
        model.off("change:button_text", renderButtonText);
        model.off("change:disabled", renderDisabled);
        model.off("change:loading", renderLoading);
        model.off("change:message", renderMessage);
        tokenChannel?.close();
    };
}

export default { render };
