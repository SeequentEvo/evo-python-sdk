function render({ model, el }) {
    const root = document.createElement("div");
    root.className = "evo-row";

    const label = document.createElement("span");
    label.className = "evo-label";

    const percent = document.createElement("span");
    percent.className = "evo-percent";

    const progress = document.createElement("div");
    progress.className = "evo-progress";
    progress.role = "progressbar";

    const bar = document.createElement("div");
    bar.className = "evo-progress-bar";
    progress.appendChild(bar);

    const message = document.createElement("span");
    message.className = "evo-message";

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

    model.on("change:label", renderLabel);
    model.on("change:value", renderValue);
    model.on("change:message", renderMessage);

    renderLabel();
    renderValue();
    renderMessage();

    return () => {
        model.off("change:label", renderLabel);
        model.off("change:value", renderValue);
        model.off("change:message", renderMessage);
    };
}

export default { render };
