function render({ model, el }) {
    const root = document.createElement("div");
    root.className = "evo-row";

    const select = document.createElement("select");
    select.className = "evo-select";
    select.id = `evo-select-${Math.random().toString(36).slice(2)}`;

    const label = document.createElement("label");
    label.className = "evo-label";
    label.htmlFor = select.id;

    const spinner = document.createElement("img");
    spinner.className = "evo-spinner";
    spinner.alt = "";

    root.append(label, select, spinner);
    el.appendChild(root);

    const renderLabel = () => {
        label.textContent = model.get("label");
    };

    const renderOptions = () => {
        select.replaceChildren();
        for (const [text, value] of model.get("options") ?? []) {
            const option = document.createElement("option");
            option.value = value;
            option.textContent = text;
            select.appendChild(option);
        }
        select.value = model.get("value");
    };

    const renderValue = () => {
        select.value = model.get("value");
    };

    const renderDisabled = () => {
        select.disabled = model.get("disabled");
    };

    const renderLoading = () => {
        const loading = model.get("loading");
        spinner.src = loading ? model.get("spinner") : "";
        spinner.style.display = loading ? "inline-block" : "none";
    };

    const onChange = () => {
        model.set("value", select.value);
        model.save_changes();
    };

    select.addEventListener("change", onChange);
    model.on("change:label", renderLabel);
    model.on("change:options", renderOptions);
    model.on("change:value", renderValue);
    model.on("change:disabled", renderDisabled);
    model.on("change:loading", renderLoading);

    renderLabel();
    renderOptions();
    renderDisabled();
    renderLoading();

    return () => {
        select.removeEventListener("change", onChange);
        model.off("change:label", renderLabel);
        model.off("change:options", renderOptions);
        model.off("change:value", renderValue);
        model.off("change:disabled", renderDisabled);
        model.off("change:loading", renderLoading);
    };
}

export default { render };
