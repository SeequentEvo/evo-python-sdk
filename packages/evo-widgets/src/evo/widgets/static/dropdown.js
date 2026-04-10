/**
 * DropdownSelectorWidget - anywidget implementation
 * Generic dropdown selector with loading state indicator
 */

const LOADING_GIF = new URL("./loading.gif", import.meta.url).href;

function render({ model, el }) {
    // Create container
    const container = document.createElement("div");
    container.className = "evo-dropdown-widget";

    // Label element
    const labelEl = document.createElement("label");
    labelEl.className = "evo-dropdown-label";
    labelEl.textContent = model.get("label") || "";

    // Select element
    const select = document.createElement("select");
    select.className = "evo-dropdown-select";
    select.disabled = model.get("disabled") || false;

    // Loading indicator
    const loading = document.createElement("img");
    loading.className = "evo-dropdown-loading";
    loading.src = LOADING_GIF;
    loading.alt = "Loading...";

    container.appendChild(labelEl);
    container.appendChild(select);
    container.appendChild(loading);

    el.appendChild(container);

    // Update functions
    function updateLabel() {
        labelEl.textContent = model.get("label") || "";
    }

    function updateOptions() {
        const options = model.get("options") || [];
        const currentValue = model.get("value");

        select.innerHTML = "";
        options.forEach(([text, value]) => {
            const option = document.createElement("option");
            option.textContent = text;
            option.value = JSON.stringify(value);
            if (JSON.stringify(value) === JSON.stringify(currentValue)) {
                option.selected = true;
            }
            select.appendChild(option);
        });
    }

    function updateDisabled() {
        select.disabled = model.get("disabled") || false;
    }

    function updateLoading() {
        const isLoading = model.get("loading") || false;
        loading.className = isLoading ? "evo-dropdown-loading visible" : "evo-dropdown-loading";
        if (isLoading) {
            select.disabled = true;
        }
    }

    // Handle selection change
    select.addEventListener("change", () => {
        if (!select.value) return;  // Guard against empty value
        const selectedValue = JSON.parse(select.value);
        model.set("value", selectedValue);
        model.save_changes();
    });

    // Listen for model changes
    model.on("change:label", updateLabel);
    model.on("change:options", updateOptions);
    model.on("change:value", updateOptions);
    model.on("change:disabled", updateDisabled);
    model.on("change:loading", updateLoading);

    // Initial render
    updateLabel();
    updateOptions();
    updateDisabled();
    updateLoading();

    return () => {
        model.off("change:label", updateLabel);
        model.off("change:options", updateOptions);
        model.off("change:value", updateOptions);
        model.off("change:disabled", updateDisabled);
        model.off("change:loading", updateLoading);
    };
}

export default { render };
