/**
 * ServiceManagerWidget - anywidget implementation
 * Main authentication/discovery widget with sign-in and cascading dropdowns
 */

const EVO_LOGO = new URL("./evo-logo.png", import.meta.url).href;
const LOADING_GIF = new URL("./loading.gif", import.meta.url).href;

function createDropdown(label, id, model, valueKey, optionsKey, loadingKey) {
    const container = document.createElement("div");
    container.className = "evo-sm-dropdown";

    const labelEl = document.createElement("label");
    labelEl.className = "evo-sm-dropdown-label";
    labelEl.textContent = label;

    const select = document.createElement("select");
    select.className = "evo-sm-dropdown-select";
    select.id = id;
    select.disabled = true;

    const loading = document.createElement("img");
    loading.className = "evo-sm-dropdown-loading";
    loading.src = LOADING_GIF;
    loading.alt = "Loading...";

    container.appendChild(labelEl);
    container.appendChild(select);
    container.appendChild(loading);

    function updateOptions() {
        const options = model.get(optionsKey) || [];
        const currentValue = model.get(valueKey);

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

        // Disable if only one option (the placeholder)
        select.disabled = options.length <= 1;
    }

    function updateLoading() {
        const isLoading = model.get(loadingKey) || false;
        loading.className = isLoading ? "evo-sm-dropdown-loading visible" : "evo-sm-dropdown-loading";
        if (isLoading) {
            select.disabled = true;
        }
    }

    select.addEventListener("change", () => {
        if (!select.value) return;  // Guard against empty value
        const selectedValue = JSON.parse(select.value);
        model.set(valueKey, selectedValue);
        model.save_changes();
    });

    return {
        element: container,
        updateOptions,
        updateLoading,
        select
    };
}

function render({ model, el }) {
    // Create main container
    const container = document.createElement("div");
    container.className = "evo-service-manager-widget";

    // Column 1 - Logo, button, selectors
    const col1 = document.createElement("div");
    col1.className = "evo-service-manager-col1";

    // Header row with logo, button, loading
    const header = document.createElement("div");
    header.className = "evo-service-manager-header";

    const logo = document.createElement("img");
    logo.className = "evo-service-manager-logo";
    logo.src = EVO_LOGO;
    logo.alt = "Evo";

    const btn = document.createElement("button");
    btn.className = "evo-service-manager-btn";
    btn.textContent = model.get("button_text") || "Sign In";

    const mainLoading = document.createElement("img");
    mainLoading.className = "evo-service-manager-loading";
    mainLoading.src = LOADING_GIF;
    mainLoading.alt = "Loading...";

    header.appendChild(logo);
    header.appendChild(btn);
    header.appendChild(mainLoading);

    // Create dropdowns
    const orgDropdown = createDropdown("Organisation", "org-select", model, "org_value", "org_options", "org_loading");
    const wsDropdown = createDropdown("Workspace", "ws-select", model, "ws_value", "ws_options", "ws_loading");

    col1.appendChild(header);
    col1.appendChild(orgDropdown.element);
    col1.appendChild(wsDropdown.element);

    // Column 2 - Prompt area
    const col2 = document.createElement("div");
    col2.className = "evo-service-manager-col2";

    const promptArea = document.createElement("div");
    promptArea.className = "evo-service-manager-prompt";
    col2.appendChild(promptArea);

    container.appendChild(col1);
    container.appendChild(col2);
    el.appendChild(container);

    // Update functions
    function updateButtonText() {
        btn.textContent = model.get("button_text") || "Sign In";
    }

    function updateButtonDisabled() {
        btn.disabled = model.get("button_disabled") || false;
    }

    function updateMainLoading() {
        const isLoading = model.get("main_loading") || false;
        mainLoading.className = isLoading ? "evo-service-manager-loading visible" : "evo-service-manager-loading";
    }

    function updatePrompt() {
        const promptText = model.get("prompt_text") || "";
        const showPrompt = model.get("show_prompt") || false;
        promptArea.textContent = promptText;
        promptArea.className = showPrompt ? "evo-service-manager-prompt visible" : "evo-service-manager-prompt";
    }

    // Button click handler
    btn.addEventListener("click", () => {
        model.set("button_clicked", true);
        model.save_changes();
    });

    // Initial updates
    updateButtonText();
    updateButtonDisabled();
    updateMainLoading();
    updatePrompt();
    orgDropdown.updateOptions();
    orgDropdown.updateLoading();
    wsDropdown.updateOptions();
    wsDropdown.updateLoading();

    // Listen for changes
    model.on("change:button_text", updateButtonText);
    model.on("change:button_disabled", updateButtonDisabled);
    model.on("change:main_loading", updateMainLoading);
    model.on("change:prompt_text", updatePrompt);
    model.on("change:show_prompt", updatePrompt);
    model.on("change:org_options", orgDropdown.updateOptions);
    model.on("change:org_loading", orgDropdown.updateLoading);
    model.on("change:org_value", orgDropdown.updateOptions);
    model.on("change:ws_options", wsDropdown.updateOptions);
    model.on("change:ws_loading", wsDropdown.updateLoading);
    model.on("change:ws_value", wsDropdown.updateOptions);

    return () => {
        model.off("change:button_text", updateButtonText);
        model.off("change:button_disabled", updateButtonDisabled);
        model.off("change:main_loading", updateMainLoading);
        model.off("change:prompt_text", updatePrompt);
        model.off("change:show_prompt", updatePrompt);
        model.off("change:org_options", orgDropdown.updateOptions);
        model.off("change:org_loading", orgDropdown.updateLoading);
        model.off("change:org_value", orgDropdown.updateOptions);
        model.off("change:ws_options", wsDropdown.updateOptions);
        model.off("change:ws_loading", wsDropdown.updateLoading);
        model.off("change:ws_value", wsDropdown.updateOptions);
    };
}

export default { render };
