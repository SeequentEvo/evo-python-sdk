/**
 * ServiceManagerWidget - anywidget implementation
 * Main authentication/discovery widget with sign-in and cascading dropdowns
 */

const EVO_LOGO = "data:image/png;base64,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";
const LOADING_GIF = "data:image/gif;base64,R0lGODlhuQEjAfQAAP///+fn587Ozr6+vrKyspqamo6OjoKCgnV1dWlpaVlZWVFRUUFBQT09PTk5OTU1Nf4BAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAQFAAAAIf8LTkVUU0NBUEUyLjADAQAAACwAAAAAuQEjAQAF/iAgjmRpnmiqrmzrvnAsz3Rt33iu73zv/8CgcEgsGo/IpHLJbDqf0Kh0Sq1ar9isdsvter/gsHhMLpvP6LR6zW673/C4fE6v2+/4vH7P7/v/gIGCg4SFhoeIiYqLjI2Oj5CRkpOUlZaXmJmam5ydnp+goaKjpKWmp6ipqqusra6vsLGys7S1tre4ubq7vL2+v8DBwsPExcbHyMnKy8zNzs/Q0dLT1NXW19jZ2tvc3d7f4OHi4+Tl5ufo6err7O3u7/Dx8vP09fb3+Pn6+/z9/v8AAwocSLCgwYMIEypcyLChw4cQI0qcSLGixYsYM2rcyLGjx48gQ4ocSbKkyZMo";

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
