/**
 * DropdownSelectorWidget - anywidget implementation
 * Generic dropdown selector with loading state indicator
 */

const LOADING_GIF = "data:image/gif;base64,R0lGODlhuQEjAfQAAP///+fn587Ozr6+vrKyspqamo6OjoKCgnV1dWlpaVlZWVFRUUFBQT09PTk5OTU1Nf4BAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAQFAAAAIf8LTkVUU0NBUEUyLjADAQAAACwAAAAAuQEjAQAF/iAgjmRpnmiqrmzrvnAsz3Rt33iu73zv/8CgcEgsGo/IpHLJbDqf0Kh0Sq1ar9isdsvter/gsHhMLpvP6LR6zW673/C4fE6v2+/4vH7P7/v/gIGCg4SFhoeIiYqLjI2Oj5CRkpOUlZaXmJmam5ydnp+goaKjpKWmp6ipqqusra6vsLGys7S1tre4ubq7vL2+v8DBwsPExcbHyMnKy8zNzs/Q0dLT1NXW19jZ2tvc3d7f4OHi4+Tl5ufo6teleading...";

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
