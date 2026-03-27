/**
 * FeedbackWidget - anywidget implementation
 * Simple progress bar with label and message display
 */

function render({ model, el }) {
    // Create container
    const container = document.createElement("div");
    container.className = "evo-feedback-widget";

    // Label element
    const labelEl = document.createElement("div");
    labelEl.className = "evo-feedback-label";
    labelEl.textContent = model.get("label") || "";

    // Progress bar container
    const progressContainer = document.createElement("div");
    progressContainer.className = "evo-feedback-progress-container";

    // Progress bar fill
    const progressBar = document.createElement("div");
    progressBar.className = "evo-feedback-progress-bar";

    progressContainer.appendChild(progressBar);

    // Message element
    const messageEl = document.createElement("div");
    messageEl.className = "evo-feedback-message";
    messageEl.textContent = model.get("message") || "";

    container.appendChild(labelEl);
    container.appendChild(progressContainer);
    container.appendChild(messageEl);

    el.appendChild(container);

    // Update functions
    function updateLabel() {
        labelEl.textContent = model.get("label") || "";
    }

    function updateProgress() {
        const progress = model.get("progress_value") || 0;
        progressBar.style.width = `${Math.min(100, Math.max(0, progress * 100))}%`;
    }

    function updateMessage() {
        messageEl.textContent = model.get("message") || "";
    }

    // Listen for changes
    model.on("change:label", updateLabel);
    model.on("change:progress_value", updateProgress);
    model.on("change:message", updateMessage);

    // Initial render
    updateLabel();
    updateProgress();
    updateMessage();

    return () => {
        model.off("change:label", updateLabel);
        model.off("change:progress_value", updateProgress);
        model.off("change:message", updateMessage);
    };
}

export default { render };
