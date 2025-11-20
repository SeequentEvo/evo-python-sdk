function render({ model, el }) {
  // Create main container
  const container = document.createElement("div");
  container.className = "feedback-container";
  
  // Create label
  const labelEl = document.createElement("div");
  labelEl.className = "feedback-label";
  labelEl.textContent = model.get("label");
  
  // Create progress bar container
  const progressContainer = document.createElement("div");
  progressContainer.className = "progress-container";
  
  // Create progress bar background
  const progressBar = document.createElement("div");
  progressBar.className = "progress-bar";
  
  // Create progress bar fill
  const progressFill = document.createElement("div");
  progressFill.className = "progress-fill";
  
  // Create progress percentage text
  const progressText = document.createElement("div");
  progressText.className = "progress-text";
  progressText.textContent = model.get("progress_percent");
  
  progressBar.appendChild(progressFill);
  progressContainer.appendChild(progressBar);
  progressContainer.appendChild(progressText);
  
  // Create message label
  const messageEl = document.createElement("div");
  messageEl.className = "feedback-message";
  messageEl.textContent = model.get("message");
  
  // Assemble the widget
  container.appendChild(labelEl);
  container.appendChild(progressContainer);
  container.appendChild(messageEl);
  
  el.appendChild(container);
  
  // Update progress bar
  const updateProgress = () => {
    const progress = model.get("progress_value");
    const percent = model.get("progress_percent");
    progressFill.style.width = `${progress * 100}%`;
    progressText.textContent = percent;
  };
  
  // Update message
  const updateMessage = () => {
    messageEl.textContent = model.get("message");
  };
  
  // Update label
  const updateLabel = () => {
    labelEl.textContent = model.get("label");
  };
  
  // Initialize
  updateProgress();
  updateMessage();
  
  // Listen to model changes
  model.on("change:progress_value", updateProgress);
  model.on("change:progress_percent", updateProgress);
  model.on("change:message", updateMessage);
  model.on("change:label", updateLabel);
}

export default { render };
