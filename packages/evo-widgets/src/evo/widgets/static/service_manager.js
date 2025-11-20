function render({ model, el }) {
  // Create main container
  const container = document.createElement("div");
  container.className = "service-manager-container";
  
  // Create main layout (left column and right column)
  const mainLayout = document.createElement("div");
  mainLayout.className = "main-layout";
  
  // Left column - controls
  const leftColumn = document.createElement("div");
  leftColumn.className = "left-column";
  
  // Header row with logo, button, and loading indicator
  const headerRow = document.createElement("div");
  headerRow.className = "header-row";
  
  // Logo
  const logo = document.createElement("div");
  logo.className = "evo-logo";
  logo.textContent = "EVO";
  headerRow.appendChild(logo);
  
  // Sign in button
  const signInBtn = document.createElement("button");
  signInBtn.className = "sign-in-button evo-button";
  signInBtn.textContent = model.get("button_text");
  signInBtn.onclick = () => {
    model.set("action", "refresh");
    model.save_changes();
  };
  headerRow.appendChild(signInBtn);
  
  // Loading spinner
  const loadingSpinner = document.createElement("div");
  loadingSpinner.className = "loading-spinner evo-spinner";
  loadingSpinner.style.display = model.get("loading") ? "block" : "none";
  headerRow.appendChild(loadingSpinner);
  
  leftColumn.appendChild(headerRow);
  
  // Organization selector
  const orgRow = document.createElement("div");
  orgRow.className = "selector-row";
  
  const orgLabel = document.createElement("label");
  orgLabel.textContent = "Organisation:";
  orgLabel.className = "selector-label evo-label";
  
  const orgSelect = document.createElement("select");
  orgSelect.className = "selector-dropdown evo-dropdown";
  orgSelect.innerHTML = '<option value="">Select Organisation</option>';
  
  const updateOrgs = () => {
    const orgs = model.get("organizations");
    const selectedId = model.get("selected_org_id");
    orgSelect.innerHTML = '<option value="">Select Organisation</option>';
    orgs.forEach(org => {
      const option = document.createElement("option");
      option.value = org.id;
      option.textContent = org.name;
      if (org.id === selectedId) {
        option.selected = true;
      }
      orgSelect.appendChild(option);
    });
    orgSelect.disabled = orgs.length === 0;
  };
  
  orgSelect.onchange = () => {
    model.set("selected_org_id", orgSelect.value);
    model.save_changes();
  };
  
  orgRow.appendChild(orgLabel);
  orgRow.appendChild(orgSelect);
  leftColumn.appendChild(orgRow);
  
  // Hub selector
  const hubRow = document.createElement("div");
  hubRow.className = "selector-row";
  
  const hubLabel = document.createElement("label");
  hubLabel.textContent = "Hub:";
  hubLabel.className = "selector-label evo-label";
  
  const hubSelect = document.createElement("select");
  hubSelect.className = "selector-dropdown evo-dropdown";
  hubSelect.innerHTML = '<option value="">Select Hub</option>';
  
  const updateHubs = () => {
    const hubs = model.get("hubs");
    const selectedCode = model.get("selected_hub_code");
    hubSelect.innerHTML = '<option value="">Select Hub</option>';
    hubs.forEach(hub => {
      const option = document.createElement("option");
      option.value = hub.code;
      option.textContent = hub.name;
      if (hub.code === selectedCode) {
        option.selected = true;
      }
      hubSelect.appendChild(option);
    });
    hubSelect.disabled = hubs.length === 0;
  };
  
  hubSelect.onchange = () => {
    model.set("selected_hub_code", hubSelect.value);
    model.save_changes();
  };
  
  hubRow.appendChild(hubLabel);
  hubRow.appendChild(hubSelect);
  leftColumn.appendChild(hubRow);
  
  // Workspace selector
  const workspaceRow = document.createElement("div");
  workspaceRow.className = "selector-row";
  
  const workspaceLabel = document.createElement("label");
  workspaceLabel.textContent = "Workspace:";
  workspaceLabel.className = "selector-label evo-label";
  
  const workspaceSelect = document.createElement("select");
  workspaceSelect.className = "selector-dropdown evo-dropdown";
  workspaceSelect.innerHTML = '<option value="">Select Workspace</option>';
  
  const updateWorkspaces = () => {
    const workspaces = model.get("workspaces");
    const selectedId = model.get("selected_workspace_id");
    workspaceSelect.innerHTML = '<option value="">Select Workspace</option>';
    workspaces.forEach(ws => {
      const option = document.createElement("option");
      option.value = ws.id;
      option.textContent = ws.name;
      if (ws.id === selectedId) {
        option.selected = true;
      }
      workspaceSelect.appendChild(option);
    });
    workspaceSelect.disabled = workspaces.length === 0;
  };
  
  workspaceSelect.onchange = () => {
    model.set("selected_workspace_id", workspaceSelect.value);
    model.save_changes();
  };
  
  workspaceRow.appendChild(workspaceLabel);
  workspaceRow.appendChild(workspaceSelect);
  leftColumn.appendChild(workspaceRow);
  
  // Right column (for future prompt area)
  const rightColumn = document.createElement("div");
  rightColumn.className = "right-column";
  
  mainLayout.appendChild(leftColumn);
  mainLayout.appendChild(rightColumn);
  container.appendChild(mainLayout);
  
  // Add to DOM
  el.appendChild(container);
  
  // Initialize
  updateOrgs();
  updateHubs();
  updateWorkspaces();
  
  // Listen to model changes
  model.on("change:organizations", updateOrgs);
  model.on("change:hubs", updateHubs);
  model.on("change:workspaces", updateWorkspaces);
  
  model.on("change:button_text", () => {
    signInBtn.textContent = model.get("button_text");
  });
  
  model.on("change:loading", () => {
    const loading = model.get("loading");
    loadingSpinner.style.display = loading ? "block" : "none";
    signInBtn.disabled = loading;
  });
  
  model.on("change:selected_org_id", updateOrgs);
  model.on("change:selected_hub_code", updateHubs);
  model.on("change:selected_workspace_id", updateWorkspaces);
}

export default { render };
