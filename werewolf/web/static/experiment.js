async function loadExperiment() {
    const root = document.getElementById('experiment-report');
    const experimentId = document.body.dataset.experimentId;
    try {
        const response = await fetch(`/api/experiments/${encodeURIComponent(experimentId)}`);
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'Could not load experiment');
        const manifest = payload.manifest || {};
        const catalog = payload.summary_catalog || {};
        root.replaceChildren();
        const heading = document.createElement('h1');
        heading.textContent = manifest.experiment_id;
        const description = document.createElement('p');
        description.className = 'lede';
        description.textContent = manifest.description || 'No hypothesis or description recorded.';
        const detail = document.createElement('pre');
        detail.className = 'experiment-json';
        detail.textContent = JSON.stringify({
            execution_contract_sha256: payload.index_entry?.manifest_content_sha256,
            progress: payload.index_entry?.progress,
            summary_revisions: catalog.revisions || [],
        }, null, 2);
        root.append(heading, description, detail);
    } catch (error) {
        root.innerHTML = `<div class="error-state">${error.message}</div>`;
    }
}

loadExperiment();
