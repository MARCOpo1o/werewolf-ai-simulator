async function loadExperiments() {
    const target = document.getElementById('experiment-list');
    try {
        const response = await fetch('/api/experiments');
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'Could not load experiments');
        const experiments = payload.experiments || [];
        if (!experiments.length) {
            target.innerHTML = '<div class="empty-state">No persisted experiments yet. Create and run one with the experiment CLI.</div>';
            return;
        }
        target.replaceChildren(...experiments.map(experiment => {
            const card = document.createElement('article');
            card.className = 'experiment-card';
            const title = document.createElement('h2');
            const link = document.createElement('a');
            link.href = `/experiments/${encodeURIComponent(experiment.experiment_id)}`;
            link.textContent = experiment.experiment_id;
            title.append(link);
            const progress = experiment.progress || {};
            const text = document.createElement('p');
            text.textContent = `${progress.completed || 0} / ${experiment.scheduled_trials || 0} completed · ${experiment.summary_revisions || 0} summary revisions`;
            card.append(title, text);
            return card;
        }));
    } catch (error) {
        target.innerHTML = `<div class="error-state">${error.message}</div>`;
    }
}

loadExperiments();
