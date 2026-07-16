const experimentId = document.body.dataset.experimentId;
const state = { detail: null, summary: null, revision: null, view: 'all_completed' };

const metricNames = {
    village_win_rate: 'Village win rate', wolf_win_rate: 'Wolf win rate',
    clean_game_rate: 'Clean game rate', fallback_game_rate: 'Fallback game rate',
    retry_rate: 'Retry rate', repair_rate: 'Repair rate',
    vote_belief_alignment: 'Vote-belief alignment', harmful_revision: 'Harmful revision',
    correct_belief_retention: 'Correct-belief retention',
    probability_movement_toward_wolves: 'Movement toward wolves',
    wolf_suspicion_awareness_error: 'Wolf awareness error',
    brier_post_discussion: 'Post-discussion Brier score',
};

function value(value, digits = 3) {
    if (value === null || value === undefined || value === '') return '—';
    return typeof value === 'number' ? value.toFixed(digits) : String(value);
}
function percent(value) { return value === null || value === undefined ? '—' : `${(Number(value) * 100).toFixed(1)}%`; }
function money(value) { return value === null || value === undefined ? 'Unavailable' : `$${Number(value).toFixed(4)}`; }
function escapeText(value) { return String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]); }
function cell(row, text) { const td = document.createElement('td'); td.textContent = text; row.append(td); return td; }
function clear(id) { const node = document.getElementById(id); node.replaceChildren(); return node; }
function card(label, main, detail = '') { const node = document.createElement('article'); node.className = 'metric-card'; node.innerHTML = `<span>${escapeText(label)}</span><strong>${escapeText(main)}</strong>${detail ? `<small>${escapeText(detail)}</small>` : ''}`; return node; }
function metricEstimate(metric, formatter = percent) { return metric && metric.estimate !== undefined && metric.estimate !== null ? formatter(metric.estimate) : '—'; }
function interval(metric, formatter = percent) { if (!metric || metric.ci_low === null || metric.ci_low === undefined) return metric?.interval_status === 'insufficient_clusters' ? 'Insufficient seed clusters' : '—'; return `${formatter(metric.ci_low)} to ${formatter(metric.ci_high)}`; }

async function fetchJson(path) {
    const response = await fetch(path);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
    return payload;
}

function renderDetail() {
    const { detail } = state;
    const manifest = detail.manifest || {};
    const progress = detail.index_entry?.progress || {};
    document.getElementById('experiment-title').textContent = manifest.experiment_id || experimentId;
    document.getElementById('experiment-description').textContent = manifest.description || 'No hypothesis or description recorded.';
    const status = clear('experiment-status');
    for (const [label, statusClass] of [
        [`${progress.completed || 0} completed`, 'ok'],
        [`${progress.failed_attempts || 0} failed attempts`, progress.failed_attempts ? 'warn' : 'ok'],
        [`${progress.interrupted_attempts || 0} interrupted`, progress.interrupted_attempts ? 'warn' : 'ok'],
        [`${detail.summary_catalog?.revisions?.length || 0} immutable summaries`, 'ok'],
    ]) { const badge = document.createElement('span'); badge.className = `status-badge ${statusClass}`; badge.textContent = label; status.append(badge); }
    const revision = document.getElementById('summary-revision');
    revision.replaceChildren();
    for (const item of detail.summary_catalog?.revisions || []) revision.append(new Option(`Revision ${item.revision} · ${item.created_at || 'unknown time'}`, String(item.revision)));
    revision.value = String(detail.summary_catalog?.current_revision || '');
    revision.onchange = () => loadSummary(Number(revision.value));
}

function viewMetrics() { return state.summary?.analysis?.views?.[state.view]?.overall || {}; }

function renderSummary() {
    const analysis = state.summary.analysis || {};
    const views = analysis.views || {};
    const selector = document.getElementById('analysis-view');
    selector.replaceChildren(...Object.keys(views).map(name => new Option(name.replaceAll('_', ' '), name)));
    selector.value = state.view in views ? state.view : 'all_completed';
    state.view = selector.value;
    selector.onchange = () => { state.view = selector.value; renderSummary(); };
    document.getElementById('view-note').textContent = analysis.view_membership_note || 'Views are evidence filters, not separate experiment cohorts.';
    renderOverview(); renderConditions(); renderComparisons(); renderCalibration(); renderOperational(); renderReproducibility();
}

function renderOverview() {
    const metrics = viewMetrics();
    const target = clear('benchmark-overview');
    target.append(
        card('Eligible games', value(metrics.games, 0), `${metrics.seed_count || 0} seed clusters`),
        card('Village win rate', metricEstimate(metrics.village_win_rate), interval(metrics.village_win_rate)),
        card('Clean-game rate', metricEstimate(metrics.clean_game_rate), `${metrics.clean_game_rate?.numerator ?? 0}/${metrics.clean_game_rate?.denominator ?? 0} games`),
        card('Fallback-game rate', metricEstimate(metrics.fallback_game_rate), `${metrics.fallback_game_rate?.numerator ?? 0}/${metrics.fallback_game_rate?.denominator ?? 0} games`),
        card('Cost per game', money(metrics.cost?.cost_per_game_usd), metrics.cost?.cost_complete ? 'Complete' : 'Partial or unavailable'),
        card('Latency coverage', percent(metrics.latency?.coverage_fraction), `${metrics.latency?.calls_with_latency ?? 0}/${metrics.latency?.total_attempted_calls ?? 0} calls`),
    );
    const notice = document.getElementById('benchmark-integrity');
    const ineligible = state.summary.analysis?.analytically_ineligible || [];
    const strategic = (state.summary.analysis?.games || []).filter(game => game.analysis_eligibility !== 'eligible');
    notice.classList.toggle('hidden', !ineligible.length && !strategic.length);
    notice.textContent = (ineligible.length || strategic.length) ? `${ineligible.length} completed source(s) have missing or changed canonical evidence; ${strategic.length} verified completed game(s) are strategically limited or ineligible. They remain visible below with reasons and are excluded from clean-eligible comparisons.` : '';
    const selected = clear('selected-metrics');
    for (const id of ['retry_rate', 'repair_rate', 'vote_belief_alignment', 'harmful_revision', 'correct_belief_retention', 'probability_movement_toward_wolves', 'wolf_suspicion_awareness_error', 'brier_post_discussion']) {
        const metric = metrics[id];
        selected.append(card(metricNames[id], metricEstimate(metric, id.includes('error') || id.includes('brier') || id.includes('movement') ? value : percent), interval(metric, id.includes('error') || id.includes('brier') || id.includes('movement') ? value : percent)));
    }
}

function renderConditions() {
    const rows = clear('condition-rows');
    const conditions = state.summary.analysis?.views?.[state.view]?.per_condition || {};
    const scheduled = state.summary.analysis?.scheduled_trial_outcomes?.per_condition || {};
    for (const [condition, metrics] of Object.entries(conditions)) {
        const row = document.createElement('tr');
        cell(row, condition); cell(row, value(metrics.games, 0)); cell(row, value(metrics.seed_count, 0));
        cell(row, metricEstimate(scheduled[condition]?.scheduled_completion_rate)); cell(row, metricEstimate(scheduled[condition]?.final_failed_trial_rate));
        cell(row, metricEstimate(metrics.village_win_rate)); cell(row, metricEstimate(metrics.wolf_win_rate));
        cell(row, metricEstimate(metrics.clean_game_rate)); cell(row, metricEstimate(metrics.fallback_game_rate)); cell(row, money(metrics.cost?.cost_per_game_usd)); rows.append(row);
    }
    document.getElementById('condition-empty').classList.toggle('hidden', Object.values(conditions).some(metrics => metrics.games));
}

function renderComparisons() {
    const rows = clear('comparison-rows');
    for (const item of state.summary.analysis?.comparisons || []) {
        const row = document.createElement('tr');
        cell(row, item.comparison_id); cell(row, item.analysis_view); cell(row, item.metric_id);
        cell(row, `${item.condition_a} − ${item.condition_b}`); cell(row, percent(item.estimate));
        cell(row, interval(item)); cell(row, value(item.n_seeds, 0)); cell(row, item.status || 'unknown'); rows.append(row);
    }
}

function renderCalibration() {
    const rows = clear('calibration-rows');
    const ece = viewMetrics().ece_post_discussion;
    for (const bin of ece?.bins || []) {
        const row = document.createElement('tr');
        cell(row, bin.bin); cell(row, value(bin.prediction_count, 0)); cell(row, value(bin.game_count, 0)); cell(row, value(bin.seed_count, 0));
        cell(row, percent(bin.mean_confidence)); cell(row, percent(bin.empirical_frequency)); cell(row, percent(bin.absolute_gap)); rows.append(row);
    }
    document.getElementById('calibration-empty').classList.toggle('hidden', Boolean(ece?.prediction_count));
}

function renderOperational() {
    const operational = state.summary.analysis?.operational || {};
    const attempts = operational.attempts || {};
    const target = clear('operational-metrics');
    const scheduled = state.summary.analysis?.scheduled_trial_outcomes?.overall || {};
    target.append(
        card('Scheduled completion', metricEstimate(scheduled.scheduled_completion_rate), `${scheduled.completed || 0} / ${scheduled.scheduled || 0} trials`),
        card('Final failed trials', metricEstimate(scheduled.final_failed_trial_rate), `${scheduled.failed || 0} scheduled trials`),
        card('Interrupted or pending', metricEstimate(scheduled.final_interrupted_or_pending_rate), `${(scheduled.interrupted || 0) + (scheduled.pending || 0) + (scheduled.running || 0)} scheduled trials`),
        card('Operational attempts', value(attempts.total, 0), `${attempts.trial_failed || 0} failed · ${attempts.trial_interrupted || 0} interrupted`),
        card('Health checks', value(operational.cost?.health_checks, 0), `${money(operational.cost?.health_checks_usd)} · ${operational.cost?.health_cost_complete ? 'complete' : 'partial/unavailable'}`),
        card('Sources excluded', value(operational.cost?.sources_excluded_from_totals, 0), operational.cost?.complete ? 'Cost evidence complete' : 'Cost evidence partial'),
    );
    const rows = clear('trial-rows');
    const games = state.summary.analysis?.games || [];
    for (const game of games) {
        const row = document.createElement('tr');
        cell(row, game.trial_id); cell(row, game.attempt_id); cell(row, value(game.seed)); cell(row, game.condition_id); cell(row, game.winner || '—'); cell(row, game.clean ? 'Yes' : 'No'); cell(row, game.analysis_eligibility || 'unavailable'); cell(row, (game.analysis_exclusion_reasons || []).join(', ') || '—'); cell(row, game.usage_reliability || 'unavailable'); cell(row, 'verified');
        const forensic = document.createElement('td'); const link = document.createElement('a'); link.href = `/experiments/${encodeURIComponent(experimentId)}/games/${encodeURIComponent(game.game_id)}`; link.textContent = game.game_id; forensic.append(link); row.append(forensic); rows.append(row);
    }
    for (const source of state.summary.analysis?.analytically_ineligible || []) {
        const row = document.createElement('tr');
        cell(row, source.trial_id); cell(row, source.attempt_id); cell(row, value(source.seed)); cell(row, source.condition_id); cell(row, '—'); cell(row, '—'); cell(row, 'ineligible'); cell(row, source.reason || source.source_status); cell(row, 'unavailable'); cell(row, source.source_status); cell(row, 'Unavailable'); rows.append(row);
    }
}

function renderReproducibility() {
    const manifest = state.detail.manifest || {}, summary = state.summary || {};
    const values = {
        'Manifest hash': summary.manifest_content_sha256,
        'Execution contract': state.detail.manifest?.execution_contract ? 'Pinned in manifest' : 'Unavailable',
        'Analysis contract': summary.analysis_contract_sha256,
        'Analysis runtime': summary.analysis_runtime_hash,
        'Summary input': summary.summary_input_sha256,
        'Lifecycle records': summary.lifecycle?.lifecycle_record_count,
        'Journal snapshot': summary.lifecycle?.lifecycle_snapshot_sha256,
        'Prompt profile': manifest.execution_contract?.prompt_profile?.name,
        'Bootstrap': JSON.stringify(summary.analysis?.bootstrap || {}),
    };
    const list = clear('reproducibility-values');
    for (const [label, content] of Object.entries(values)) { const dt = document.createElement('dt'); dt.textContent = label; const dd = document.createElement('dd'); dd.textContent = value(content); list.append(dt, dd); }
    const exports = clear('export-links');
    for (const name of ['trials.csv', 'attempts.csv', 'metrics.csv', 'comparisons.csv', 'calibration.csv']) { const link = document.createElement('a'); link.href = `/api/experiments/${encodeURIComponent(experimentId)}/exports/${state.revision}/${name}`; link.textContent = `Download ${name}`; exports.append(link); }
}

async function loadSummary(revision) {
    try {
        state.summary = await fetchJson(`/api/experiments/${encodeURIComponent(experimentId)}/summaries/${revision}`);
        state.revision = revision; state.view = 'all_completed'; renderSummary();
    } catch (error) { const target = document.getElementById('experiment-error'); target.textContent = error.message; target.classList.remove('hidden'); }
}

async function init() {
    try {
        state.detail = await fetchJson(`/api/experiments/${encodeURIComponent(experimentId)}`);
        renderDetail();
        const revision = state.detail.summary_catalog?.current_revision;
        if (revision) await loadSummary(revision);
        else document.getElementById('experiment-error').textContent = 'No immutable summary has been generated for this experiment yet.';
    } catch (error) { const target = document.getElementById('experiment-error'); target.textContent = error.message; target.classList.remove('hidden'); }
}

init();
