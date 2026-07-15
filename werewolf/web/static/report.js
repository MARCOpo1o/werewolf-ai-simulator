const reportState = { report: null, includePrivate: false };
const gameId = document.getElementById('report-content').dataset.gameId;
const svgNS = 'http://www.w3.org/2000/svg';

const fmt = {
    number(value, digits = 3) {
        return value === null || value === undefined ? 'Unavailable' : Number(value).toFixed(digits);
    },
    percent(value) {
        return value === null || value === undefined ? '—' : `${(Number(value) * 100).toFixed(1)}%`;
    },
    money(value) {
        return value === null || value === undefined ? 'Unavailable' : `$${Number(value).toFixed(6)}`;
    },
    value(value) {
        if (value === null || value === undefined || value === '') return 'Unavailable';
        if (typeof value === 'boolean') return value ? 'Yes' : 'No';
        if (typeof value === 'object') return JSON.stringify(value);
        return String(value);
    },
};

function el(tag, text = null, className = '') {
    const node = document.createElement(tag);
    if (text !== null) node.textContent = text;
    if (className) node.className = className;
    return node;
}

function statusBadge(value) {
    return el('span', fmt.value(value), `badge ${value || ''}`);
}

function metric(label, value) {
    const card = el('div', null, 'metric-card');
    card.append(el('span', label), el('strong', fmt.value(value)));
    return card;
}

function renderKeyValues(targetId, values) {
    const target = document.getElementById(targetId);
    target.replaceChildren();
    for (const [label, value] of Object.entries(values)) {
        target.append(el('dt', label), el('dd', fmt.value(value)));
    }
}

function playerLabel(id) {
    if (id === null || id === undefined) return 'Moderator';
    const player = (reportState.report.players || []).find(item => item.id === Number(id));
    const role = player?.role ? ` · ${player.role}` : '';
    return `P${id}${role}`;
}

function renderStatuses(overview) {
    const target = document.getElementById('report-statuses');
    target.replaceChildren(
        statusBadge(overview.display_status),
        statusBadge(overview.integrity_status),
        statusBadge(overview.analysis_eligibility),
        statusBadge(overview.usage_reliability),
    );
}

function renderOverview(report) {
    const overview = report.overview || {};
    renderStatuses(overview);
    document.getElementById('report-title').textContent = overview.game_id || gameId;
    const warning = document.getElementById('analysis-warning');
    const eligible = overview.analysis_eligibility === 'eligible';
    warning.classList.toggle('hidden', eligible);
    warning.textContent = eligible ? '' : `Strategic analysis is ${overview.analysis_eligibility || 'unavailable'}: ${(overview.analysis_exclusion_reasons || []).join(', ') || 'no reason recorded'}.`;

    const usage = report.usage || overview.usage || {};
    const cards = document.getElementById('overview-cards');
    cards.replaceChildren(
        metric('Winner', overview.winner ? `${overview.winner} team` : 'Not finished'),
        metric('Rounds', overview.rounds ?? '—'),
        metric('LLM attempts', usage.attempts ?? 0),
        metric('Retries', usage.retries ?? 0),
        metric('Fallbacks', usage.fallbacks ?? 0),
        metric('Known cost', fmt.money(usage.known_cost_usd)),
    );
    renderKeyValues('overview-config', {
        Seed: overview.seed,
        Players: overview.n_players,
        Wolves: overview.n_wolves,
        Seers: overview.n_seers,
        'Discussion cycles': overview.discussion_cycles,
        'Belief snapshots': overview.belief_snapshots,
        'Cost completeness': usage.cost_completeness,
        'Cost sources': (usage.cost_sources || []).join(', ') || 'Unavailable',
    });
    const models = document.getElementById('overview-models');
    models.replaceChildren();
    for (const [role, info] of Object.entries(overview.requested_models || {})) {
        const block = el('div', null, 'model-block');
        block.append(
            el('strong', role),
            el('code', info?.requested_model || info?.model || info?.alias || 'Unavailable'),
            el('small', `Alias: ${info?.alias || 'unavailable'}`, 'muted'),
        );
        models.append(block);
    }
}

function optionValues(events, key, roleMap = null) {
    const values = new Set();
    for (const event of events) {
        const value = roleMap ? roleMap.get(event.speaker_id) : event[key];
        if (value !== null && value !== undefined && value !== '') values.add(String(value));
    }
    return [...values].sort();
}

function makeFilter(id, label, values) {
    const wrapper = el('label', label);
    const select = el('select');
    select.id = id;
    select.append(new Option('All', 'all'));
    for (const value of values) select.append(new Option(value, value));
    select.addEventListener('change', renderTimelineList);
    wrapper.append(select);
    return wrapper;
}

function eventDescription(event) {
    const p = event.payload || {};
    switch (event.type) {
        case 'message': return p.text || '(empty message)';
        case 'thought': return p.thought || '(empty thought)';
        case 'vote': return `${playerLabel(p.voter_id)} voted for P${p.target_id}`;
        case 'kill': return `Night target P${p.victim_id}; votes ${JSON.stringify(p.votes || {})}`;
        case 'divine_result': return `Checked P${p.target_id}: ${p.is_werewolf ? 'werewolf' : 'not werewolf'}`;
        case 'death_announcement': return `P${p.victim_id} died (${p.cause || 'unknown cause'})${p.victim_role ? ` · revealed ${p.victim_role}` : ''}`;
        case 'elimination': return `P${p.eliminated_id} eliminated · ${p.eliminated_role || 'role unavailable'}`;
        case 'belief_snapshot': return `${p.checkpoint || 'snapshot'} · ${p.valid ? 'valid' : `invalid (${p.invalid_reason || 'unknown'})`}`;
        case 'phase_change': return `Phase changed to ${p.new_phase}`;
        case 'game_status': return `${p.alive_wolves} wolves and ${p.alive_villagers} villagers alive`;
        case 'win': return `${p.winner} team won`;
        default: return JSON.stringify(p);
    }
}

function renderTimeline(report) {
    const events = report.timeline || [];
    const roleMap = new Map((report.players || []).map(player => [player.id, player.role]));
    const filters = document.getElementById('timeline-filters');
    filters.replaceChildren(
        makeFilter('filter-phase', 'Phase', optionValues(events, 'phase')),
        makeFilter('filter-player', 'Player', optionValues(events, 'speaker_id').map(value => `P${value}`)),
        makeFilter('filter-role', 'Role', optionValues(events, null, roleMap)),
        makeFilter('filter-type', 'Event type', optionValues(events, 'type')),
        makeFilter('filter-channel', 'Channel', optionValues(events, 'channel')),
    );
    renderTimelineList();
}

function selectedFilter(id) {
    return document.getElementById(id)?.value || 'all';
}

function renderTimelineList() {
    const report = reportState.report;
    const roleMap = new Map((report.players || []).map(player => [player.id, player.role]));
    const events = (report.timeline || []).filter(event => {
        const player = event.speaker_id === null || event.speaker_id === undefined ? null : `P${event.speaker_id}`;
        const role = roleMap.get(event.speaker_id);
        return (selectedFilter('filter-phase') === 'all' || event.phase === selectedFilter('filter-phase'))
            && (selectedFilter('filter-player') === 'all' || player === selectedFilter('filter-player'))
            && (selectedFilter('filter-role') === 'all' || role === selectedFilter('filter-role'))
            && (selectedFilter('filter-type') === 'all' || event.type === selectedFilter('filter-type'))
            && (selectedFilter('filter-channel') === 'all' || event.channel === selectedFilter('filter-channel'));
    });
    document.getElementById('timeline-count').textContent = `${events.length} events`;
    const target = document.getElementById('timeline-list');
    target.replaceChildren();
    for (const event of events) {
        const card = el('article', null, `timeline-event ${event.channel !== 'public' ? 'private-event' : ''}`);
        const meta = el('div', null, 'event-meta');
        meta.append(
            el('div', `Round ${event.round ?? '—'}`),
            el('div', event.phase || 'unknown phase'),
            el('div', event.discussion_cycle ? `Cycle ${event.discussion_cycle}` : event.channel || ''),
            el('div', playerLabel(event.speaker_id)),
        );
        const body = el('div', null, 'event-body');
        body.append(el('strong', `${event.type || 'unknown'} · ${event.event_id || 'no event ID'}`), el('p', eventDescription(event)));
        body.append(el('div', `Line ${event.source_line ?? '—'} · call ${event.source_call_id || 'unavailable'} · link ${event.link_quality || 'unavailable'}`, 'event-provenance'));
        card.append(meta, body);
        target.append(card);
    }
    if (!events.length) target.append(el('div', 'No events match these filters.', 'empty-state'));
}

function svgNode(tag, attributes = {}) {
    const node = document.createElementNS(svgNS, tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    return node;
}

function renderBeliefChart(trajectories, observer) {
    const container = document.getElementById('belief-chart');
    container.replaceChildren();
    const points = trajectories.filter(item => item.observer_id === observer);
    if (!points.length) { container.append(el('div', 'No valid probabilities for this observer.', 'empty-state')); return; }
    const checkpointOrder = { pre_discussion: 0, post_discussion: 1 };
    const keys = [...new Set(points.map(item => `${item.round}:${item.checkpoint}`))].sort((a, b) => {
        const [ar, ac] = a.split(':'); const [br, bc] = b.split(':');
        return Number(ar) - Number(br) || (checkpointOrder[ac] ?? 9) - (checkpointOrder[bc] ?? 9);
    });
    const width = 900, height = 330, left = 55, right = 170, top = 25, bottom = 55;
    const plotWidth = width - left - right, plotHeight = height - top - bottom;
    const x = index => left + (keys.length === 1 ? plotWidth / 2 : index * plotWidth / (keys.length - 1));
    const y = value => top + (1 - value) * plotHeight;
    const svg = svgNode('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': `Belief trajectories for P${observer}` });
    for (let tick = 0; tick <= 4; tick++) {
        const value = tick / 4, yy = y(value);
        svg.append(svgNode('line', { x1: left, x2: width - right, y1: yy, y2: yy, class: 'chart-grid' }));
        const label = svgNode('text', { x: left - 9, y: yy + 4, 'text-anchor': 'end', class: 'chart-label' }); label.textContent = `${Math.round(value * 100)}%`; svg.append(label);
    }
    keys.forEach((key, index) => { const [round, checkpoint] = key.split(':'); const label = svgNode('text', { x: x(index), y: height - 27, 'text-anchor': 'middle', class: 'chart-label' }); label.textContent = `R${round} ${checkpoint === 'pre_discussion' ? 'pre' : 'post'}`; svg.append(label); });
    const colors = ['#60a5fa', '#f87171', '#34d399', '#c084fc', '#fbbf24', '#22d3ee', '#fb7185'];
    const targets = [...new Set(points.map(item => item.target_id))].sort((a, b) => a - b);
    targets.forEach((target, targetIndex) => {
        const targetPoints = points.filter(item => item.target_id === target);
        const lookup = new Map(targetPoints.map(item => [`${item.round}:${item.checkpoint}`, item]));
        const actualWolf = targetPoints.some(item => item.actual_is_wolf === true);
        const color = colors[targetIndex % colors.length];
        let segment = [];
        const flushSegment = () => {
            if (segment.length > 1) {
                svg.append(svgNode('polyline', { points: segment.join(' '), fill: 'none', stroke: color, 'stroke-width': actualWolf ? 4 : 2, 'stroke-dasharray': actualWolf ? '8 4' : '' }));
            }
            segment = [];
        };
        keys.forEach((key, index) => {
            const point = lookup.get(key);
            if (!point) { flushSegment(); return; }
            const xx = x(index), yy = y(point.probability);
            segment.push(`${xx},${yy}`);
            svg.append(svgNode('circle', {
                cx: xx, cy: yy, r: point.snapshot_valid ? 4 : 6,
                class: `chart-point ${point.snapshot_valid ? 'valid' : 'invalid'}`,
                fill: point.snapshot_valid ? color : 'none',
                stroke: point.snapshot_valid ? color : '#fb7185',
            }));
        });
        flushSegment();
        const legend = svgNode('text', { x: width - right + 18, y: top + 18 + targetIndex * 22, class: 'chart-legend' }); legend.textContent = `P${target}${actualWolf ? ' · actual wolf' : ''}`; svg.append(legend);
    });
    container.append(svg);
}

function renderBeliefs(report) {
    const beliefs = report.beliefs || {};
    const unavailable = document.getElementById('belief-unavailable');
    const content = document.getElementById('belief-content');
    const observerSelect = document.getElementById('belief-observer');
    if (!beliefs.available) {
        unavailable.textContent = beliefs.reason === 'private_data_not_requested' ? 'Reveal private forensic data to inspect internal beliefs.' : `Belief analysis unavailable: ${beliefs.reason || 'no snapshots'}.`;
        unavailable.classList.remove('hidden'); content.classList.add('hidden'); observerSelect.classList.add('hidden'); return;
    }
    unavailable.classList.add('hidden'); content.classList.remove('hidden'); observerSelect.classList.remove('hidden');
    const observers = [...new Set((beliefs.trajectories || []).map(item => item.observer_id))].sort((a, b) => a - b);
    observerSelect.replaceChildren(...observers.map(id => new Option(playerLabel(id), String(id))));
    const renderObserver = () => {
        const observer = Number(observerSelect.value);
        renderBeliefChart(beliefs.trajectories || [], observer);
        const changes = document.getElementById('belief-change-rows'); changes.replaceChildren();
        for (const item of (beliefs.changes || []).filter(change => change.observer_id === observer)) {
            const row = el('tr');
            [item.round, playerLabel(item.observer_id), `P${item.target_id}`, fmt.percent(item.pre_probability), fmt.percent(item.post_probability), fmt.number(item.delta), fmt.number(item.movement_toward_truth), item.evidence_quality, item.most_influential_recent_speaker === null || item.most_influential_recent_speaker === undefined ? 'Not recorded' : playerLabel(item.most_influential_recent_speaker)].forEach(value => row.append(el('td', fmt.value(value))));
            changes.append(row);
        }
        const accuracy = document.getElementById('belief-accuracy-rows'); accuracy.replaceChildren();
        for (const item of (beliefs.trajectories || []).filter(point => point.observer_id === observer)) {
            const row = el('tr');
            [item.checkpoint, playerLabel(item.observer_id), `P${item.target_id}`, fmt.percent(item.probability), item.snapshot_valid ? 'Valid' : 'Partial / invalid', item.actual_is_wolf === null ? 'Unknown' : item.actual_is_wolf ? 'Wolf' : 'Not wolf', fmt.number(item.squared_error), item.brier_schema_applicable ? fmt.number(item.brier_score_contribution) : 'Not applicable'].forEach(value => row.append(el('td', fmt.value(value))));
            accuracy.append(row);
        }
    };
    observerSelect.onchange = renderObserver;
    if (observers.length) renderObserver();
}

function decisionText(event) {
    return eventDescription(event);
}

function renderDecisions(report) {
    const decisions = report.decisions || {};
    const groups = decisions.attempt_groups || [];
    document.getElementById('decision-summary').replaceChildren(
        metric('Decision groups', groups.length), metric('Retry groups', (decisions.retry_groups || []).length),
        metric('Fallback groups', (decisions.fallback_groups || []).length), metric('Malformed groups', (decisions.malformed_groups || []).length),
    );
    const eventRows = document.getElementById('decision-event-rows'); eventRows.replaceChildren();
    for (const event of decisions.decision_events || []) {
        const row = el('tr');
        [event.round, event.phase, event.type, playerLabel(event.speaker_id), decisionText(event), event.link_quality || 'unavailable'].forEach(value => row.append(el('td', fmt.value(value))));
        eventRows.append(row);
    }
    const attemptPanel = document.getElementById('attempt-panel');
    attemptPanel.classList.toggle('hidden', !groups.length);
    const attemptRows = document.getElementById('attempt-group-rows'); attemptRows.replaceChildren();
    for (const group of groups) {
        const row = el('tr');
        [group.required_action, playerLabel(group.player_id), group.requested_model, group.attempt_count, group.retry_count, (group.parse_methods || []).join(', ') || 'none', group.fallback_used ? 'Yes' : 'No', group.final_error_category].forEach(value => row.append(el('td', fmt.value(value))));
        attemptRows.append(row);
    }
}

function renderManipulation(report) {
    const signals = report.manipulation_signals || {};
    const unavailable = document.getElementById('manipulation-unavailable');
    const content = document.getElementById('manipulation-content');
    if (!signals.available) {
        unavailable.textContent = signals.reason === 'private_data_not_requested' ? 'Reveal private forensic data to inspect ground-truth manipulation and resistance signals.' : `Signals unavailable: ${signals.reason || 'insufficient evidence'}.`;
        unavailable.classList.remove('hidden'); content.classList.add('hidden'); return;
    }
    unavailable.classList.add('hidden'); content.classList.remove('hidden');
    const rows = document.getElementById('suspicion-change-rows'); rows.replaceChildren();
    for (const item of signals.wolf_suspicion_changes || []) {
        const row = el('tr');
        [item.round, playerLabel(item.observer_id), playerLabel(item.target_id), fmt.percent(item.pre_probability), fmt.percent(item.post_probability), fmt.number(item.delta), fmt.number(item.wolf_suspicion_recovery), item.most_influential_recent_speaker === null || item.most_influential_recent_speaker === undefined ? 'Not recorded' : playerLabel(item.most_influential_recent_speaker)].forEach(value => row.append(el('td', fmt.value(value))));
        rows.append(row);
    }
    const cards = document.getElementById('revision-cards'); cards.replaceChildren();
    const revisions = [...(signals.harmful_revisions || []), ...(signals.resistance_signals || [])];
    for (const item of revisions) {
        const card = el('div', null, 'episode');
        card.append(el('h4', `${item.revision.replaceAll('_', ' ')} · ${playerLabel(item.observer_id)}`), el('p', `Round ${item.round} · pre top ${(item.pre_top_suspects || []).map(id => `P${id}`).join(', ')} → post top ${(item.post_top_suspects || []).map(id => `P${id}`).join(', ')}`), el('p', `Vote ${item.vote_target === null || item.vote_target === undefined ? 'unavailable' : `P${item.vote_target}`} · belief aligned ${fmt.value(item.vote_matches_post_belief)}`));
        cards.append(card);
    }
    if (!revisions.length) cards.append(el('div', 'No revision signals were available.', 'empty-state'));
}

function breakdownPanel(title, data) {
    const panel = el('article', null, 'panel'); panel.append(el('h3', title));
    const table = el('table'); const body = el('tbody');
    for (const [name, values] of Object.entries(data || {})) {
        const row = el('tr'); row.append(el('td', name), el('td', `${values.attempts ?? values.calls ?? 0} attempts`), el('td', fmt.money(values.known_cost_usd ?? values.cost_usd)));
        body.append(row);
    }
    table.append(body); panel.append(table); return panel;
}

function renderUsage(report) {
    const usage = report.usage || {};
    document.getElementById('usage-cards').replaceChildren(
        metric('Attempts', usage.attempts ?? 0), metric('Decision groups', usage.decision_groups ?? 0),
        metric('Retries', usage.retries ?? 0), metric('Fallbacks', usage.fallbacks ?? 0),
        metric('Known cost', fmt.money(usage.known_cost_usd)), metric('Cost completeness', usage.cost_completeness),
        metric('Known-cost calls', usage.calls_with_known_cost ?? 0), metric('Unknown-cost calls', usage.calls_without_known_cost ?? 0),
    );
    const target = document.getElementById('usage-breakdowns'); target.replaceChildren();
    if (usage.by_player) target.append(breakdownPanel('By player', usage.by_player));
    if (usage.by_requested_model) target.append(breakdownPanel('By requested model', usage.by_requested_model));
    target.append(breakdownPanel('By action', usage.by_required_action || {}), breakdownPanel('By phase', usage.by_phase || {}));
}

function renderReproducibility(report) {
    const repro = report.reproducibility || {}, runtime = repro.runtime || {};
    renderKeyValues('repro-values', {
        'Code commit': repro.code_commit,
        'Prompt version': repro.prompt_version,
        'Log schema': repro.log_schema_version,
        'Event schema': repro.event_schema_version,
        'Belief schema': repro.belief_schema_version,
        'Validity policy': repro.validity_policy_version,
        'Report schema': repro.report_schema_version,
        'Eligibility policy': repro.analysis_eligibility_policy_version,
        'Python': runtime.python,
        'xAI SDK': runtime.xai_sdk,
        'LiteLLM': runtime.litellm,
        'Flask': runtime.flask,
        'Requirements SHA-256': runtime.requirements_sha256,
    });
    const source = report.source || {};
    renderKeyValues('source-values', {
        'Raw log': source.log_name,
        'Created at': source.created_at,
        'Timestamp source': source.created_at_source,
        'Size': source.size_bytes === undefined ? null : `${source.size_bytes} bytes`,
        'JSONL SHA-256': source.sha256,
        'Record counts': source.record_counts,
    });
    const warnings = document.getElementById('source-warnings'); warnings.replaceChildren();
    if ((source.warnings || []).length) {
        const list = el('ul', null, 'warning-list');
        for (const warning of source.warnings) list.append(el('li', `${warning.code}${warning.source_line ? ` · line ${warning.source_line}` : ''}: ${warning.message}`));
        warnings.append(list);
    }
}

function renderReport(report) {
    reportState.report = report;
    document.getElementById('raw-log-link').href = report.links?.raw || `/api/games/${encodeURIComponent(gameId)}/raw`;
    renderOverview(report); renderTimeline(report); renderBeliefs(report); renderDecisions(report); renderManipulation(report); renderUsage(report); renderReproducibility(report);
    document.getElementById('report-loading').classList.add('hidden');
    document.getElementById('report-content').classList.remove('hidden');
    document.getElementById('report-nav').classList.remove('hidden');
}

async function loadReport(includePrivate = false) {
    if (!gameId) return;
    document.getElementById('report-error').classList.add('hidden');
    document.getElementById('report-loading').classList.remove('hidden');
    try {
        const response = await fetch(`/api/games/${encodeURIComponent(gameId)}/report?include_private=${includePrivate}`);
        if (!response.ok) throw new Error(`Report request failed (${response.status})`);
        reportState.includePrivate = includePrivate;
        const report = await response.json();
        const note = document.getElementById('privacy-note');
        note.classList.toggle('private', includePrivate);
        note.textContent = includePrivate ? 'Private forensic view: roles, thoughts, belief instrumentation, and ground-truth signals are visible. This is spoiler protection, not authentication.' : 'Spoiler-safe view: private roles, thoughts, beliefs, and manipulation signals are excluded by the server.';
        document.getElementById('private-toggle').textContent = includePrivate ? 'Return to spoiler-safe view' : 'Reveal private forensic data';
        renderReport(report);
    } catch (error) {
        const target = document.getElementById('report-error'); target.textContent = error.message; target.classList.remove('hidden');
        document.getElementById('report-loading').classList.add('hidden');
    }
}

document.getElementById('private-toggle').addEventListener('click', () => loadReport(!reportState.includePrivate));
if (gameId) loadReport(false); else { document.getElementById('report-loading').classList.add('hidden'); document.getElementById('report-error').textContent = 'Game not found.'; document.getElementById('report-error').classList.remove('hidden'); }
