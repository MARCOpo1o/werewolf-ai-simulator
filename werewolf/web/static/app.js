let gameState = null;
let phaseEvents = [];
let eventIndex = 0;
let isSteppingThroughEvents = false;
let autoPlayActive = false;
let autoPlayTimer = null;
let prefetchedData = null;
let prefetchPromise = null;
let modelCatalog = [];
const healthCache = new Map();

const PLAYER_EMOJIS = ['👤', '🧑', '👩', '🧔', '👨', '👵', '🧓', '👱', '🧑‍🦰', '👩‍🦳', '🧑‍🦱', '👨‍🦲', '🧑‍🦲', '👴', '👧'];
const AUTO_PLAY_DELAY_MS = 0;

async function fetchState() {
    const response = await fetch('/api/state');
    const data = await response.json();
    if (data.game) {
        gameState = data.game;
        renderGame();
    }
    return data;
}

async function newGame(payload) {
    const response = await fetch('/api/new', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (data.error) {
        showSetupErrors(data.errors || { request: { message: data.error } });
        return false;
    }
    gameState = data.game;
    phaseEvents = [];
    eventIndex = 0;
    isSteppingThroughEvents = false;
    prefetchedData = null;
    prefetchPromise = null;
    if (autoPlayActive) stopAutoPlay();
    renderGame();
    clearCenterDisplay();
    const btn = document.getElementById('advance-btn');
    btn.disabled = false;
    btn.textContent = 'Start Game';
    document.getElementById('autoplay-btn').disabled = false;
    return true;
}

async function fetchModels() {
    const response = await fetch('/api/models');
    const data = await response.json();
    modelCatalog = data.models || [];
    populateModelSelectors();
}

function populateModelSelectors() {
    const selectors = ['quick-model', 'wolf-model', 'villager-model', 'seer-model'];
    const options = modelCatalog.map(model =>
        `<option value="${model.alias}">${model.display_name}</option>`
    ).join('');
    selectors.forEach(id => {
        const select = document.getElementById(id);
        select.innerHTML = options;
    });
    const defaultAlias = modelCatalog.some(m => m.alias === 'fast') ? 'fast' : modelCatalog[0]?.alias;
    document.getElementById('quick-model').value = defaultAlias || '';
    document.getElementById('wolf-model').value = defaultAlias || '';
    document.getElementById('villager-model').value = defaultAlias || '';
    document.getElementById('seer-model').value = defaultAlias || '';
    renderModelDetails();
}

function selectedGameType() {
    return document.querySelector('input[name="game-type"]:checked')?.value || 'quick';
}

function selectedAliases() {
    if (selectedGameType() === 'quick') {
        return [document.getElementById('quick-model').value];
    }
    const aliases = [
        document.getElementById('wolf-model').value,
        document.getElementById('villager-model').value,
    ];
    if (parseInt(document.getElementById('input-seers').value, 10) > 0) {
        aliases.push(document.getElementById('seer-model').value);
    }
    return [...new Set(aliases.filter(Boolean))];
}

function getModel(alias) {
    return modelCatalog.find(model => model.alias === alias);
}

function modelSummary(model) {
    if (!model) return '';
    const ready = model.key_configured ? 'Key configured' : 'Missing API key';
    const experimental = model.experimental ? '<span class="tag experimental">Experimental</span>' : '';
    return `<div class="model-summary">
        <div><strong>${model.display_name}</strong> · ${model.provider} · ${model.speed_tier} speed · ${model.cost_tier} cost</div>
        <p>${model.description}</p>
        <div class="tag-row"><span class="tag ${model.key_configured ? 'ready' : 'missing'}">${ready}</span>${experimental}</div>
    </div>`;
}

function renderModelDetails() {
    const quick = getModel(document.getElementById('quick-model').value);
    document.getElementById('quick-model-detail').innerHTML = modelSummary(quick);
    const assignments = [
        ['Wolves', getModel(document.getElementById('wolf-model').value)],
        ['Village', getModel(document.getElementById('villager-model').value)],
        ['Seer', getModel(document.getElementById('seer-model').value)],
    ];
    document.getElementById('matchup-model-detail').innerHTML = assignments.map(
        ([role, model]) => `<div class="matchup-summary"><span>${role}</span>${modelSummary(model)}</div>`
    ).join('');
}

function updateSetupMode() {
    const matchup = selectedGameType() === 'matchup';
    document.getElementById('quick-model-section').classList.toggle('hidden', matchup);
    document.getElementById('matchup-model-section').classList.toggle('hidden', !matchup);
    renderModelDetails();
    renderCachedHealth();
}

function updateSeerControl() {
    const inactive = parseInt(document.getElementById('input-seers').value, 10) === 0;
    const seerSelect = document.getElementById('seer-model');
    seerSelect.disabled = inactive;
    if (inactive) seerSelect.value = document.getElementById('villager-model').value;
    document.getElementById('seer-inactive-note').classList.toggle('hidden', !inactive);
    renderModelDetails();
    renderCachedHealth();
}

function optionalNumber(id, integer = false) {
    const raw = document.getElementById(id).value.trim();
    if (raw === '') return null;
    const value = Number(raw);
    if (!Number.isFinite(value) || (integer && !Number.isInteger(value))) {
        return raw;
    }
    return value;
}

function generationPayload() {
    return {
        temperature: optionalNumber('input-temperature'),
        top_p: optionalNumber('input-top-p'),
        max_output_tokens: optionalNumber('input-max-tokens', true),
        provider_seed: optionalNumber('input-provider-seed', true),
        structured_output: document.getElementById('input-structured').checked,
    };
}

function customPayload() {
    return {
        generation_config: generationPayload(),
        reasoning_override: document.getElementById('input-reasoning').value || null,
    };
}

function canonicalize(value) {
    if (Array.isArray(value)) return value.map(canonicalize);
    if (value && typeof value === 'object') {
        return Object.keys(value).sort().reduce((out, key) => {
            out[key] = canonicalize(value[key]);
            return out;
        }, {});
    }
    return value;
}

async function healthFingerprint(alias) {
    const canonical = JSON.stringify(canonicalize({ alias, ...customPayload() }));
    const bytes = new TextEncoder().encode(canonical);
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    return Array.from(new Uint8Array(digest)).map(byte => byte.toString(16).padStart(2, '0')).join('');
}

async function checkSelectedModels() {
    const button = document.getElementById('health-check-btn');
    button.disabled = true;
    button.textContent = 'Checking…';
    clearSetupErrors();
    try {
        for (const alias of selectedAliases()) {
            const key = await healthFingerprint(alias);
            if (!healthCache.has(key)) {
                const response = await fetch(`/api/models/${encodeURIComponent(alias)}/health-check`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(customPayload()),
                });
                healthCache.set(key, await response.json());
            }
        }
        await renderCachedHealth();
    } catch (error) {
        showSetupErrors({ health: { message: error.message } });
    } finally {
        button.disabled = false;
        button.textContent = 'Check selected models';
    }
}

async function renderCachedHealth() {
    const container = document.getElementById('health-status');
    const entries = [];
    for (const alias of selectedAliases()) {
        const key = await healthFingerprint(alias);
        const result = healthCache.get(key);
        const model = getModel(alias);
        const status = result?.status || 'not checked';
        entries.push(`<span class="health-chip health-${status.replace(' ', '-')}">${model?.display_name || alias}: ${status}</span>`);
    }
    container.innerHTML = entries.join('');
}

function showSetupErrors(errors) {
    const container = document.getElementById('setup-errors');
    container.innerHTML = Object.entries(errors).map(([field, error]) =>
        `<div><strong>${field}:</strong> ${error.message || error}</div>`
    ).join('');
    container.classList.remove('hidden');
}

function clearSetupErrors() {
    const container = document.getElementById('setup-errors');
    container.innerHTML = '';
    container.classList.add('hidden');
}

function applyPhaseResult(data, fromAutoPlay = false) {
    const btn = document.getElementById('advance-btn');
    gameState = data.game;
    renderGame();

    if (data.result.done) {
        btn.textContent = 'Game Over';
        btn.disabled = true;
        showWinner();
        if (autoPlayActive) stopAutoPlay();
        document.getElementById('autoplay-btn').disabled = true;
        return;
    }

    phaseEvents = filterDisplayableEvents(data.result.phase_events || []);
    eventIndex = 0;

    if (phaseEvents.length > 0) {
        isSteppingThroughEvents = true;
        showCurrentEvent();
        updateButtonText();
        btn.disabled = false;
        if (!fromAutoPlay && !autoPlayActive) prefetchNextPhase();
    } else {
        isSteppingThroughEvents = false;
        clearCenterDisplay();
        btn.textContent = 'Next Phase';
        btn.disabled = false;
        if (!fromAutoPlay && !autoPlayActive) prefetchNextPhase();
    }
}

function prefetchNextPhase() {
    if (prefetchPromise) return;
    prefetchPromise = fetch('/api/advance', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            prefetchedData = data;
            prefetchPromise = null;
        })
        .catch(() => { prefetchPromise = null; });
}

async function advancePhase(fromAutoPlay = false) {
    const btn = document.getElementById('advance-btn');

    if (isSteppingThroughEvents) {
        eventIndex++;
        if (eventIndex < phaseEvents.length) {
            showCurrentEvent();
            updateButtonText();
        } else {
            isSteppingThroughEvents = false;
            clearCenterDisplay();
            btn.textContent = 'Next Phase';
            btn.disabled = false;
            if (!autoPlayActive) prefetchNextPhase();
        }
        return;
    }

    if (!fromAutoPlay) {
        btn.disabled = true;
        btn.textContent = 'Loading...';
    }

    try {
        let data;
        if (prefetchedData) {
            data = prefetchedData;
            prefetchedData = null;
        } else {
            const response = await fetch('/api/advance', { method: 'POST' });
            data = await response.json();
        }
        if (data.error) {
            alert('Error: ' + data.error);
            btn.disabled = false;
            btn.textContent = 'Next Phase';
            return;
        }
        applyPhaseResult(data, fromAutoPlay);
    } catch (err) {
        alert('Error: ' + err.message);
        btn.disabled = false;
        btn.textContent = 'Next Phase';
    }
}

function stopAutoPlay() {
    autoPlayActive = false;
    if (autoPlayTimer) {
        clearTimeout(autoPlayTimer);
        autoPlayTimer = null;
    }
    const abtn = document.getElementById('autoplay-btn');
    abtn.textContent = 'Auto Play';
    abtn.classList.remove('auto-playing');
    if (!gameState || !gameState.winner) {
        const btn = document.getElementById('advance-btn');
        btn.disabled = false;
        if (isSteppingThroughEvents) updateButtonText();
        else btn.textContent = 'Next Phase';
    }
}

function toggleAutoPlay() {
    if (!gameState || gameState.winner) return;
    if (autoPlayActive) {
        stopAutoPlay();
        return;
    }
    autoPlayActive = true;
    document.getElementById('autoplay-btn').textContent = 'Pause';
    document.getElementById('autoplay-btn').classList.add('auto-playing');
    document.getElementById('advance-btn').disabled = true;
    autoPlayTick();
}

function autoPlayTick() {
    if (!autoPlayActive) return;

    if (isSteppingThroughEvents && eventIndex < phaseEvents.length) {
        eventIndex++;
        showCurrentEvent();
        updateButtonText();
        autoPlayTimer = setTimeout(autoPlayTick, AUTO_PLAY_DELAY_MS);
        return;
    }

    if (isSteppingThroughEvents && eventIndex >= phaseEvents.length) {
        isSteppingThroughEvents = false;
        clearCenterDisplay();
    }

    (async () => {
        let data;
        if (prefetchedData) {
            data = prefetchedData;
            prefetchedData = null;
        } else {
            try {
                const response = await fetch('/api/advance', { method: 'POST' });
                data = await response.json();
            } catch (err) {
                alert('Error: ' + err.message);
                stopAutoPlay();
                return;
            }
        }
        if (data.error) {
            alert('Error: ' + data.error);
            stopAutoPlay();
            return;
        }
        applyPhaseResult(data, true);
        if (gameState.winner) return;
        autoPlayTimer = setTimeout(autoPlayTick, AUTO_PLAY_DELAY_MS);
    })();
}

function filterDisplayableEvents(events) {
    return events.filter(e => 
        e.type === 'message' || 
        e.type === 'thought' || 
        e.type === 'vote' ||
        e.type === 'death_announcement' ||
        e.type === 'elimination' ||
        e.type === 'divine_result' ||
        e.type === 'kill' ||
        e.type === 'runoff_announcement' ||
        e.type === 'no_elimination'
    );
}

function updateButtonText() {
    const btn = document.getElementById('advance-btn');
    const remaining = phaseEvents.length - eventIndex - 1;
    if (remaining > 0) {
        btn.textContent = `Next (${remaining} more)`;
    } else {
        btn.textContent = 'End Phase';
    }
}

function showCurrentEvent() {
    const event = phaseEvents[eventIndex];
    if (!event) return;

    const display = document.getElementById('center-display');
    const content = display.querySelector('.display-content');
    
    display.classList.remove('hidden', 'speech', 'thought', 'action', 'wolf', 'seer', 'village');
    
    highlightPlayer(event.speaker_id);
    
    if (event.type === 'message') {
        const player = getPlayer(event.speaker_id);
        const roleClass = getRoleClass(player);
        display.classList.add('speech', roleClass);
        
        const channelLabel = event.channel === 'werewolf' ? '(whisper to wolves)' : '';
        content.innerHTML = `
            <div class="display-header">
                <span class="speaker-icon">${PLAYER_EMOJIS[event.speaker_id % PLAYER_EMOJIS.length]}</span>
                <span class="speaker-name ${roleClass}">Player ${event.speaker_id}</span>
                <span class="speaker-role">${player?.role || ''} ${channelLabel}</span>
            </div>
            <div class="display-text">"${event.payload.text}"</div>
            <div class="display-hint">💬 Speaking</div>
        `;
    } else if (event.type === 'thought') {
        const player = getPlayer(event.speaker_id);
        const roleClass = getRoleClass(player);
        display.classList.add('thought', roleClass);
        
        content.innerHTML = `
            <div class="display-header">
                <span class="speaker-icon">${PLAYER_EMOJIS[event.speaker_id % PLAYER_EMOJIS.length]}</span>
                <span class="speaker-name ${roleClass}">Player ${event.speaker_id}</span>
                <span class="speaker-role">${player?.role || ''}</span>
            </div>
            <div class="display-text">${event.payload.thought}</div>
            <div class="display-hint">💭 Thinking (hidden from others)</div>
        `;
    } else if (event.type === 'vote') {
        const voter = getPlayer(event.speaker_id);
        const target = getPlayer(event.payload.target_id);
        display.classList.add('action');
        
        content.innerHTML = `
            <div class="action-icon">🗳️</div>
            <div class="display-text">
                <span class="${getRoleClass(voter)}">Player ${event.speaker_id}</span> 
                votes to eliminate 
                <span class="${getRoleClass(target)}">Player ${event.payload.target_id}</span>
            </div>
        `;
    } else if (event.type === 'divine_result') {
        const isWolf = event.payload.is_werewolf;
        display.classList.add('action', 'seer');
        
        content.innerHTML = `
            <div class="action-icon">🔮</div>
            <div class="display-text">
                <span class="seer">Seer (P${event.speaker_id})</span> investigates 
                <span class="${isWolf ? 'wolf' : 'village'}">Player ${event.payload.target_id}</span>
            </div>
            <div class="result ${isWolf ? 'wolf' : 'village'}">
                ${isWolf ? '🐺 WEREWOLF!' : '✓ NOT A WOLF'}
            </div>
        `;
    } else if (event.type === 'death_announcement' || event.type === 'elimination') {
        const victimId = event.payload.victim_id ?? event.payload.eliminated_id;
        const victim = getPlayer(victimId);
        const cause = event.payload.cause;
        const isWolfKill = cause === 'wolf_kill';
        display.classList.add('action', 'death');
        
        content.innerHTML = `
            <div class="action-icon">${isWolfKill ? '🐺💀' : '⚰️'}</div>
            <div class="display-text">
                <span class="${getRoleClass(victim)}">Player ${victimId}</span> 
                was ${isWolfKill ? 'killed by the wolves!' : 'voted out!'}
            </div>
            <div class="result ${getRoleClass(victim)}">
                Role: ${victim?.role?.toUpperCase() || 'UNKNOWN'}
            </div>
        `;
    } else if (event.type === 'kill') {
        display.classList.add('action', 'wolf');
        content.innerHTML = `
            <div class="action-icon">🐺🎯</div>
            <div class="display-text">
                Wolves have chosen their target: 
                <span class="victim">Player ${event.payload.victim_id}</span>
            </div>
        `;
    } else if (event.type === 'runoff_announcement') {
        const candidates = event.payload.candidates.map(c => `Player ${c}`).join(' vs ');
        display.classList.add('action');
        content.innerHTML = `
            <div class="action-icon">⚖️</div>
            <div class="display-text">
                Vote tied! Runoff: ${candidates}
            </div>
            <div class="display-hint">Only these candidates can be voted for</div>
        `;
    } else if (event.type === 'no_elimination') {
        display.classList.add('action');
        content.innerHTML = `
            <div class="action-icon">🤝</div>
            <div class="display-text">
                Runoff tied — no one is eliminated today!
            </div>
            <div class="display-hint">The village could not reach a consensus</div>
        `;
    }

    addEventToLog(event);
}

function highlightPlayer(playerId) {
    document.querySelectorAll('.player-card').forEach(card => {
        card.classList.remove('active', 'speaking');
    });
    if (playerId !== null && playerId !== undefined) {
        const card = document.querySelector(`.player-card[data-player-id="${playerId}"]`);
        if (card) {
            card.classList.add('active', 'speaking');
        }
    }
}

function clearCenterDisplay() {
    const display = document.getElementById('center-display');
    display.classList.add('hidden');
    highlightPlayer(null);
}

function getPlayer(id) {
    return gameState?.players?.find(p => p.id === id);
}

function getRoleClass(player) {
    if (!player) return '';
    if (player.role === 'werewolf') return 'wolf';
    if (player.role === 'seer') return 'seer';
    return 'village';
}

function renderGame() {
    if (!gameState) return;

    document.getElementById('round-info').textContent = `Round: ${gameState.round}`;
    document.getElementById('phase-info').textContent = `Phase: ${formatPhase(gameState.phase)}`;
    document.getElementById('team-counts').textContent =
        `Wolves: ${gameState.alive_wolves} | Villagers: ${gameState.alive_villagers}`;
    const assignments = gameState.model_assignment || {};
    const activeAliases = [...new Set(Object.values(assignments)
        .filter(item => item.active !== false)
        .map(item => item.alias || item.requested_model)
        .filter(Boolean))];
    document.getElementById('model-info').textContent = activeAliases.length
        ? `Models: ${activeAliases.join(' vs ')}` : 'Model: -';

    renderPlayers();
    renderWinner();
}

function formatPhase(phase) {
    const phases = {
        'setup': 'Setup',
        'night_wolf_chat': 'Night - Wolf Chat',
        'night_wolf_kill': 'Night - Wolf Kill',
        'night_seer': 'Night - Seer',
        'day_announce': 'Day - Announcement',
        'day_discuss': 'Day - Discussion',
        'day_vote': 'Day - Vote'
    };
    return phases[phase] || phase;
}

function renderPlayers() {
    const circle = document.getElementById('player-circle');
    circle.innerHTML = '';

    const players = gameState.players;
    const n = players.length;
    const rect = circle.getBoundingClientRect();
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    const radius = Math.min(centerX, centerY) * 0.78;

    players.forEach((player, i) => {
        const angle = (2 * Math.PI * i / n) - Math.PI / 2;
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);

        const card = document.createElement('div');
        card.className = 'player-card';
        card.dataset.playerId = player.id;
        if (!player.alive) card.classList.add('dead');

        const roleClass = getRoleClass(player);
        card.classList.add(roleClass);

        card.style.left = `${x}px`;
        card.style.top = `${y}px`;

        card.innerHTML = `
            <div class="avatar">${PLAYER_EMOJIS[player.id % PLAYER_EMOJIS.length]}</div>
            <div class="player-name">Player ${player.id}</div>
            <div class="player-role">${player.role}</div>
        `;

        card.onclick = () => showMemoryModal(player.id);
        circle.appendChild(card);
    });
}

function addEventToLog(event) {
    const list = document.getElementById('event-list');
    const item = document.createElement('div');
    item.className = 'event-item ' + event.type;
    item.innerHTML = formatEventForLog(event);
    list.appendChild(item);
    list.scrollTop = list.scrollHeight;
}

function formatEventForLog(event) {
    const pid = event.speaker_id !== null ? `P${event.speaker_id}` : '';

    switch (event.type) {
        case 'phase_change':
            return `<strong>— ${formatPhase(event.payload.new_phase)} —</strong>`;
        case 'message':
            const channel = event.channel === 'werewolf' ? ' (wolf)' : '';
            return `💬 ${pid}${channel}: "${truncate(event.payload.text, 50)}"`;
        case 'thought':
            return `💭 ${pid}: "${truncate(event.payload.thought, 50)}"`;
        case 'vote':
            return `🗳️ ${pid} → P${event.payload.target_id}`;
        case 'elimination':
            return `⚰️ <strong>P${event.payload.eliminated_id}</strong> eliminated (${event.payload.eliminated_role})`;
        case 'death_announcement':
            const cause = event.payload.cause === 'wolf_kill' ? '🐺 killed' : '🗳️ voted out';
            return `💀 P${event.payload.victim_id} ${cause}`;
        case 'divine_result':
            const result = event.payload.is_werewolf ? '🐺 WOLF!' : '✓ clear';
            return `🔮 ${pid} → P${event.payload.target_id}: ${result}`;
        case 'game_status':
            return `📊 Wolves: ${event.payload.alive_wolves}, Village: ${event.payload.alive_villagers}`;
        case 'win':
            return `🏆 <strong>${event.payload.winner.toUpperCase()} WINS!</strong>`;
        case 'kill':
            return `🐺 Wolves target P${event.payload.victim_id}`;
        case 'runoff_announcement':
            return `⚖️ Vote tied! Runoff: ${event.payload.candidates.map(c => `P${c}`).join(' vs ')}`;
        case 'no_elimination':
            return `🤝 Runoff tied — no elimination today`;
        default:
            return JSON.stringify(event.payload);
    }
}

function truncate(text, maxLen) {
    if (!text) return '';
    if (text.length <= maxLen) return text;
    return text.substring(0, maxLen - 3) + '...';
}

function showWinner() {
    const display = document.getElementById('winner-display');
    if (gameState.winner) {
        display.classList.remove('hidden', 'wolf-win', 'village-win');
        display.classList.add(gameState.winner === 'wolf' ? 'wolf-win' : 'village-win');
        display.textContent = `${gameState.winner.toUpperCase()} WINS!`;
        clearCenterDisplay();
    }
}

function renderWinner() {
    const display = document.getElementById('winner-display');
    if (gameState.winner) {
        showWinner();
    } else {
        display.classList.add('hidden');
    }
}

function showMemoryModal(playerId) {
    const modal = document.getElementById('memory-modal');
    const playerIdSpan = document.getElementById('memory-player-id');
    const content = document.getElementById('memory-content');

    const player = getPlayer(playerId);
    playerIdSpan.textContent = `${playerId} (${player?.role || 'unknown'})`;

    const playerEvents = gameState.events.filter(e =>
        e.speaker_id === playerId ||
        (e.payload && (e.payload.victim_id === playerId || e.payload.target_id === playerId || e.payload.eliminated_id === playerId))
    );

    content.innerHTML = playerEvents.map(e => {
        return `<div class="event-item ${e.type}">${formatEventForLog(e)}</div>`;
    }).join('') || '<p>No events for this player yet.</p>';

    modal.classList.remove('hidden');
}

function hideMemoryModal() {
    document.getElementById('memory-modal').classList.add('hidden');
}

function showNewGameModal() {
    clearSetupErrors();
    document.getElementById('new-game-modal').classList.remove('hidden');
    renderCachedHealth();
}

function hideNewGameModal() {
    document.getElementById('new-game-modal').classList.add('hidden');
}

document.getElementById('new-game-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const nPlayers = parseInt(document.getElementById('input-players').value);
    const nWolves = parseInt(document.getElementById('input-wolves').value);
    const nSeers = parseInt(document.getElementById('input-seers').value);
    const seed = parseInt(document.getElementById('input-seed').value);
    const payload = {
        n_players: nPlayers,
        n_wolves: nWolves,
        n_seers: nSeers,
        seed,
        discussion_cycles: parseInt(document.getElementById('input-cycles').value, 10),
        belief_snapshots: document.getElementById('input-snapshots').checked,
        ...customPayload(),
    };
    if (selectedGameType() === 'quick') {
        payload.model = document.getElementById('quick-model').value;
    } else {
        payload.role_models = {
            werewolf: document.getElementById('wolf-model').value,
            villager: document.getElementById('villager-model').value,
            seer: document.getElementById('seer-model').value,
        };
    }
    const started = await newGame(payload);
    if (started) hideNewGameModal();
});

document.querySelectorAll('input[name="game-type"]').forEach(input =>
    input.addEventListener('change', updateSetupMode));
document.querySelectorAll('.model-select').forEach(select =>
    select.addEventListener('change', () => {
        if (select.id === 'villager-model') updateSeerControl();
        else renderModelDetails();
        renderCachedHealth();
    }));
document.getElementById('input-seers').addEventListener('change', updateSeerControl);
document.getElementById('health-check-btn').addEventListener('click', checkSelectedModels);
const healthRelevantControls = [
    'input-temperature',
    'input-top-p',
    'input-max-tokens',
    'input-provider-seed',
    'input-structured',
    'input-reasoning',
];
healthRelevantControls.forEach(id => {
    const element = document.getElementById(id);
    element.addEventListener('input', renderCachedHealth);
    element.addEventListener('change', renderCachedHealth);
});

document.addEventListener('keydown', (e) => {
    if (e.code !== 'Space' && e.code !== 'Enter') return;
    if (document.querySelector('.modal:not(.hidden)')) return;
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) return;
    if (!gameState || gameState.winner) return;
    if (e.code === 'Space') e.preventDefault();
    advancePhase();
});

document.addEventListener('DOMContentLoaded', () => Promise.all([fetchState(), fetchModels()]));
