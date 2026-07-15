const historyState = { games: [], nextCursor: null, loading: false };

function makeCell(text, className = '') {
    const cell = document.createElement('td');
    cell.textContent = text;
    if (className) cell.className = className;
    return cell;
}

function badge(text, type) {
    const element = document.createElement('span');
    element.className = `history-badge ${type || ''}`;
    element.textContent = text;
    return element;
}

function formatCost(game) {
    if (game.known_cost_usd === null || game.known_cost_usd === undefined) return 'Unavailable';
    return `$${Number(game.known_cost_usd).toFixed(4)}`;
}

function filteredGames() {
    const query = document.getElementById('history-search').value.trim().toLowerCase();
    const status = document.getElementById('history-status').value;
    return historyState.games.filter(game => {
        if (status !== 'all' && game.display_status !== status) return false;
        if (!query) return true;
        const haystack = [game.game_id, game.seed, ...(game.models || [])].join(' ').toLowerCase();
        return haystack.includes(query);
    });
}

function renderHistory() {
    const rows = document.getElementById('history-rows');
    const table = document.getElementById('history-table-wrap');
    const empty = document.getElementById('history-empty');
    rows.replaceChildren();
    const games = filteredGames();
    table.classList.toggle('hidden', games.length === 0);
    empty.classList.toggle('hidden', games.length !== 0 || historyState.loading);

    for (const game of games) {
        const row = document.createElement('tr');
        const gameCell = document.createElement('td');
        const link = document.createElement('a');
        link.href = `/games/${encodeURIComponent(game.game_id)}`;
        link.textContent = game.game_id;
        const date = document.createElement('small');
        date.textContent = game.created_at ? new Date(game.created_at).toLocaleString() : 'Unknown time';
        gameCell.append(link, date);
        row.append(gameCell);

        const statusCell = document.createElement('td');
        statusCell.append(badge(game.display_status, game.display_status));
        row.append(statusCell);
        row.append(makeCell(game.winner ? `${game.winner} won` : '—'));
        row.append(makeCell((game.models || []).join(', ') || 'Unknown'));
        row.append(makeCell(game.seed ?? '—', 'numeric'));
        row.append(makeCell(game.rounds ?? '—', 'numeric'));
        row.append(makeCell(formatCost(game), 'numeric'));
        const integrityCell = document.createElement('td');
        integrityCell.append(badge(game.integrity_status || 'unknown', game.integrity_status));
        row.append(integrityCell);
        rows.append(row);
    }
    document.getElementById('history-more').classList.toggle('hidden', !historyState.nextCursor);
}

async function loadHistory({ append = false } = {}) {
    if (historyState.loading) return;
    historyState.loading = true;
    document.getElementById('history-loading').classList.remove('hidden');
    document.getElementById('history-error').classList.add('hidden');
    try {
        const params = new URLSearchParams({ limit: '50' });
        if (append && historyState.nextCursor) params.set('cursor', historyState.nextCursor);
        const response = await fetch(`/api/games?${params}`);
        if (!response.ok) throw new Error(`History request failed (${response.status})`);
        const body = await response.json();
        historyState.games = append ? historyState.games.concat(body.games || []) : (body.games || []);
        historyState.nextCursor = body.next_cursor || null;
    } catch (error) {
        const target = document.getElementById('history-error');
        target.textContent = error.message;
        target.classList.remove('hidden');
    } finally {
        historyState.loading = false;
        document.getElementById('history-loading').classList.add('hidden');
        renderHistory();
    }
}

document.getElementById('history-search').addEventListener('input', renderHistory);
document.getElementById('history-status').addEventListener('change', renderHistory);
document.getElementById('history-refresh').addEventListener('click', () => loadHistory());
document.getElementById('history-more').addEventListener('click', () => loadHistory({ append: true }));
loadHistory();
