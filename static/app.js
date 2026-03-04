/**
 * CV-модуль ParkCloud — Фронтенд-приложение
 *
 * Подключается к бэкенду через SSE для обновлений занятости в реальном времени
 * и отрисовывает интерактивную карту парковки + журнал событий.
 */

// ============================================================
// Состояние
// ============================================================
const state = {
    spaces: {},          // {id: {occupancy_pct, confidence, is_occupied}}
    filter: 'all',       // all | free | occupied
    eventSource: null,
    eventsLog: [],
    maxEvents: 100,
};

// ============================================================
// Отрисовка SVG-карты
// ============================================================

const MAP_SCALE = 10; // 1 метр = 10 SVG-единиц

function initMap(spacesData) {
    const layer = document.getElementById('spaces-layer');
    layer.innerHTML = '';

    spacesData.forEach(space => {
        const poly = space.polygon;
        const points = poly.map(p => `${p[0] * MAP_SCALE},${p[1] * MAP_SCALE}`).join(' ');

        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.classList.add('parking-space');
        g.id = `svg-space-${space.id}`;
        g.setAttribute('data-space-id', space.id);

        // Фоновый полигон
        const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        polygon.setAttribute('points', points);
        polygon.setAttribute('fill', '#00b894');
        polygon.setAttribute('fill-opacity', '0.3');
        polygon.setAttribute('stroke', '#00b894');
        polygon.setAttribute('stroke-width', '1.5');
        polygon.setAttribute('rx', '2');
        g.appendChild(polygon);

        // Подпись
        const cx = poly.reduce((s, p) => s + p[0], 0) / poly.length * MAP_SCALE;
        const cy = poly.reduce((s, p) => s + p[1], 0) / poly.length * MAP_SCALE;

        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', cx);
        text.setAttribute('y', cy + 1);
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('dominant-baseline', 'middle');
        text.setAttribute('fill', 'white');
        text.setAttribute('font-size', '9');
        text.setAttribute('font-family', 'monospace');
        text.setAttribute('font-weight', 'bold');
        text.textContent = space.id;
        g.appendChild(text);

        // Всплывающая подсказка при наведении
        g.addEventListener('mouseenter', () => {
            const s = state.spaces[space.id];
            if (s) {
                const status = s.is_occupied ? 'Занято' : 'Свободно';
                g.querySelector('polygon').setAttribute('stroke-width', '3');
            }
        });
        g.addEventListener('mouseleave', () => {
            g.querySelector('polygon').setAttribute('stroke-width', '1.5');
        });

        layer.appendChild(g);
    });

    // Иконки камер
    drawCameras();
}

function drawCameras() {
    const layer = document.getElementById('cameras-layer');
    layer.innerHTML = '';

    const cameras = [
        { id: 'cam_01', x: 5, y: 285, label: '📷 Cam 1' },
        { id: 'cam_02', x: 550, y: 285, label: '📷 Cam 2' },
    ];

    cameras.forEach(cam => {
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', cam.x);
        text.setAttribute('y', cam.y);
        text.setAttribute('fill', '#f39c12');
        text.setAttribute('font-size', '11');
        text.setAttribute('font-family', 'sans-serif');
        text.textContent = cam.label;
        layer.appendChild(text);
    });
}

function updateMapSpace(spaceId, occupancyPct, isOccupied) {
    const g = document.getElementById(`svg-space-${spaceId}`);
    if (!g) return;

    const polygon = g.querySelector('polygon');
    let color, opacity;

    if (isOccupied || occupancyPct >= 70) {
        color = '#e17055';
        opacity = 0.5;
    } else if (occupancyPct >= 30) {
        color = '#fdcb6e';
        opacity = 0.4;
    } else {
        color = '#00b894';
        opacity = 0.3;
    }

    polygon.setAttribute('fill', color);
    polygon.setAttribute('fill-opacity', opacity);
    polygon.setAttribute('stroke', color);
}

// ============================================================
// Список мест
// ============================================================

function renderSpacesList() {
    const container = document.getElementById('spaces-list');
    const sorted = Object.entries(state.spaces).sort((a, b) => a[0].localeCompare(b[0]));

    container.innerHTML = sorted
        .filter(([id, s]) => {
            if (state.filter === 'free') return !s.is_occupied;
            if (state.filter === 'occupied') return s.is_occupied;
            return true;
        })
        .map(([id, s]) => {
            const occ = s.occupancy_pct;
            const cls = s.is_occupied ? 'occupied' : occ >= 30 ? 'uncertain' : 'free';
            const barColor = s.is_occupied ? 'var(--color-occupied)' :
                             occ >= 30 ? 'var(--color-uncertain)' : 'var(--color-free)';
            return `
                <div class="space-item ${cls}" data-space-id="${id}">
                    <span class="space-id">${id}</span>
                    <div class="space-bar">
                        <div class="space-bar-fill" style="width:${occ}%;background:${barColor}"></div>
                    </div>
                    <span class="space-occupancy">${occ.toFixed(0)}%</span>
                </div>
            `;
        })
        .join('');
}

// ============================================================
// Статистика в шапке
// ============================================================

function updateStats(summary) {
    document.getElementById('total-spaces').textContent = summary.total || '—';
    document.getElementById('occupied-spaces').textContent = summary.occupied || '—';
    document.getElementById('free-spaces').textContent = summary.free || '—';
    document.getElementById('occupancy-rate').textContent =
        summary.occupancy_rate != null ? `${summary.occupancy_rate}%` : '—';
}

// ============================================================
// Журнал событий
// ============================================================

function addEvent(event) {
    if (event.event_type === 'full_snapshot') return; // пропускаем снимки в журнале

    state.eventsLog.unshift(event);
    if (state.eventsLog.length > state.maxEvents) {
        state.eventsLog.pop();
    }
    renderEventsLog();
}

function renderEventsLog() {
    const container = document.getElementById('events-log');
    container.innerHTML = state.eventsLog.slice(0, 30).map(ev => {
        const time = ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString('ru-RU') : '—';
        const isOccupied = ev.event_type === 'space_occupied';
        const typeLabel = isOccupied ? '🔴 ЗАНЯТО' : '🟢 СВОБОДНО';
        const typeCls = isOccupied ? 'occupied' : 'freed';
        return `
            <div class="event-item">
                <span class="event-time">${time}</span>
                <span class="event-type ${typeCls}">${typeLabel}</span>
                <span class="event-detail">${ev.space_id} (${ev.occupancy_pct?.toFixed(0) ?? 0}%)</span>
            </div>
        `;
    }).join('');
}

// ============================================================
// SSE-подключение
// ============================================================

function connectSSE() {
    const statusEl = document.getElementById('connection-status');

    if (state.eventSource) {
        state.eventSource.close();
    }

    state.eventSource = new EventSource('/api/stream');

    state.eventSource.onopen = () => {
        statusEl.textContent = '🟢 Подключено';
        statusEl.style.color = 'var(--color-free)';
    };

    state.eventSource.onerror = () => {
        statusEl.textContent = '🔴 Отключено';
        statusEl.style.color = 'var(--color-occupied)';
        // Автопереподключение обеспечивается EventSource
    };

    // Обработка событий-снимков
    state.eventSource.addEventListener('full_snapshot', (e) => {
        const data = JSON.parse(e.data);
        const meta = data.metadata;
        if (!meta) return;

        updateStats(meta);

        if (meta.spaces) {
            for (const [id, info] of Object.entries(meta.spaces)) {
                state.spaces[id] = info;
                updateMapSpace(id, info.occupancy_pct, info.is_occupied);
            }
            renderSpacesList();
        }
    });

    // Обработка событий смены состояния
    state.eventSource.addEventListener('space_occupied', (e) => {
        const data = JSON.parse(e.data);
        updateSpaceFromEvent(data);
        addEvent(data);
    });

    state.eventSource.addEventListener('space_freed', (e) => {
        const data = JSON.parse(e.data);
        updateSpaceFromEvent(data);
        addEvent(data);
    });

    state.eventSource.addEventListener('occupancy_update', (e) => {
        const data = JSON.parse(e.data);
        updateSpaceFromEvent(data);
    });
}

function updateSpaceFromEvent(data) {
    const id = data.space_id;
    if (!id || id === '__all__') return;

    const isOccupied = data.occupancy_pct >= 60;
    state.spaces[id] = {
        occupancy_pct: data.occupancy_pct,
        confidence: data.confidence,
        is_occupied: isOccupied,
    };
    updateMapSpace(id, data.occupancy_pct, isOccupied);
    renderSpacesList();
}

// ============================================================
// Начальная загрузка
// ============================================================

async function loadInitialData() {
    try {
        // Загрузка мест
        const spacesRes = await fetch('/api/spaces');
        const spacesData = await spacesRes.json();

        spacesData.spaces.forEach(s => {
            state.spaces[s.id] = {
                occupancy_pct: s.occupancy_pct,
                confidence: s.confidence,
                is_occupied: s.is_occupied,
            };
        });

        initMap(spacesData.spaces);
        renderSpacesList();

        // Загрузка статуса
        const statusRes = await fetch('/api/status');
        const statusData = await statusRes.json();
        updateStats(statusData);

        // Загрузка камер
        const camerasRes = await fetch('/api/cameras');
        const camerasData = await camerasRes.json();
        document.getElementById('cameras-count').textContent =
            `${camerasData.cameras.length} камер(ы)`;

        // Загрузка конфигурации
        const configRes = await fetch('/api/config');
        const configData = await configRes.json();
        document.getElementById('fusion-strategy').value = configData.fusion_strategy;

    } catch (err) {
        console.error('Failed to load initial data:', err);
    }
}

// ============================================================
// Обработчики событий
// ============================================================

document.addEventListener('DOMContentLoaded', async () => {
    await loadInitialData();
    connectSSE();

    // Кнопки фильтрации
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.filter = btn.dataset.filter;
            renderSpacesList();
        });
    });

    // Смена стратегии фузии
    document.getElementById('fusion-strategy').addEventListener('change', async (e) => {
        try {
            await fetch(`/api/config/fusion?strategy=${e.target.value}`, { method: 'POST' });
        } catch (err) {
            console.error('Failed to set fusion strategy:', err);
        }
    });
});
