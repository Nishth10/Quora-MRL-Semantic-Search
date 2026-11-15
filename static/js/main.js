document.addEventListener('DOMContentLoaded', () => {
    // State
    let currentDim = 64;
    let lastQuery = '';

    // DOM Elements
    const dimsSelector = document.getElementById('dimsSelector');
    const searchInput = document.getElementById('searchInput');
    const searchBtn = document.getElementById('searchBtn');
    const exampleBtns = document.querySelectorAll('.example-btn');
    const metricsBar = document.getElementById('metricsBar');
    const latencySeconds = document.getElementById('latencySeconds');
    const latencyMs = document.getElementById('latencyMs');
    const metricDim = document.getElementById('metricDim');
    const metricDimSub = document.getElementById('metricDimSub');
    const metricBytes = document.getElementById('metricBytes');
    const metricSavings = document.getElementById('metricSavings');
    const breakdownEnc = document.getElementById('breakdownEnc');
    const breakdownMrl = document.getElementById('breakdownMrl');
    const breakdownSearch = document.getElementById('breakdownSearch');
    const resultsSection = document.getElementById('resultsSection');
    const resultsList = document.getElementById('resultsList');
    const resultsCount = document.getElementById('resultsCount');
    const loadingState = document.getElementById('loadingState');
    const compareAllBtn = document.getElementById('compareAllBtn');
    const comparisonContainer = document.getElementById('comparisonContainer');
    const comparisonTableBody = document.getElementById('comparisonTableBody');
    const closeCompareBtn = document.getElementById('closeCompareBtn');

    // 1. MRL Dimension Selector Click
    dimsSelector.addEventListener('click', (e) => {
        const chip = e.target.closest('.dim-chip');
        if (!chip) return;

        document.querySelectorAll('.dim-chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        currentDim = parseInt(chip.getAttribute('data-dim'), 10);

        // If there's already an active search, re-run with new dimension
        if (lastQuery.trim().length > 0) {
            performSearch(lastQuery, currentDim);
        }
    });

    // 2. Search Triggers
    searchBtn.addEventListener('click', () => {
        const query = searchInput.value.trim();
        if (query) {
            performSearch(query, currentDim);
        }
    });

    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            const query = searchInput.value.trim();
            if (query) {
                performSearch(query, currentDim);
            }
        }
    });

    // 3. Example Chips Click
    exampleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            searchInput.value = btn.textContent;
            performSearch(btn.textContent, currentDim);
        });
    });

    // 4. Perform Search via Flask API
    async function performSearch(query, dim) {
        lastQuery = query;
        loadingState.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        comparisonContainer.classList.add('hidden');
        compareAllBtn.disabled = true;

        try {
            const res = await fetch('/api/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query, dim: dim, top_k: 10 })
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.error || 'Search failed');
            }

            const data = await res.json();
            renderMetrics(data);
            renderResults(data);
            compareAllBtn.disabled = false;
        } catch (err) {
            alert('Error during search: ' + err.message);
        } finally {
            loadingState.classList.add('hidden');
        }
    }

    // 5. Render Latency & Metrics Showcase
    function renderMetrics(data) {
        metricsBar.classList.remove('hidden');

        // Latency in seconds and ms (Key Sanchit requirement!)
        latencySeconds.textContent = `${data.latency_seconds.toFixed(4)} s`;
        latencyMs.textContent = `${data.latency_ms.toFixed(1)} ms total time`;

        // Dimension & Compression
        metricDim.textContent = `${data.dim}d`;
        metricDimSub.textContent = `of 384d (${data.compression.compression_factor} smaller)`;
        metricBytes.textContent = `${data.compression.bytes_per_vector} B/vec`;
        metricSavings.textContent = `${data.compression.savings_percent}% storage saved`;

        // Detailed breakdown
        if (data.breakdown_ms) {
            breakdownEnc.textContent = `Enc: ${data.breakdown_ms.query_encoding.toFixed(1)}ms`;
            breakdownMrl.textContent = `Slice: ${data.breakdown_ms.mrl_slicing.toFixed(1)}ms`;
            breakdownSearch.textContent = `Sim: ${data.breakdown_ms.vector_search.toFixed(1)}ms`;
        }
    }

    // 6. Render Results List
    function renderResults(data) {
        resultsSection.classList.remove('hidden');
        resultsCount.textContent = `Top ${data.results.length} matches (${data.dim}d representation)`;
        resultsList.innerHTML = '';

        data.results.forEach(item => {
            const card = document.createElement('div');
            card.className = 'result-card';

            const scoreColor = item.percentage > 75 ? '#10b981' : item.percentage > 50 ? '#6366f1' : '#f59e0b';

            card.innerHTML = `
                <div class="result-rank">#${item.rank}</div>
                <div class="result-body">
                    <div class="result-question">${escapeHtml(item.question)}</div>
                    <div class="result-meta">
                        <span class="result-tag">${escapeHtml(item.category)}</span>
                        <div class="sim-score-wrapper">
                            <span>Cosine Score: <strong>${item.score.toFixed(4)}</strong></span>
                            <div class="sim-bar">
                                <div class="sim-fill" style="width: ${item.percentage}%; background: ${scoreColor};"></div>
                            </div>
                            <span>(${item.percentage}%)</span>
                        </div>
                    </div>
                </div>
            `;
            resultsList.appendChild(card);
        });
    }

    // 7. Multi-Dimension Comparison
    compareAllBtn.addEventListener('click', async () => {
        if (!lastQuery) return;

        compareAllBtn.disabled = true;
        compareAllBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Comparing across dimensions...';

        try {
            const res = await fetch('/api/compare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: lastQuery })
            });

            if (!res.ok) throw new Error('Comparison failed');
            const data = await res.json();
            renderComparison(data);
        } catch (err) {
            alert('Comparison error: ' + err.message);
        } finally {
            compareAllBtn.disabled = false;
            compareAllBtn.innerHTML = '<i class="fa-solid fa-chart-simple"></i> Compare Latency & Quality Across All Dimensions';
        }
    });

    function renderComparison(data) {
        comparisonContainer.classList.remove('hidden');
        comparisonTableBody.innerHTML = '';

        data.comparisons.forEach(row => {
            const tr = document.createElement('tr');
            const badgeClass = row.top1_matches_384d ? 'exact' : 'drift';
            const badgeText = row.top1_matches_384d ? 'Identical to 384d' : 'Rank Drift';

            tr.innerHTML = `
                <td><strong>${row.dim}d</strong></td>
                <td>${row.latency_ms.toFixed(2)} ms</td>
                <td>${row.bytes_per_vec} B</td>
                <td><span style="color: #166534; font-weight:600;">-${row.savings_pct}%</span></td>
                <td class="text-col" title="${escapeHtml(row.top1_match)}">${escapeHtml(row.top1_match)}</td>
                <td><strong>${row.top1_score.toFixed(4)}</strong></td>
                <td><span class="status-badge ${badgeClass}">${badgeText}</span></td>
            `;
            comparisonTableBody.appendChild(tr);
        });

        // Scroll to comparison
        comparisonContainer.scrollIntoView({ behavior: 'smooth' });
    }

    closeCompareBtn.addEventListener('click', () => {
        comparisonContainer.classList.add('hidden');
    });

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text || '';
        return div.innerHTML;
    }
});
