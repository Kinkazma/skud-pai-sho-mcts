// Selection belongs to a displayed campaign, not to the shared select element.
const dashboardLaneSelections = new Map();
function syncDashboardLane(select, scope, entries, fallback) {
  const previous = select.dataset.laneScope;
  if (previous) dashboardLaneSelections.set(previous, select.value);
  let selected = dashboardLaneSelections.get(scope) || fallback;
  if (!entries.some(([value]) => value === selected)) selected = fallback;
  const unchanged = select.options.length === entries.length && entries.every(([value,label],i) => select.options[i].value === value && select.options[i].textContent === label);
  if (!unchanged) select.replaceChildren(...entries.map(([value,label]) => new Option(label,value)));
  select.value = selected;
  select.disabled = false;
  select.dataset.laneScope = scope;
  return selected;
}
function gen5MatchRows(history, lane) {
  // Old single-reference receipts remain interpretable as Gen3.1.
  const rows = history.match_bins || (history.bins || []).map(row => ({...row,lane:'Historical',generation:'Gen3.1'}));
  return rows.filter(row => lane.startsWith('Gen3.') ? row.lane === 'Historical' && row.generation === lane : row.lane === lane);
}
if (typeof module !== 'undefined') module.exports = {syncDashboardLane,gen5MatchRows};
