// Small UI helpers: health badge, prediction status rendering hooks.

document.addEventListener('DOMContentLoaded', function () {
  const badge = document.getElementById('health-badge');
  if (badge) {
    fetch('/health')
      .then(function (r) { return r.json(); })
      .then(function (d) {
        badge.textContent = d.status === 'ok' ? 'System OK' : 'System issue';
        badge.className = 'badge rounded-pill ' + (d.status === 'ok' ? 'bg-success' : 'bg-danger');
      })
      .catch(function () { badge.textContent = 'Health check unavailable'; badge.className = 'badge rounded-pill bg-warning'; });
  }
});
