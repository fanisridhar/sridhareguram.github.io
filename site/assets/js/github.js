// Live repository list from the public GitHub REST API (no token needed).
// Rendered with textContent only, so repo metadata can never inject markup.
(function () {
  var cfg = window.PORTFOLIO || {};
  var grid = document.getElementById("repo-grid");
  if (!grid || !cfg.github) return;

  var statusEl = document.getElementById("repo-status");
  var searchEl = document.getElementById("repo-search");
  var sortEl = document.getElementById("repo-sort");
  var CACHE_KEY = "repos:" + cfg.github;
  var TTL_MS = 10 * 60 * 1000;
  var repos = [];

  function readCache() {
    try {
      var raw = sessionStorage.getItem(CACHE_KEY);
      if (!raw) return null;
      var parsed = JSON.parse(raw);
      return Date.now() - parsed.at < TTL_MS ? parsed.data : null;
    } catch (e) {
      return null;
    }
  }

  function writeCache(data) {
    try {
      sessionStorage.setItem(CACHE_KEY, JSON.stringify({ at: Date.now(), data: data }));
    } catch (e) {
      /* cache is an optimisation only */
    }
  }

  function setStatus(text) {
    statusEl.textContent = text;
    statusEl.hidden = !text;
  }

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text) node.textContent = text;
    return node;
  }

  function render() {
    var q = searchEl.value.trim().toLowerCase();
    var sort = sortEl.value;
    var list = repos.filter(function (r) {
      if (!q) return true;
      return (
        r.name.toLowerCase().indexOf(q) !== -1 ||
        (r.description || "").toLowerCase().indexOf(q) !== -1 ||
        (r.language || "").toLowerCase().indexOf(q) !== -1
      );
    });

    list.sort(function (a, b) {
      if (sort === "stars") return b.stargazers_count - a.stargazers_count;
      if (sort === "name") return a.name.localeCompare(b.name);
      return new Date(b.pushed_at) - new Date(a.pushed_at);
    });

    grid.textContent = "";
    list.forEach(function (r) {
      var a = el("a", "repo");
      a.href = r.html_url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      a.appendChild(el("b", "", r.name));
      a.appendChild(el("p", "", r.description || "No description yet."));
      var meta = el("div", "meta");
      if (r.language) meta.appendChild(el("span", "", r.language));
      if (r.stargazers_count) meta.appendChild(el("span", "", "★ " + r.stargazers_count));
      meta.appendChild(
        el("span", "", "Updated " + new Date(r.pushed_at).toLocaleDateString(undefined, { year: "numeric", month: "short" }))
      );
      a.appendChild(meta);
      grid.appendChild(a);
    });

    setStatus(list.length ? "" : q ? "No repositories match that search." : "No public repositories to show yet.");
  }

  function showSkeletons() {
    grid.textContent = "";
    for (var i = 0; i < 6; i++) grid.appendChild(el("div", "skeleton"));
  }

  function load() {
    var cached = readCache();
    if (cached) {
      repos = cached;
      render();
      return;
    }
    showSkeletons();
    var url = "https://api.github.com/users/" + encodeURIComponent(cfg.github) + "/repos?per_page=100&sort=pushed";
    var ctrl = "AbortController" in window ? new AbortController() : null;
    var timer = ctrl ? setTimeout(function () { ctrl.abort(); }, 10000) : null;

    fetch(url, { headers: { Accept: "application/vnd.github+json" }, signal: ctrl ? ctrl.signal : undefined })
      .then(function (res) {
        if (!res.ok) throw new Error("GitHub responded " + res.status);
        return res.json();
      })
      .then(function (data) {
        repos = data.filter(function (r) {
          return !r.fork && !r.archived;
        });
        writeCache(repos);
        render();
      })
      .catch(function () {
        grid.textContent = "";
        setStatus("Could not load repositories right now. You can browse them directly on GitHub.");
      })
      .then(function () {
        if (timer) clearTimeout(timer);
      });
  }

  searchEl.addEventListener("input", render);
  sortEl.addEventListener("change", render);
  load();
})();
