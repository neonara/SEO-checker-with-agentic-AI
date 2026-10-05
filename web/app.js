(() => {
  "use strict";

  const POLL_MS = 2000;
  const RING_LENGTH = 2 * Math.PI * 52;
  const $ = (id) => document.getElementById(id);

  const MODE_HINTS = {
    quick:
      "Fastest. Uses the small model for everything, so findings are less detailed.",
    auto: "Uses the strong model and switches to the small one if it runs out of quota.",
    deep: "Strong model only. Best quality, but the scan fails if its quota is used up.",
  };
  // What to suggest next, by the server's error_code.
  const FAILURE_HELP = {
    blocked:
      "Scanning again rarely helps. If this is your site, allow automated visits in its firewall or bot protection, then scan again.",
    too_few_checks:
      "This can happen when a site is slow or partly unreachable, or the models are briefly unavailable. Try again in a few minutes.",
    quota:
      "Try a Quick scan, or come back later. This scan did not count toward your hourly limit.",
    timeout:
      "Try a Quick scan, or come back in a few minutes. This scan did not count toward your hourly limit.",
    internal:
      "Try again in a few minutes. This scan did not count toward your hourly limit.",
  };
  const SEVERITY = {
    critical: {
      label: "Critical",
      plural: "Critical",
      icon: "M12 2 21 7v10l-9 5-9-5V7zM12 7v6M12 16.5v.5",
    },
    warning: {
      label: "Warning",
      plural: "Warnings",
      icon: "M12 3 22 20H2zM12 10v4.5M12 17.5v.5",
    },
    good: {
      label: "Passed",
      plural: "Passed",
      icon: "M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zM7.5 12.5l3 3 6-6.5",
    },
  };
  const SEVERITY_ORDER = ["critical", "warning", "good"];

  let pollTimer = null;
  let tickTimer = null;
  let lastSubmit = null; // {url, mode, competitor_url} of the scan on screen
  let filter = "all";
  let currentReport = null;
  let currentJobId = null;

  // --- DOM helpers -----------------------------------------------------
  // Everything in a report is text written by a model about someone else's
  // website. It only ever reaches the page as text nodes, never as HTML.
  function el(tag, props, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props || {})) {
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else if (value !== null && value !== undefined && value !== false)
        node.setAttribute(key, value);
    }
    for (const child of children) if (child) node.append(child);
    return node;
  }

  function icon(path) {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("aria-hidden", "true");
    const p = document.createElementNS(ns, "path");
    p.setAttribute("d", path);
    p.setAttribute("fill", "none");
    p.setAttribute("stroke", "currentColor");
    p.setAttribute("stroke-width", "2");
    p.setAttribute("stroke-linecap", "round");
    p.setAttribute("stroke-linejoin", "round");
    svg.append(p);
    return svg;
  }

  function show(view) {
    for (const name of ["home", "running", "failed", "result"])
      $("view-" + name).hidden = name !== view;
    if (view === "home") renderRecent();
    window.scrollTo(0, 0);
  }

  function gradeVar(score) {
    const letter =
      score >= 90
        ? "a"
        : score >= 80
          ? "b"
          : score >= 70
            ? "c"
            : score >= 60
              ? "d"
              : "f";
    return `var(--grade-${letter})`;
  }

  const fmtScore = (n) =>
    Number.isFinite(n) ? String(Math.round(n * 10) / 10) : "?";
  const fmtDate = (value) => {
    const d = new Date(value);
    return Number.isNaN(d.getTime())
      ? String(value)
      : d.toLocaleDateString(undefined, {
          day: "numeric",
          month: "short",
          year: "numeric",
        });
  };
  const fmtDuration = (seconds) => {
    const s = Math.max(0, Math.round(seconds));
    return s < 60
      ? `${s} s`
      : `${Math.floor(s / 60)} min ${String(s % 60).padStart(2, "0")} s`;
  };

  // --- Recent scans ----------------------------------------------------
  // The scans started from this browser, kept in this browser. The server
  // has no list to offer: a report is only reachable by its unguessable id,
  // so one visitor never sees another's scans.
  const RECENT_KEY = "recent-scans";
  const RECENT_MAX = 8;
  function loadRecent() {
    try {
      const list = JSON.parse(localStorage.getItem(RECENT_KEY) || "[]");
      return Array.isArray(list)
        ? list.filter((s) => s && /^[0-9a-f]{32}$/.test(s.id))
        : [];
    } catch (e) {
      return [];
    }
  }
  function saveRecent(list) {
    try {
      localStorage.setItem(RECENT_KEY, JSON.stringify(list.slice(0, RECENT_MAX)));
    } catch (e) {
      /* storage unavailable: the list just isn't kept */
    }
  }
  function rememberScan(entry) {
    saveRecent([entry, ...loadRecent().filter((s) => s.id !== entry.id)]);
  }
  // Fills in the outcome of a scan this browser started; a report opened
  // from someone else's link is not added.
  function updateScan(id, fields) {
    saveRecent(
      loadRecent().map((s) => (s.id === id ? { ...s, ...fields } : s)),
    );
  }
  function forgetScan(id) {
    saveRecent(loadRecent().filter((s) => s.id !== id));
  }
  function renderRecent() {
    const list = loadRecent();
    $("recent").hidden = !list.length;
    $("recent-list").replaceChildren(
      ...list.map((s) => {
        const done = Number.isFinite(s.score);
        return el(
          "li",
          null,
          el(
            "a",
            { href: "#job=" + s.id },
            el("span", {
              class: "recent-grade",
              style: done ? `--grade:${gradeVar(s.score)}` : null,
              text: done ? String(s.grade || "?") : "…",
            }),
            el(
              "span",
              { class: "recent-main" },
              el("b", { text: String(s.url || "") }),
              el("small", {
                text: done
                  ? `${fmtScore(s.score)} out of 100 · ${fmtDate(s.at)}`
                  : `In progress · ${fmtDate(s.at)}`,
              }),
            ),
          ),
        );
      }),
    );
  }
  $("recent-clear").addEventListener("click", () => {
    saveRecent([]);
    renderRecent();
    $("url").focus();
  });

  // --- Starting a scan -------------------------------------------------
  function setFormError(message) {
    $("form-error").textContent = message || "";
    $("form-error").hidden = !message;
  }

  async function startScan(request) {
    setFormError("");
    $("scan-btn").disabled = true;
    $("scan-btn").classList.add("busy");
    $("scan-btn").textContent = "Starting";
    try {
      const resp = await fetch("/api/audit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      });
      const body = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        // 422 comes back as a list of field errors; everything else as text.
        const detail =
          typeof body.detail === "string"
            ? body.detail
            : "Enter a web address, like example.com.";
        throw new Error(detail);
      }
      lastSubmit = request;
      rememberScan({ id: body.job_id, url: request.url, at: Date.now() });
      location.hash = "job=" + body.job_id;
    } catch (err) {
      show("home");
      setFormError(
        err instanceof TypeError
          ? "Couldn't reach the scanner. Check your connection and try again."
          : err.message,
      );
      $("url").focus();
    } finally {
      $("scan-btn").disabled = false;
      $("scan-btn").classList.remove("busy");
      $("scan-btn").textContent = "Scan site";
    }
  }

  $("scan-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const url = $("url").value.trim();
    if (!url) {
      setFormError("Enter a web address, like example.com.");
      $("url").focus();
      return;
    }
    startScan({
      url,
      mode: $("mode").value,
      competitor_url: $("competitor").value.trim() || null,
    });
  });

  const syncModeHint = () => {
    $("mode-hint").textContent = MODE_HINTS[$("mode").value];
  };
  $("mode").addEventListener("change", syncModeHint);
  syncModeHint();

  // --- Running ---------------------------------------------------------
  function stopTimers() {
    clearTimeout(pollTimer);
    clearInterval(tickTimer);
    pollTimer = tickTimer = null;
  }

  // One plain sentence for what the scan is doing right now, read off the
  // same log lines the activity box shows.
  function currentActivity(logs, stage, finished, total) {
    const last = logs[logs.length - 1] || "";
    const wait = /rate limited, waiting ([\d.]+)s/.exec(last);
    if (wait)
      return {
        title: "Waiting for the AI service",
        detail: `It limits how fast we can ask. Resuming in about ${Math.ceil(Number(wait[1]))} s.`,
      };
    if (stage <= 0)
      return { title: "Starting the scan", detail: "Reaching the site." };
    if (stage === 1)
      return { title: "Planning the scan", detail: "Deciding which checks to run." };
    if (stage === 2)
      return {
        title: "Inspecting the live page",
        detail: total
          ? `${Math.min(finished, total)} of ${total} checks finished.`
          : "Running each check.",
      };
    if (stage === 3) {
      let round = 0;
      for (const line of logs) {
        const m = /^\[Critic\] round (\d+)/.exec(line);
        if (m) round = Number(m[1]);
      }
      return round
        ? {
            title: "Correcting the report",
            detail: `The reviewer asked for changes (round ${round}).`,
          }
        : {
            title: "Writing the report",
            detail: "Combining the checks, then double-checking the result.",
          };
    }
    return { title: "Saving the report", detail: "Almost done." };
  }

  function renderProgress(job) {
    $("running-url").textContent = job.url;
    const logs = job.logs || [];

    let stage = 0,
      total = 0,
      finished = 0;
    for (const line of logs) {
      const m = /^Stage (\d)\/4/.exec(line);
      if (m) stage = Math.max(stage, Number(m[1]));
      const d = /Dispatching (\d+) specialist/.exec(line);
      if (d) total = Number(d[1]);
      if (/specialist (done|FAILED)/.test(line)) finished += 1;
    }
    [...$("steps").children].forEach((li, i) => {
      li.dataset.state =
        i + 1 < stage ? "done" : i + 1 === stage ? "active" : "todo";
      if (i + 1 === stage) li.setAttribute("aria-current", "step");
      else li.removeAttribute("aria-current");
    });
    const now = currentActivity(logs, stage, finished, total);
    // Only touch the text when it changes, so screen readers aren't re-told.
    if ($("now-title").textContent !== now.title)
      $("now-title").textContent = now.title;
    if ($("now-detail").textContent !== now.detail)
      $("now-detail").textContent = now.detail;
    if (total)
      $("inspect-count").textContent =
        `${Math.min(finished, total)} of ${total} checks finished`;

    const log = $("log");
    const pinned = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
    for (let i = log.childElementCount; i < logs.length; i++) {
      log.append(
        el("div", {
          class: /^Stage \d\/4/.test(logs[i]) ? "stage" : "",
          text: logs[i],
        }),
      );
    }
    if (pinned) log.scrollTop = log.scrollHeight;
  }

  function beginRunning(job) {
    $("log").replaceChildren();
    $("now-title").textContent = "Starting the scan";
    $("now-detail").textContent = "Reaching the site.";
    $("inspect-count").textContent = "Run each check on the live page";
    show("running");
    const tick = () => {
      $("elapsed").textContent = fmtDuration(
        Date.now() / 1000 - job.started_at,
      );
    };
    tick();
    clearInterval(tickTimer);
    tickTimer = setInterval(tick, 1000);
  }

  async function poll(jobId, first) {
    let job;
    try {
      const resp = await fetch("/api/audit/" + encodeURIComponent(jobId));
      if (resp.status === 404) {
        stopTimers();
        forgetScan(jobId);
        return showFailed({
          title: "This scan isn't available",
          reason:
            "Finished reports are kept, so this scan most likely never finished: it was interrupted when the scanner restarted, or the link is incomplete.",
          help: "Run the scan again to get a fresh report.",
          canRetry: Boolean(lastSubmit),
        });
      }
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      job = await resp.json();
    } catch (err) {
      // A dropped connection shouldn't lose a scan that is still running.
      pollTimer = setTimeout(() => poll(jobId, first), POLL_MS * 2);
      return;
    }

    lastSubmit = lastSubmit || {
      url: job.url,
      mode: job.mode || "auto",
      competitor_url: job.competitor_url || null,
    };

    if (job.status === "running") {
      if (first || $("view-running").hidden) beginRunning(job);
      renderProgress(job);
      pollTimer = setTimeout(() => poll(jobId, false), POLL_MS);
    } else if (job.status === "done") {
      stopTimers();
      updateScan(jobId, {
        url: (job.report && job.report.url) || job.url,
        score: Number(job.report && job.report.overall_score),
        grade: job.report && job.report.grade,
      });
      renderResult(job);
    } else {
      stopTimers();
      forgetScan(jobId);
      showFailed({
        title: "The scan couldn't finish",
        url: job.url,
        reason: job.error || "The scanner stopped without saying why.",
        help: FAILURE_HELP[job.error_code] || FAILURE_HELP.internal,
        canRetry: true,
      });
    }
  }

  function showFailed({ title, url, reason, help, canRetry }) {
    $("failed-title").textContent = title;
    $("failed-url").textContent = url || "";
    $("failed-url").hidden = !url;
    $("failed-reason").textContent = reason;
    $("failed-help").textContent = help;
    $("failed-retry").hidden = !canRetry;
    show("failed");
  }

  // --- Result ----------------------------------------------------------
  function countBySeverity(report) {
    const counts = { critical: 0, warning: 0, good: 0 };
    for (const cat of report.categories || []) {
      for (const f of cat.findings || [])
        if (f.severity in counts) counts[f.severity] += 1;
    }
    return counts;
  }

  function renderResult(job) {
    const report = job.report || {};
    currentReport = report;
    filter = "all";
    const score = Number(report.overall_score);

    $("ring").style.setProperty("--grade", gradeVar(score));
    $("grade").textContent = report.grade || "?";
    $("ring").setAttribute("role", "img");
    $("ring").setAttribute(
      "aria-label",
      `Grade ${report.grade}, ${fmtScore(score)} out of 100`,
    );
    const ring = $("ring-value");
    ring.style.strokeDasharray = String(RING_LENGTH);
    ring.style.strokeDashoffset = String(RING_LENGTH);

    $("result-url").textContent = report.url || job.url;
    $("score-line").replaceChildren(
      el("strong", { text: fmtScore(score) }),
      " out of 100",
    );

    const badges = [];
    if (report.review_status === "approved") {
      badges.push(
        el(
          "span",
          { class: "badge good" },
          icon(SEVERITY.good.icon),
          "Reviewed and approved",
        ),
      );
    } else if (report.review_status === "not_approved") {
      badges.push(
        el(
          "span",
          { class: "badge warning" },
          icon(SEVERITY.warning.icon),
          "Reviewer did not sign off",
        ),
      );
    }
    const skipped = report.skipped_categories || [];
    if (skipped.length) {
      const ran = (report.categories || []).length;
      badges.push(
        el(
          "span",
          { class: "badge warning" },
          icon(SEVERITY.warning.icon),
          `${ran} of ${ran + skipped.length} checks ran`,
        ),
      );
    }
    const trend = report.trend;
    if (trend && Number.isFinite(Number(trend.score_delta))) {
      const delta = Number(trend.score_delta);
      const words =
        delta > 0
          ? `Up ${fmtScore(delta)} points`
          : delta < 0
            ? `Down ${fmtScore(-delta)} points`
            : "No change";
      badges.push(
        el("span", {
          class:
            "badge " + (delta > 0 ? "good" : delta < 0 ? "critical" : "plain"),
          text: `${words} since ${fmtDate(trend.previous_timestamp)}`,
        }),
      );
    }
    $("badges").replaceChildren(...badges);
    $("summary").textContent = report.summary || "";

    const counts = countBySeverity(report);
    $("n-findings").textContent = String(
      counts.critical + counts.warning + counts.good,
    );
    $("n-wins").textContent = String((report.quick_wins || []).length);

    renderFilters(counts);
    renderCategories();
    renderWins(report);
    renderDetails(job);
    $("pdf-link").href =
      "/api/audit/" + encodeURIComponent(currentJobId || "") + "/pdf";
    $("pdf-link").hidden = !currentJobId;
    selectTab("findings");
    show("result");

    // The one piece of motion on the page: the ring fills to the score.
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        ring.style.strokeDashoffset = String(
          RING_LENGTH * (1 - Math.min(Math.max(score, 0), 100) / 100),
        );
      }),
    );
  }

  function renderFilters(counts) {
    const total = counts.critical + counts.warning + counts.good;
    const make = (key, label) =>
      el("button", {
        type: "button",
        class: "chip",
        "aria-pressed": String(filter === key),
        text: label,
        onclick: () => {
          filter = key;
          renderFilters(counts);
          renderCategories();
        },
      });
    $("filters").replaceChildren(
      make("all", `All ${total}`),
      ...SEVERITY_ORDER.filter((s) => counts[s]).map((s) =>
        make(s, `${SEVERITY[s].plural} ${counts[s]}`),
      ),
    );
  }

  function renderCategories() {
    const categories = [...(currentReport.categories || [])].sort(
      (a, b) => a.score - b.score,
    );
    const nodes = [];
    for (const cat of categories) {
      const all = [...(cat.findings || [])].sort(
        (a, b) =>
          SEVERITY_ORDER.indexOf(a.severity) -
          SEVERITY_ORDER.indexOf(b.severity),
      );
      const visible =
        filter === "all" ? all : all.filter((f) => f.severity === filter);
      if (!visible.length) continue;

      const problems = all.filter((f) => f.severity !== "good").length;
      const meta = `${problems ? `${problems} to fix` : "Nothing to fix"}, counts for ${Math.round(cat.weight * 100)}% of the score`;
      const items = visible.map((f) => {
        const sev = SEVERITY[f.severity] || SEVERITY.warning;
        const fix = (f.recommendation || "").trim();
        const noAction =
          /^no (further )?(action|changes?)( is| are)? (needed|required)/i.test(
            fix,
          );
        return el(
          "li",
          { class: "finding " + f.severity },
          icon(sev.icon),
          el(
            "div",
            null,
            el(
              "p",
              { class: "issue" },
              el("span", { class: "sev", text: sev.label + ": " }),
              f.issue || "",
            ),
            fix && !noAction ? el("p", { class: "fix", text: fix }) : null,
          ),
        );
      });

      nodes.push(
        el(
          "details",
          {
            class: "category",
            style: `--grade:${gradeVar(cat.score)}`,
            open: filter !== "all" || problems > 0 ? "" : null,
          },
          el(
            "summary",
            null,
            el("h3", null, cat.name, el("span", { class: "meta", text: meta })),
            el(
              "div",
              { class: "meter", "aria-hidden": "true" },
              el("i", {
                style: `width:${Math.min(Math.max(cat.score, 0), 100)}%`,
              }),
            ),
            el("span", {
              class: "num",
              text: fmtScore(cat.score),
              "aria-label": `${fmtScore(cat.score)} out of 100`,
            }),
          ),
          el("ul", { class: "findings" }, ...items),
        ),
      );
    }
    $("categories").replaceChildren(
      ...(nodes.length
        ? nodes
        : [
            el("p", {
              class: "empty",
              text: "No findings in this report.",
            }),
          ]),
    );
  }

  function renderWins(report) {
    const wins = report.quick_wins || [];
    $("panel-wins").replaceChildren(
      wins.length
        ? el("ul", { class: "wins" }, ...wins.map((w) => el("li", { text: w })))
        : el("p", {
            class: "empty",
            text: "This report has no quick wins listed.",
          }),
    );
  }

  function renderDetails(job) {
    const report = job.report || {};
    const modeNames = {
      quick: "Quick",
      auto: "Standard",
      deep: "Thorough",
    };
    const fact = (term, value) => [
      el("dt", { text: term }),
      el("dd", { text: value }),
    ];
    const nodes = [
      el("h3", { text: "This scan" }),
      el(
        "dl",
        { class: "facts" },
        ...fact("Address", report.url || job.url),
        ...(job.mode ? fact("Depth", modeNames[job.mode] || job.mode) : []),
        ...(job.competitor_url
          ? fact("Compared with", job.competitor_url)
          : []),
        ...(job.finished_at
          ? fact("Finished", new Date(job.finished_at * 1000).toLocaleString())
          : []),
        ...(job.finished_at && job.started_at
          ? fact("Took", fmtDuration(job.finished_at - job.started_at))
          : []),
        ...fact(
          "Review",
          report.review_status === "approved"
            ? "Approved by the reviewer"
            : "Not approved by the reviewer",
        ),
      ),
    ];

    const issues = report.unresolved_review_issues || [];
    if (report.review_status === "not_approved") {
      nodes.push(el("h3", { text: "What the reviewer still objected to" }));
      nodes.push(
        el("p", {
          text: "A second model checks every report against the raw evidence. It did not approve this one, so read the findings with extra care.",
        }),
      );
      if (issues.length)
        nodes.push(el("ul", null, ...issues.map((i) => el("li", { text: i }))));
    }

    const skipped = report.skipped_categories || [];
    if (skipped.length) {
      nodes.push(el("h3", { text: "Checks that did not run" }));
      nodes.push(
        el("p", {
          text: "These are not part of the score. A missing check is not a passed check.",
        }),
      );
      nodes.push(
        el(
          "ul",
          null,
          ...skipped.map((c) =>
            el(
              "li",
              null,
              el("strong", { text: c.name + ": " }),
              c.reason || "",
            ),
          ),
        ),
      );
    }

    if (report.data_limitations) {
      nodes.push(el("h3", { text: "What this scan could not check" }));
      nodes.push(el("p", { text: report.data_limitations }));
    }

    const history = el(
      "div",
      null,
      el("p", { class: "empty", text: "Loading earlier scans…" }),
    );
    nodes.push(el("h3", { text: "Earlier scans of this site" }), history);
    $("panel-details").replaceChildren(...nodes);
    loadHistory(report.url || job.url, history);
  }

  async function loadHistory(url, container) {
    let rows = [];
    try {
      const host = new URL(url).host;
      const resp = await fetch(
        "/api/history/" + encodeURIComponent(host) + "?limit=10",
      );
      if (resp.ok) rows = (await resp.json()).history || [];
    } catch (err) {
      /* history is a nice-to-have; an empty table is fine */
    }

    if (rows.length < 2) {
      container.replaceChildren(
        el("p", {
          class: "empty",
          text: "This is the first scan of this site. Scan it again later to see how the score moves.",
        }),
      );
      return;
    }
    container.replaceChildren(
      el(
        "div",
        { class: "table-scroll" },
        el(
          "table",
          null,
          el(
            "thead",
            null,
            el(
              "tr",
              null,
              el("th", { text: "Date" }),
              el("th", { text: "Score" }),
              el("th", { text: "Grade" }),
            ),
          ),
          el(
            "tbody",
            null,
            ...rows.map((r) =>
              el(
                "tr",
                null,
                el(
                  "td",
                  null,
                  r.public_id && r.public_id !== currentJobId
                    ? el("a", {
                        href: "#job=" + encodeURIComponent(r.public_id),
                        text: new Date(r.timestamp).toLocaleString(),
                      })
                    : new Date(r.timestamp).toLocaleString(),
                ),
                el("td", { text: fmtScore(Number(r.overall_score)) }),
                el("td", { text: r.grade || "" }),
              ),
            ),
          ),
        ),
      ),
    );
  }

  // --- Tabs ------------------------------------------------------------
  const TABS = ["findings", "wins", "details"];
  function selectTab(name, focus) {
    for (const t of TABS) {
      const selected = t === name;
      $("tab-" + t).setAttribute("aria-selected", String(selected));
      $("tab-" + t).tabIndex = selected ? 0 : -1;
      $("panel-" + t).hidden = !selected;
    }
    if (focus) $("tab-" + name).focus();
  }
  TABS.forEach((name, i) => {
    $("tab-" + name).addEventListener("click", () => selectTab(name));
    $("tab-" + name).addEventListener("keydown", (event) => {
      const step =
        event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
      if (step) selectTab(TABS[(i + step + TABS.length) % TABS.length], true);
    });
  });

  // --- Light / dark mode -----------------------------------------------
  // Follows the system until the visitor uses the toggle; after that their
  // choice is remembered on this device.
  const systemDark = window.matchMedia("(prefers-color-scheme: dark)");
  const activeTheme = () =>
    document.documentElement.dataset.theme ||
    (systemDark.matches ? "dark" : "light");
  function syncThemeToggle() {
    const mode = activeTheme();
    const label =
      mode === "dark" ? "Switch to light mode" : "Switch to dark mode";
    $("theme-toggle").dataset.mode = mode;
    $("theme-toggle").setAttribute("aria-label", label);
    $("theme-toggle").title = label;
  }
  $("theme-toggle").addEventListener("click", () => {
    const next = activeTheme() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("theme", next);
    } catch (err) {
      /* choice lasts for this visit only */
    }
    syncThemeToggle();
  });
  systemDark.addEventListener("change", syncThemeToggle);
  syncThemeToggle();

  // --- Navigation ------------------------------------------------------
  function goHome() {
    stopTimers();
    if (location.hash) history.pushState(null, "", location.pathname);
    show("home");
    $("url").focus();
  }
  function rescan() {
    if (lastSubmit) startScan(lastSubmit);
    else goHome();
  }
  $("home-link").addEventListener("click", goHome);
  $("new-scan").addEventListener("click", goHome);
  $("failed-new").addEventListener("click", goHome);
  $("rescan").addEventListener("click", rescan);
  $("failed-retry").addEventListener("click", rescan);

  // The job id lives in the address, so a refresh or a shared link resumes
  // the same scan for as long as the server still has it.
  function route() {
    stopTimers();
    const match = /^#job=([0-9a-f]{8,64})$/.exec(location.hash);
    currentJobId = match ? match[1] : null;
    if (match) poll(match[1], true);
    else show("home");
  }
  window.addEventListener("hashchange", route);
  route();
})();
