/* skai-metrics progress site — renders everything from window.SUMMARY */
(function () {
  "use strict";

  var S = window.SUMMARY;
  if (!S) {
    document.getElementById("stat-grid").textContent =
      "Could not load data/summary.js — regenerate it with: python3 scripts/make_figures.py";
    return;
  }

  var BANDS = ["g", "r", "i", "z", "Y"];
  var FIG_BASE = "../figures/";

  /* ---------- helpers ---------- */

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function fmtInt(x) {
    return Number(x).toLocaleString("en-US");
  }

  function fmt(x, d) {
    return Number(x).toFixed(d == null ? 3 : d);
  }

  function signed(x, d) {
    var v = Number(x);
    return (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(d == null ? 3 : d);
  }

  /* ---------- 1. hero ---------- */

  document.getElementById("project-title").textContent = S.project.title;
  document.getElementById("project-goal").textContent = S.project.goal;
  document.getElementById("project-phase").textContent = S.project.phase;

  var stats = [
    { label: "Exposures analyzed", value: fmtInt(S.dataset.n_survey_90s_clean),
      note: "clean 90 s survey exposures" },
    { label: "Nights scored", value: fmtInt(S.dataset.n_nights_scored),
      note: S.dataset.date_range[0] + " → " + S.dataset.date_range[1] },
    { label: "Validation log-correlation", value: fmt(S.validation.logcorr_calibrated, 3),
      note: "vs DES pipeline t_eff" },
    { label: "Median |error|", value: fmt(S.validation.median_abs_err_calibrated, 3),
      note: "calibrated · naive paper constants: " + fmt(S.validation.median_abs_err_naive_paper, 3) },
    { label: "Validation tests", value: S.validation.tests_passed + " / " + S.validation.tests_total,
      note: "passing on real data" },
    { label: "Median night score", value: fmt(S.nights.median_score, 1),
      note: "composite schedule score, 0–100" }
  ];

  var grid = document.getElementById("stat-grid");
  stats.forEach(function (s) {
    var tile = el("div", "stat");
    tile.appendChild(el("p", "label", s.label));
    tile.appendChild(el("p", "value", s.value));
    if (s.note) tile.appendChild(el("p", "note", s.note));
    grid.appendChild(tile);
  });

  var perBand = BANDS.map(function (b) {
    return b + " " + fmtInt(S.dataset.exposures_by_band[b]);
  }).join(" · ");
  document.getElementById("dataset-line").textContent =
    "Source: " + S.dataset.source + " — " + fmtInt(S.dataset.n_rows_raw) +
    " raw rows · exposures by band: " + perBand;

  /* ---------- figures (sections 3, 4, 5) ---------- */

  var lightbox = document.getElementById("lightbox");
  var lightboxImg = document.getElementById("lightbox-img");
  var lightboxCap = document.getElementById("lightbox-cap");
  var lastFocus = null;

  function openLightbox(src, title, caption, from) {
    lastFocus = from || null;
    lightboxImg.src = src;
    lightboxImg.alt = title;
    lightboxCap.textContent = title + " — " + caption;
    lightbox.hidden = false;
    document.body.style.overflow = "hidden";
    document.getElementById("lightbox-close").focus();
  }

  function closeLightbox() {
    lightbox.hidden = true;
    lightboxImg.src = "";
    document.body.style.overflow = "";
    if (lastFocus) lastFocus.focus();
  }

  lightbox.addEventListener("click", function (e) {
    if (e.target === lightbox) closeLightbox();
  });
  document.getElementById("lightbox-close").addEventListener("click", closeLightbox);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && !lightbox.hidden) closeLightbox();
  });

  function figCard(f) {
    var card = el("div", "fig-card");
    var btn = el("button", "fig-img");
    btn.type = "button";
    btn.setAttribute("aria-label", "View full size: " + f.title);
    var img = document.createElement("img");
    img.src = FIG_BASE + f.file;
    img.alt = f.title;
    img.loading = "lazy";
    btn.appendChild(img);
    btn.addEventListener("click", function () {
      openLightbox(img.src, f.title, f.caption, btn);
    });
    card.appendChild(btn);
    card.appendChild(el("p", "fig-title", f.title));
    card.appendChild(el("p", "fig-cap", f.caption));
    return card;
  }

  (S.figures || []).forEach(function (f) {
    var host = document.getElementById("figs-" + f.section);
    if (host) host.appendChild(figCard(f));
  });

  /* ---------- 3. validation ---------- */

  document.getElementById("validation-lede").textContent =
    "The reconstructed metric is checked against DES's own quality pipeline over " +
    fmtInt(S.dataset.n_survey_90s_clean) + " exposures: log-correlation " +
    fmt(S.validation.logcorr_calibrated, 3) + " after per-band calibration (" +
    fmt(S.validation.logcorr_blur_cloud_only, 3) + " with the blur × cloud core alone), " +
    "median absolute error " + fmt(S.validation.median_abs_err_calibrated, 3) +
    " versus " + fmt(S.validation.median_abs_err_naive_paper, 3) +
    " with naive paper constants. All " + S.validation.tests_total +
    " validation tests pass on real data.";

  var cal = S.calibration;
  var calRows = [
    { label: "Fiducial FWHM (arcsec)", data: cal.fiducial_fwhm_arcsec, d: 3 },
    { label: "Sky-brightness α", data: cal.sky_alpha, d: 3 },
    { label: "t_eff pass threshold", data: cal.teff_thresholds, d: 2 }
  ];
  var calTable = document.getElementById("calibration-table");
  var thead = el("thead");
  var hr = el("tr");
  hr.appendChild(el("th", null, "Parameter"));
  BANDS.forEach(function (b) { hr.appendChild(el("th", null, b)); });
  thead.appendChild(hr);
  calTable.appendChild(thead);
  var tbody = el("tbody");
  calRows.forEach(function (row) {
    var tr = el("tr");
    var th = el("th", null, row.label);
    th.setAttribute("scope", "row");
    tr.appendChild(th);
    BANDS.forEach(function (b) {
      tr.appendChild(el("td", null, fmt(row.data[b], row.d)));
    });
    tbody.appendChild(tr);
  });
  calTable.appendChild(tbody);

  /* ---------- 4. blur ---------- */

  document.getElementById("blur-callout-body").textContent =
    "Pooled across all exposures, blur appears to get better with airmass — the opposite of " +
    "atmospheric physics — because the scheduler only pointed low when the seeing was already " +
    "good (zenith seeing vs airmass correlation " + fmt(cal.zenith_seeing_airmass_corr, 3) + "). " +
    "Conditioning on the night recovers the physical X^0.6 law. Evaluating a scheduler from " +
    "logged data inherits the logging policy's choices; a median " +
    fmt(S.blur.median_teff_lost_to_pointing_pct, 1) + "% of t_eff is lost to pointing away from zenith.";

  var expChips = [
    { label: "Pooled fit (sign flipped)", val: signed(cal.airmass_exponent_fit_pooled, 3), cls: "exp-flipped" },
    { label: "Within-night fit (recovered)", val: signed(cal.airmass_exponent_fit_within_night, 3), cls: "exp-recovered" },
    { label: "Theory (X^0.6)", val: signed(cal.airmass_exponent_theory, 1), cls: "" }
  ];
  var expRow = document.getElementById("exponent-row");
  expChips.forEach(function (c) {
    var chip = el("div", "exp-chip" + (c.cls ? " " + c.cls : ""));
    chip.appendChild(el("span", "exp-val", c.val));
    chip.appendChild(document.createTextNode(c.label));
    expRow.appendChild(chip);
  });

  /* ---------- 5. schedule ---------- */

  document.getElementById("schedule-lede").textContent =
    "Every DES night gets a composite score (0–100) combining mean t_eff, open-shutter " +
    "efficiency, effective efficiency and the fraction of exposures passing survey thresholds. " +
    fmtInt(S.dataset.n_nights_scored) + " nights scored; median " + fmt(S.nights.median_score, 1) + ".";

  function nightsTable(tableEl, rows, scoreCls) {
    var cols = [
      { h: "Night", f: function (r) { return r.night; } },
      { h: "Score", f: function (r) { return fmt(r.score, 1); }, cls: scoreCls },
      { h: "Exp.", f: function (r) { return fmtInt(r.n_exposures); } },
      { h: "Mean t_eff", f: function (r) { return fmt(r.mean_teff, 3); } },
      { h: "Pass frac.", f: function (r) { return fmt(r.pass_fraction, 2); } },
      { h: "Med. airmass", f: function (r) { return fmt(r.median_airmass, 2); } }
    ];
    var thead = el("thead");
    var hr = el("tr");
    cols.forEach(function (c) { hr.appendChild(el("th", null, c.h)); });
    thead.appendChild(hr);
    tableEl.appendChild(thead);
    var tb = el("tbody");
    rows.forEach(function (r) {
      var tr = el("tr");
      cols.forEach(function (c) {
        var td = el("td", c.cls || null, c.f(r));
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
    tableEl.appendChild(tb);
  }

  nightsTable(document.getElementById("best-table"), S.nights.best, "score-good");
  nightsTable(document.getElementById("worst-table"), S.nights.worst, "score-crit");

  /* ---------- sparkline strip (inline SVG, no libraries) ---------- */

  (function sparkline() {
    var T = S.nights.timeline || [];
    if (!T.length) return;

    var W = 1000, H = 170;
    var M = { top: 10, right: 10, bottom: 26, left: 38 };
    var iw = W - M.left - M.right;
    var ih = H - M.top - M.bottom;

    var pts = T.map(function (d) {
      return { night: d.night, score: d.score, eff: d.effective_efficiency,
               t: Date.parse(d.night + "T00:00:00Z") };
    });
    var t0 = pts[0].t, t1 = pts[pts.length - 1].t;
    function X(t) { return M.left + (t - t0) / (t1 - t0) * iw; }
    function Y(s) { return M.top + (1 - s / 100) * ih; }
    pts.forEach(function (p) { p.x = X(p.t); p.y = Y(p.score); });

    var NS = "http://www.w3.org/2000/svg";
    function sv(tag, attrs) {
      var n = document.createElementNS(NS, tag);
      for (var k in attrs) n.setAttribute(k, attrs[k]);
      return n;
    }

    var svg = sv("svg", { viewBox: "0 0 " + W + " " + H, role: "img" });
    svg.setAttribute("aria-label",
      "Strip plot of the composite schedule score for all " + pts.length +
      " scored nights from " + T[0].night + " to " + T[T.length - 1].night + ".");

    // horizontal gridlines + score ticks
    [0, 25, 50, 75, 100].forEach(function (s) {
      svg.appendChild(sv("line", { class: "spark-grid",
        x1: M.left, x2: W - M.right, y1: Y(s), y2: Y(s) }));
      var t = sv("text", { class: "spark-tick", x: M.left - 7, y: Y(s) + 3.5,
        "text-anchor": "end" });
      t.textContent = s;
      svg.appendChild(t);
    });

    // year ticks along the bottom
    var y0 = new Date(t0).getUTCFullYear(), y1 = new Date(t1).getUTCFullYear();
    for (var yr = y0 + 1; yr <= y1; yr++) {
      var tx = X(Date.parse(yr + "-01-01T00:00:00Z"));
      svg.appendChild(sv("line", { class: "spark-grid",
        x1: tx, x2: tx, y1: H - M.bottom, y2: H - M.bottom + 4 }));
      var lbl = sv("text", { class: "spark-tick", x: tx, y: H - 8,
        "text-anchor": "middle" });
      lbl.textContent = yr;
      svg.appendChild(lbl);
    }

    // median reference line
    var med = S.nights.median_score;
    svg.appendChild(sv("line", { class: "spark-median",
      x1: M.left, x2: W - M.right, y1: Y(med), y2: Y(med) }));

    // one dot per night
    pts.forEach(function (p) {
      svg.appendChild(sv("circle", { class: "spark-dot", cx: p.x, cy: p.y, r: 2 }));
    });

    // hover marker + tooltip
    var hover = sv("circle", { class: "spark-hover", r: 4.5, cx: -20, cy: -20 });
    hover.style.display = "none";
    svg.appendChild(hover);

    var wrap = document.getElementById("spark-wrap");
    var tip = document.getElementById("spark-tip");
    var card = wrap.closest(".spark-card");
    wrap.appendChild(svg);

    document.getElementById("spark-sub").textContent =
      pts.length + " nights · gray line = median (" + fmt(med, 1) + ")";

    var xs = pts.map(function (p) { return p.x; }); // already sorted by date

    function nearest(vx) {
      var lo = 0, hi = xs.length - 1;
      while (lo < hi) {
        var mid = (lo + hi) >> 1;
        if (xs[mid] < vx) lo = mid + 1; else hi = mid;
      }
      if (lo > 0 && Math.abs(xs[lo - 1] - vx) < Math.abs(xs[lo] - vx)) lo--;
      return pts[lo];
    }

    function showTip(e) {
      var r = svg.getBoundingClientRect();
      var vx = (e.clientX - r.left) / r.width * W;
      var p = nearest(vx);
      hover.setAttribute("cx", p.x);
      hover.setAttribute("cy", p.y);
      hover.style.display = "";
      tip.innerHTML = "";
      tip.appendChild(el("div", "tip-night", p.night));
      var l1 = el("div", "tip-val", "score " + fmt(p.score, 1));
      var l2 = el("div", "tip-val", "effective efficiency " + fmt(p.eff, 3));
      tip.appendChild(l1);
      tip.appendChild(l2);
      tip.hidden = false;
      var cr = card.getBoundingClientRect();
      var px = (p.x / W) * r.width + (r.left - cr.left);
      var py = (p.y / H) * r.height + (r.top - cr.top);
      var tw = tip.offsetWidth;
      var left = px + 12;
      if (left + tw > cr.width - 8) left = px - tw - 12;
      tip.style.left = left + "px";
      tip.style.top = Math.max(4, py - tip.offsetHeight - 10) + "px";
    }

    function hideTip() {
      tip.hidden = true;
      hover.style.display = "none";
    }

    svg.addEventListener("mousemove", showTip);
    svg.addEventListener("mouseleave", hideTip);
  })();

  /* ---------- 6. progress timeline ---------- */

  var STATUS = {
    "done": { cls: "done", label: "Done" },
    "in progress": { cls: "inprogress", label: "In progress" },
    "next": { cls: "next", label: "Next" }
  };

  var tl = document.getElementById("timeline");
  (S.progress || []).forEach(function (p) {
    var st = STATUS[p.status] || STATUS.next;
    var li = el("li", st.cls);
    li.appendChild(el("span", "tl-dot"));
    var meta = el("div", "tl-meta");
    meta.appendChild(el("span", "tl-status", st.label));
    if (p.date) meta.appendChild(el("span", "tl-date", p.date));
    li.appendChild(meta);
    li.appendChild(el("p", "tl-text", p.milestone));
    tl.appendChild(li);
  });

  /* ---------- footer ---------- */

  document.getElementById("generated-line").textContent =
    "Data snapshot generated " + S.generated + " by scripts/make_figures.py.";

  /* ---------- nav scroll-spy ---------- */

  (function spy() {
    var links = Array.prototype.slice.call(document.querySelectorAll(".nav-links a"));
    var byId = {};
    links.forEach(function (a) { byId[a.getAttribute("href").slice(1)] = a; });
    var sections = Object.keys(byId)
      .map(function (id) { return document.getElementById(id); })
      .filter(Boolean);
    if (!("IntersectionObserver" in window)) return;
    var current = null;
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) {
          if (current) current.classList.remove("active");
          current = byId[en.target.id];
          if (current) current.classList.add("active");
        }
      });
    }, { rootMargin: "-20% 0px -70% 0px" });
    sections.forEach(function (s) { io.observe(s); });
  })();

})();
