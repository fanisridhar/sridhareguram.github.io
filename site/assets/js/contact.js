// Contact form: client-side validation, then POST to the Worker backend.
// With no endpoint configured it opens a prefilled email instead.
(function () {
  var form = document.getElementById("contact-form");
  if (!form) return;

  var cfg = window.PORTFOLIO || {};
  var statusEl = document.getElementById("form-status");
  var submitBtn = form.querySelector("button[type=submit]");
  var EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

  function setStatus(kind, text) {
    statusEl.hidden = false;
    statusEl.className = "form-status " + kind;
    statusEl.textContent = text;
  }

  function fieldError(name, message) {
    var input = form.elements[name];
    var err = document.getElementById(name + "-err");
    err.textContent = message || "";
    if (message) input.setAttribute("aria-invalid", "true");
    else input.removeAttribute("aria-invalid");
    return !message;
  }

  function validate(values) {
    var ok = true;
    ok = fieldError("name", values.name.length < 2 ? "Please enter your name." : "") && ok;
    ok = fieldError("email", !EMAIL_RE.test(values.email) ? "Please enter a valid email address." : "") && ok;
    ok =
      fieldError(
        "message",
        values.message.length < 10 ? "Please write at least a sentence." : values.message.length > 4000 ? "Please keep it under 4000 characters." : ""
      ) && ok;
    if (!ok) {
      var bad = form.querySelector("[aria-invalid=true]");
      if (bad) bad.focus();
    }
    return ok;
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var values = {
      name: form.elements.name.value.trim(),
      email: form.elements.email.value.trim(),
      message: form.elements.message.value.trim(),
      website: form.elements.website.value,
    };
    if (!validate(values)) return;

    if (!cfg.contactEndpoint) {
      var subject = encodeURIComponent("Portfolio message from " + values.name);
      var body = encodeURIComponent(values.message + "\n\n" + values.name + "\n" + values.email);
      window.location.href = "mailto:" + cfg.email + "?subject=" + subject + "&body=" + body;
      setStatus("ok", "Opening your email app with the message ready to send. If nothing opens, write to " + cfg.email + ".");
      return;
    }

    submitBtn.disabled = true;
    setStatus("ok", "Sending...");
    var ctrl = "AbortController" in window ? new AbortController() : null;
    var timer = ctrl ? setTimeout(function () { ctrl.abort(); }, 15000) : null;

    fetch(cfg.contactEndpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
      signal: ctrl ? ctrl.signal : undefined,
    })
      .then(function (res) {
        return res.json().catch(function () { return {}; }).then(function (data) {
          if (!res.ok) throw new Error(data.error || "Request failed");
          return data;
        });
      })
      .then(function () {
        form.reset();
        setStatus("ok", "Thanks, your message was sent. I will reply to " + values.email + ".");
      })
      .catch(function () {
        setStatus("bad", "Sorry, that did not go through. Please email me directly at " + cfg.email + ".");
      })
      .then(function () {
        if (timer) clearTimeout(timer);
        submitBtn.disabled = false;
      });
  });
})();
