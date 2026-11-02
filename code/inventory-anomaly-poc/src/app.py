"""Flask inventory site: browse counts, review model-flagged counts (confirm / dismiss)."""
import sqlite3
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template_string, request, url_for

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "inventory.db"
app = Flask(__name__)

BASE = """<!doctype html><title>Inventory review</title>
<style>body{font:15px system-ui;margin:2rem auto;max-width:900px;padding:0 1rem}table{border-collapse:collapse;width:100%}
td,th{border-bottom:1px solid #ddd;padding:.4rem .6rem;text-align:left}.pill{padding:.1rem .5rem;border-radius:1rem;background:#eee}
.pending{background:#fff3cd}.confirmed{background:#f8d7da}.dismissed{background:#d1e7dd}form{display:inline}</style>
<h1>Inventory anomaly review</h1><nav><a href="{{ url_for('queue') }}">Review queue</a> ·
<a href="{{ url_for('search') }}">Counts</a> · <a href="{{ url_for('api_flags') }}">/api/flags</a></nav><hr>{% block b %}{% endblock %}"""


def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


@app.route("/")
def queue():
    status = request.args.get("status", "pending")
    con = db()
    rows = con.execute("""SELECT f.rowid AS fid, f.item_id, f.location_id, f.week, f.score, f.review_status,
                                 c.count FROM flags f JOIN counts c USING(item_id, location_id, week)
                          WHERE f.review_status = ? ORDER BY f.score DESC LIMIT 200""", (status,)).fetchall()
    stats = dict(con.execute("SELECT review_status, COUNT(*) FROM flags GROUP BY review_status").fetchall())
    return render_template_string(BASE.replace("{% block b %}{% endblock %}", """
<p>{% for k,v in stats.items() %}<span class="pill {{k}}">{{k}}: {{v}}</span> {% endfor %}</p>
<table><tr><th>Item</th><th>Loc</th><th>Week</th><th>Count</th><th>Score</th><th>Status</th><th>Action</th></tr>
{% for r in rows %}<tr><td>{{r.item_id}}</td><td>{{r.location_id}}</td><td>{{r.week}}</td><td>{{r.count}}</td>
<td>{{'%.3f'|format(r.score)}}</td><td><span class="pill {{r.review_status}}">{{r.review_status}}</span></td><td>
<form method="post" action="{{ url_for('review', fid=r.fid) }}"><button name="decision" value="confirmed">Confirm anomaly</button>
<button name="decision" value="dismissed">Dismiss</button></form></td></tr>{% endfor %}</table>"""), rows=rows, stats=stats)


@app.post("/review/<int:fid>")
def review(fid):
    d = request.form.get("decision")
    if d not in ("confirmed", "dismissed"):
        return "bad decision", 400
    con = db()
    con.execute("UPDATE flags SET review_status = ? WHERE rowid = ?", (d, fid))
    con.commit()
    return redirect(url_for("queue"))


@app.route("/counts")
def search():
    item = request.args.get("item", type=int)
    loc = request.args.get("loc", type=int)
    rows = []
    if item is not None:
        q, args = "SELECT * FROM counts WHERE item_id = ?", [item]
        if loc is not None:
            q, args = q + " AND location_id = ?", args + [loc]
        rows = db().execute(q + " ORDER BY location_id, week DESC LIMIT 100", args).fetchall()
    return render_template_string(BASE.replace("{% block b %}{% endblock %}", """
<form method="get">item <input name="item" size="4" value="{{request.args.get('item','')}}"> location
<input name="loc" size="4" value="{{request.args.get('loc','')}}"> <button>Search</button></form>
<table><tr><th>Item</th><th>Loc</th><th>Week</th><th>Count</th></tr>
{% for r in rows %}<tr><td>{{r.item_id}}</td><td>{{r.location_id}}</td><td>{{r.week}}</td><td>{{r.count}}</td></tr>{% endfor %}</table>"""), rows=rows)


@app.route("/api/flags")
def api_flags():
    rows = db().execute("SELECT item_id, location_id, week, score, review_status FROM flags ORDER BY score DESC LIMIT 100").fetchall()
    return jsonify([dict(r) for r in rows])


if __name__ == "__main__":
    app.run(debug=False, port=5000)
