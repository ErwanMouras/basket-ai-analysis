"""Follow the complete analysis: python -m pipeline.monitor --run runs/analysis/name."""

import argparse
import json
import os
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LABELS = {"loading": "Chargement des modèles", "perception": "Inférence vidéo",
          "recovery": "Relecture GPU des numéros et des maillots",
          "refinement": "Nettoyage des détections et intégration des preuves",
          "identity_motion": "Continuité des identités entre trajectoires",
          "color_features": "Mesures de couleur manquantes sur GPU",
          "color_calibration": "Calibration des couleurs sur ce match",
          "identity_colors": "Consolidation des couleurs", "identity_ocr": "Consolidation OCR",
          "identity_resolution": "Résolution des identités", "identity_collisions": "Vérification des identités",
          "identity_handoffs": "Raccords des trajectoires",
          "statistics": "Calcul des statistiques", "publication": "Écriture et empreintes",
          "health": "Vérification finale"}
TERMINAL = {"completed", "partial", "failed", "interrupted"}


def read_progress(run):
    path = Path(run) / "progress.json"
    if not path.is_file():
        return {"status": "waiting", "stage": "loading", "overall_percent": 0,
                "frames": 0, "stage_frames": 0, "elapsed_s": 0}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get('status') == 'running' and state.get('pid'):
        try:
            os.kill(state['pid'], 0)
        except ProcessLookupError:
            state.update(status='interrupted', error='Le processus d’analyse s’est arrêté avant la finalisation.')
    if state.get('status') == 'running' and state.get('last_progress_at'):
        idle = max(0, (datetime.now(UTC) - datetime.fromisoformat(state['last_progress_at'])).total_seconds())
        state['seconds_since_progress'] = idle
        if idle > 90:
            state.update(status='stalled', throughput_fps=None, perception_remaining_s=None,
                         error=f'Aucune progression depuis {idle / 60:.1f} min. Le processus est vivant, mais son activité ne prouve pas que les calculs avancent.')
    if state.get('status') != 'running':
        state.update(throughput_fps=None, perception_remaining_s=None)
    return state


def serve(run, port):
    page = Path(__file__).with_name("progress.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/progress.json":
                data = json.dumps({**read_progress(run), "labels": LABELS}, ensure_ascii=False).encode()
                mime = "application/json; charset=utf-8"
            elif self.path == "/":
                data, mime = page, "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"Suivi : http://127.0.0.1:{server.server_port}", flush=True)
        server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.serve:
        serve(args.run, args.port)
        return
    while True:
        p = read_progress(args.run)
        stage = LABELS.get(p["stage"], p["stage"])
        total = p.get("stage_total_frames")
        counters = f"{p.get('stage_frames', 0):,}/{total:,}" if total else "en cours"
        print(f"{p['overall_percent']:6.2f}% global estimé | {stage}: {counters} | "
              f"vidéo {p.get('frames', 0):,}/{p.get('total_frames') or '?'} | "
              f"{p.get('elapsed_s', 0) / 60:.1f} min | {p['status']}", flush=True)
        if p.get("error"):
            print(p["error"], flush=True)
        if args.once or p["status"] in TERMINAL:
            return
        time.sleep(2)


if __name__ == "__main__":
    main()
