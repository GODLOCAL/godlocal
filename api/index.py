import json
import time
from http.server import BaseHTTPRequestHandler


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split('?')[0]
        if path in ('/', '/health'):
            body = json.dumps({'status': 'ok', 'ts': int(time.time())}).encode()
        elif path in ('/status', '/mobile/status'):
            body = json.dumps({
                'kill_switch_active': False,
                'circuit_breaker': {'is_tripped': False, 'consecutive_losses': 0, 'daily_loss_sol': 0.0},
                'sparks': [],
                'thoughts': [],
                'ts': int(time.time())
            }).encode()
        else:
            body = json.dumps({'error': 'not found'}).encode()
            self.send_response(404)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        _ = self.rfile.read(length)
        body = json.dumps({'ok': True}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(body)
