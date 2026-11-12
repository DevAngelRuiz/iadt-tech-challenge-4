"""Painel receptor da equipe, servido somente na interface local."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from urllib.parse import urlparse
from src.reporting.alert_store import AlertStore

PAGE = """<!doctype html><html lang="pt-BR"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Central de alertas</title>
<style>body{font:16px system-ui;background:#f3f6fa;color:#172438;margin:32px auto;max-width:1000px;padding:0 20px}
header{display:flex;justify-content:space-between;align-items:center}article{background:white;padding:20px;margin:14px 0;border-radius:12px;border-left:6px solid #d28b10}
article.critical{border-color:#c43142}small{color:#526275}pre{white-space:pre-wrap;background:#f3f6fa;padding:12px;overflow-wrap:anywhere}
button{padding:10px;border:0;border-radius:6px;background:#234c82;color:white;cursor:pointer}#status{color:#526275}</style>
<header><h1>Central de alertas</h1><span>Protótipo acadêmico</span></header>
<p>Alertas recebidos do monitor e da análise multimodal. Os cenários de demonstração usam dados sintéticos.</p>
<p id="status">Conectando ao receptor local…</p><main id="alerts"></main>
<script>
const node=(tag,text)=>{let e=document.createElement(tag);e.textContent=text;return e};
async function refresh(){try{const response=await fetch('/api/alerts');if(!response.ok)throw Error();
 const data=await response.json();document.querySelector('#status').textContent=`${data.length} alertas recebidos · atualização ${new Date().toLocaleTimeString('pt-BR')}`;
 const root=document.querySelector('#alerts');root.replaceChildren();if(!data.length)root.append(node('p','Nenhum alerta recebido. O receptor está pronto.'));
 for(const a of data){let card=document.createElement('article');if(a.severity==='CRITICO')card.className='critical';
 card.append(node('h2',`${a.severity} · ${a.type}`),node('p',a.message),
 node('small',`Paciente: ${a.patient_id||'demo'} · Origem: ${a.source||'multimodal'} · Recebido: ${a.received_at}`));
 if(a.timestamp)card.append(node('p',`Instante no dado: ${a.timestamp}`));
 if(a.evidence)card.append(node('pre',JSON.stringify(a.evidence,null,2)));
 if(a.processing_latency_ms!=null)card.append(node('p',`Tempo de processamento até enfileiramento: ${a.processing_latency_ms} ms`));
 if(a.acknowledged_at)card.append(node('p',`Leitura confirmada: ${a.acknowledged_at}`));
 else{let b=node('button','Confirmar leitura');b.onclick=async()=>{b.disabled=true;const r=await fetch('/api/alerts/'+a.id+'/ack',{method:'POST'});if(!r.ok)alert('Falha ao confirmar leitura');await refresh()};card.append(b)}root.append(card)}
 }catch(e){document.querySelector('#status').textContent='Sem conexão com o receptor. Tentando novamente…'}}
refresh();setInterval(refresh,1000);
</script></html>"""


def create_server(store, port=8765):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, body, content_type="application/json; charset=utf-8"):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self.reply(200, PAGE, "text/html; charset=utf-8")
            elif path == "/api/alerts":
                self.reply(200, json.dumps(store.list(), ensure_ascii=False))
            else:
                self.reply(404, '{}')

        def do_POST(self):
            origin = self.headers.get("Origin")
            allowed = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
            if origin and origin not in allowed:
                self.reply(403, '{}')
                return
            parts = urlparse(self.path).path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "alerts"] and parts[3] == "ack":
                self.reply(200 if store.acknowledge(parts[2]) else 404, '{}')
            else:
                self.reply(404, '{}')

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default="outputs/alerts.sqlite")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = create_server(AlertStore(args.store), args.port)
    print(f"Receptor disponível em http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
