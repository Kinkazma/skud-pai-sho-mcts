#!/usr/bin/env python3
"""Loopback-only English training control. Does not start a run when opened."""
import argparse,json,os,secrets,subprocess,sys,time
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SCALE=(ROOT/'cli.mjs').exists()
TOKEN=secrets.token_hex(24)
process=None
active=None
settings=None
log=None
HTML='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Pai Sho training</title>
<style>body{font:17px system-ui;background:#f2f3ef;color:#23342c;max-width:1050px;margin:40px auto;padding:0 22px}h1{font-size:32px}section{background:white;border:1px solid #d7ded6;border-radius:12px;padding:20px;margin:18px 0}label{display:inline-block;margin:8px}input,select,button{font:inherit;padding:7px;border:1px solid #9aa89c;border-radius:6px}input{width:120px}button{cursor:pointer;background:#e5eee3}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}#error{color:#9d2b2b}.muted{color:#576b5c}</style>
<h1>Pai Sho — training control</h1><p class="muted">Generation, model and search budget are separate. Opening this page does not start training.</p>
<section><h2>New bounded run</h2><label>Generation <select id="generation">GENERATIONS</select></label><label>Budget <select id="budget"><option>8</option><option selected>32</option></select></label><label>Seconds <input id="seconds" type="number" value="60" min="1" max="3600"></label><label>Workers <input id="workers" type="number" value="2" min="1" max="64"></label><p><button onclick="action('start')">Start from frozen weights</button> <button onclick="action('pause')">Request cooperative pause</button> <button onclick="action('resume')">Continue saved run</button></p><p>MODE_NOTE</p></section>
<section><h2>Observed state</h2><p id="summary">Loading…</p><p id="error"></p><pre id="state"></pre></section>
<script>const token='TOKEN';async function action(command){try{const body={command,generation:document.querySelector('#generation').value,budget:+document.querySelector('#budget').value,seconds:+document.querySelector('#seconds').value,workers:+document.querySelector('#workers').value};const r=await fetch('/action',{method:'POST',headers:{'Content-Type':'application/json','X-Control-Token':token},body:JSON.stringify(body)});const d=await r.json();if(!r.ok)throw Error(d.error);document.querySelector('#error').textContent='';await refresh()}catch(e){document.querySelector('#error').textContent=e.message}}async function refresh(){const d=await(await fetch('/state')).json();document.querySelector('#summary').textContent=d.running?'Running':d.exitCode===null?'Idle or completed':'Exited: '+d.exitCode;document.querySelector('#state').textContent=JSON.stringify(d,null,2)}refresh();setInterval(refresh,3000);</script></html>'''
def state():
 data={"kind":"Scale evolution" if SCALE else "Skud Gen3 training","running":process is not None and process.poll() is None,"exitCode":None if process is None else process.poll(),"output":str(active.relative_to(ROOT)) if active else None,"settings":settings}
 if active:
  for name in ['status.json','progress.json','summary.json']:
   p=active/name
   if p.is_file():
    try:
     value=json.loads(p.read_text());data[name]={k:v for k,v in value.items() if k not in ['recent_games','last_game','champions','initialParentsByBudget','referencesByBudget','referencePoolByBudget']}
     for k in ['model','replay_index']:
      if isinstance(data[name].get(k),str):data[name][k]=data[name][k].replace(str(ROOT)+'/', '')
    except (ValueError,OSError):pass
  p=active.with_suffix('.log')
  if p.is_file():
   with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2500));data['logTail']=f.read().decode(errors='replace')
 return data
class Handler(BaseHTTPRequestHandler):
 def respond(self,code,value,kind='application/json'):
  body=(json.dumps(value) if kind=='application/json' else value).encode();self.send_response(code);self.send_header('Content-Type',kind+'; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
 def do_GET(self):
  if self.path=='/state':self.respond(200,state());return
  if self.path!='/':self.respond(404,{'error':'Not found'});return
  generations=['1','2','3'] if SCALE else ['3.2','3.3','3.4','3.5'];selected=str(json.loads((ROOT/'release.json').read_text())['defaultGeneration'])
  if selected not in generations:selected=generations[-1]
  html=HTML.replace('GENERATIONS',''.join(f'<option {"selected" if g==selected else ""}>{g}</option>' for g in generations)).replace('TOKEN',TOKEN).replace('MODE_NOTE','Scale continues within the original deadline; paused time is not added back.' if SCALE else 'Skud continuation preserves the previous generation and budget, and restores its durable model and replay into a new output. Only duration and worker count change; this is not a bit-identical thread restart.')
  self.respond(200,html,'text/html')
 def do_POST(self):
  global process,active,log,settings
  if self.path!='/action' or self.headers.get('X-Control-Token')!=TOKEN:self.respond(403,{'error':'Invalid local control request'});return
  try:
   size=int(self.headers.get('Content-Length','0'))
   if not 0<size<=4096:raise ValueError('Invalid request length')
   a=json.loads(self.rfile.read(size));command=a['command'];running=process is not None and process.poll() is None
   if command=='pause':
    if not running or not active:raise ValueError('No active run')
    (active/('STOP' if SCALE else 'pause-request.json')).write_text('{}\n');self.respond(200,{'requested':'pause'});return
   if running:raise ValueError('A run is already active')
   if command not in ['start','resume']:raise ValueError('Unknown action')
   for k in ['seconds','workers','budget']:
    if type(a[k]) is not int or a[k]<1:raise ValueError('Positive integer settings required')
   if a['seconds']>3600 or a['workers']>min(64,os.cpu_count() or 1) or a['budget'] not in [8,32]:raise ValueError('Setting out of range')
   if a['generation'] not in (['1','2','3'] if SCALE else ['3.2','3.3','3.4','3.5']):raise ValueError('Invalid generation')
   previous=active
   if command=='resume' and settings:
    a['generation']=settings['generation'];a['budget']=settings['budget']
   if command=='resume' and not previous:raise ValueError('No saved run in this dashboard session. Use the documented CLI for earlier runs.')
   if SCALE and command=='resume':argv=['node','cli.mjs','resume','--output',str(previous)]
   else:
    active=ROOT/'runs'/('ui-'+str(time.time_ns()));active.parent.mkdir(exist_ok=True)
    argv=(['node','cli.mjs','train','--budgets',str(a['budget'])] if SCALE else [sys.executable,'manage.py','train','--budget',str(a['budget'])])+['--generation',a['generation'],'--seconds',str(a['seconds']),'--workers',str(a['workers']),'--output',str(active)]
    if not SCALE and a['budget']==8:argv+=['--heuristic-reference']
    if command=='resume':argv+=['--resume-from',str(previous)]
   settings={k:a[k] for k in ['generation','budget','seconds','workers']}
   if log:log.close()
   log=active.with_suffix('.log').open('ab',buffering=0);process=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
   self.respond(200,{'started':str(active.relative_to(ROOT))})
  except (ValueError,KeyError,OSError) as error:self.respond(400,{'error':str(error)})
 def log_message(self,*args):pass
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8770);a=p.parse_args();server=HTTPServer(('127.0.0.1',a.port),Handler)
 print(f'Open http://127.0.0.1:{a.port}/ (loopback only)',flush=True)
 try:server.serve_forever()
 except KeyboardInterrupt:
  if process and process.poll() is None and active:
   (active/('STOP' if SCALE else 'pause-request.json')).write_text('{}\n');process.wait()
 finally:server.server_close()
