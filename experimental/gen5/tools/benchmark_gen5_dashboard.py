#!/usr/bin/env python3
"""Four frozen ABBA runs with and without a deliberately busy local interface."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time
import urllib.request
from report_gen5_preflight import parity


def run(binary,base,output):
    output.mkdir(parents=True,exist_ok=False)
    results=[]
    for index,enabled in enumerate([False,True,True,False]):
        campaign=output/f'run-{index}';campaign.mkdir()
        config=dict(base,output=str(campaign/'training'),learn=False,human_fraction=0,human_dataset=None,
                    historical=False,history_interval=0,replay_index=None,seconds=180,games=256,game_seconds=30,seed=95818)
        (campaign/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        server=None;done=threading.Event();requests=[];errors=[]
        def poll():
            while not done.is_set():
                try:
                    t=time.monotonic()
                    with urllib.request.urlopen('http://127.0.0.1:18765/api/status',timeout=2) as response:
                        value=json.load(response)
                    requests.append(time.monotonic()-t)
                    if value['games']:
                        with urllib.request.urlopen('http://127.0.0.1:18765/examples/files/'+str(value['games'][0]['id'])+'.psr',timeout=2) as response:response.read()
                except Exception as e:errors.append(str(e))
                done.wait(.2)
        with (campaign/'training.log').open('wb') as log:
            if enabled:
                server=subprocess.Popen([sys.executable,str(Path(__file__).with_name('paisho_control.py')),'gen5','--campaign',str(campaign),'--port','18765'],stdout=log,stderr=log)
                for _ in range(100):
                    try:
                        urllib.request.urlopen('http://127.0.0.1:18765/',timeout=.2).close();break
                    except OSError:time.sleep(.02)
                else:raise RuntimeError('dashboard did not start')
                thread=threading.Thread(target=poll);thread.start()
            t=time.monotonic();child=subprocess.Popen([str(binary),'run',str(campaign/'config.json')],stdout=log,stderr=log)
            _,status,usage=os.wait4(child.pid,0);child.returncode=os.waitstatus_to_exitcode(status)
            wall=time.monotonic()-t
            done.set()
            cpu=0.
            if enabled:
                thread.join();server.terminate();_,status,u=os.wait4(server.pid,0);server.returncode=os.waitstatus_to_exitcode(status);cpu=u.ru_utime+u.ru_stime
            if child.returncode:raise RuntimeError('native run failed')
        report=json.loads((campaign/'training/report.json').read_text())
        results.append({'interface':enabled,'wall_seconds':wall,'native_cpu_seconds':usage.ru_utime+usage.ru_stime,'interface_cpu_seconds_including_startup':cpu,'requests':len(requests),'request_errors':errors,'max_request_seconds':max(requests,default=0),'report':report})
        print(json.dumps({k:v for k,v in results[-1].items() if k!='report'}),flush=True)
    checks=[parity(output/'run-0/training',output/f'run-{i}/training') for i in range(1,4)]
    off=sum(r['wall_seconds'] for r in results if not r['interface'])
    on=sum(r['wall_seconds'] for r in results if r['interface'])
    cpu=sum(r['interface_cpu_seconds_including_startup'] for r in results)
    result={'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'runs':results,'parity':checks,'wall_change_percent':100*(on/off-1),
            'interface_capacity_percent_including_startup':100*cpu/(on*base['threads']),
            'poll_seconds':.2,'normal_poll_seconds':5,'interpretation':'Frozen ABBA plus measured interface CPU; short timing variation is not a statistical proof of a universal 1% bound.'}
    (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('runs','parity')}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.binary.resolve(),json.loads(a.config.read_text()),a.output.resolve())
