import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
from collections import deque
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from multiprocessing import get_context
from datetime import datetime,timezone
import fcntl,json,subprocess,sys,time,traceback
import experiment_runtime as r
ROOT=r.ROOT

def now():return datetime.now(timezone.utc).isoformat()
def main():
 lock=(ROOT/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((ROOT/'validation/preflight.json').read_text())['status']=='passed'
 for rel,digest in json.loads((ROOT/'source_manifest.json').read_text()).items():assert r.sha(ROOT/rel)==digest,rel
 config=json.loads((ROOT/'protocol.json').read_text());entries=config['experiments']
 r.atomic(ROOT/'execution_manifest.json',dict(started_at=now(),protocol_sha256=r.sha(ROOT/'protocol.json'),source_manifest_sha256=r.sha(ROOT/'source_manifest.json'),code_sha256={p.name:r.sha(p) for p in (ROOT/'code').glob('*.py')},python=sys.version))
 jobs=deque();expected={};paths={e['spec']['name']:[] for e in entries};lookup={e['spec']['name']:e for e in entries}
 for e in entries:
  name=e['spec']['name'];expected[name]=0
  for n in range(0,len(e['seeds']),config['chunk_size']):
   jobs.append((e['spec'],e['params'],e['seeds'][n:n+config['chunk_size']]));expected[name]+=1
 completed=events=0;merged=set();started=time.monotonic();milestones=set()
 def status(state):
  elapsed=time.monotonic()-started
  estimate=(config['total_evaluation_events']-events)/(events/elapsed) if events>0 else None
  r.atomic(ROOT/'status.json',dict(state=state,pid=os.getpid(),updated_at=now(),completed_episodes=completed,total_episodes=config['total_episodes'],percent_episodes=100*completed/config['total_episodes'],completed_evaluation_events=events,total_evaluation_events=config['total_evaluation_events'],percent_events=100*events/config['total_evaluation_events'],reused_naive_episodes=1000,completed_experiments=sorted(merged),elapsed_seconds=elapsed,estimated_remaining_seconds=estimate))
 def render(final=False):
  cmd=[sys.executable,'-B',str(ROOT/'code/render.py')]+(['--final'] if final else [])
  subprocess.run(cmd,cwd=ROOT,check=True)
 render();status('running')
 with ProcessPoolExecutor(max_workers=config['workers'],mp_context=get_context('spawn')) as pool:
  pending={}
  def submit():
   while jobs and len(pending)<2*config['workers']:
    j=jobs.popleft();pending[pool.submit(r.run_chunk,j)]=j
  submit()
  while pending:
   done,_=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
   for f in done:
    spec,params,seeds=pending.pop(f);name=spec['name'];path=f.result();paths[name].append(path)
    completed+=len(seeds);events+=len(seeds)*spec.get('events',5000)
    if len(paths[name])==expected[name]:
     r.merge(lookup[name],paths[name]);merged.add(name)
    print(json.dumps(dict(completed=completed,total=config['total_episodes'],experiment=name,elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
   submit();status('running')
   stages={'eta':['eta'],'fig3':[n for n in expected if n.startswith('fig3')],'fig2':[n for n in expected if n.startswith('fig2')],'fig4':[n for n in expected if n.startswith('fig4')]}
   ready={stage for stage,names in stages.items() if set(names)<=merged}
   if ready-milestones and jobs or ready-milestones and pending:
    render();milestones|=ready
 assert completed==config['total_episodes'] and events==config['total_evaluation_events']
 status('rendering');render(final=True)
 r.atomic(ROOT/'completion.json',dict(status='complete',completed_at=now(),episodes=completed,evaluation_events=events,parameters=config['parameters'],all_figures_generated=True,published_assets_modified=False))
 status('complete');print('COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  error=traceback.format_exc();r.atomic(ROOT/'error.json',dict(at=now(),traceback=error));print(error,flush=True);raise
