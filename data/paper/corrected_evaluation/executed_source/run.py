import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from multiprocessing import get_context
from collections import deque
from pathlib import Path
import csv,datetime,fcntl,json,subprocess,sys,time,traceback
import evaluate as e
R=e.ROOT
def now():return datetime.datetime.now().astimezone().isoformat()
def main():
 lock=(R/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((R/'validation/preflight.json').read_text())['status']=='passed'
 for name,h in json.loads((R/'source_manifest.json').read_text()).items():assert e.r.sha(R/name)==h,name
 cfg=e.P;pid=e.r.sha(R/'protocol.json');jobs=deque()
 # Group by figure family; never select seeds using outcomes.
 for case in sorted(cfg['cases'],key=lambda c:(0 if c['name'].startswith('illustration_') else 1 if c['family']=='asymmetry' else 2 if c['name'].startswith('recovery_') else 3)):
  for seed in case['seeds']:
   if not e.valid_record(case,seed,pid):jobs.append((case['name'],seed,pid))
 initial_rows=[json.loads(p.read_text()) for p in (R/'data').glob('*/*.json')]
 initial_events=sum(x['events'] for x in initial_rows);started=time.monotonic()
 def status(state):
  rows=[json.loads(p.read_text()) for p in (R/'data').glob('*/*.json')];done=len(rows);events=sum(x['events'] for x in rows);added=events-initial_events;elapsed=time.monotonic()-started
  info=dict(state=state,pid=os.getpid(),updated_at=now(),completed_episodes=done,total_episodes=cfg['total_episodes'],percent_episodes=100*done/cfg['total_episodes'],completed_events=events,total_events=cfg['total_events'],percent_events=100*events/cfg['total_events'],reused_episodes=sum(x['source']=='validated_corrected_pilot' for x in rows),new_episodes_completed=done-len(initial_rows),workers=cfg['workers'],elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed/added*(cfg['total_events']-events) if added>0 else None,folder=str(R),final_figures_folder=str(R.parent))
  e.r.atomic(R/'status.json',info);e.r.atomic(R.parent/'status_avaliacoes_completas.json',info)
 status('running')
 with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=get_context('spawn')) as pool:
  pending={}
  def submit():
   while jobs and len(pending)<2*cfg['workers']:
    j=jobs.popleft();pending[pool.submit(e.run_job,j)]=j
  submit()
  while pending:
   done,_=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
   for f in done:
    j=pending.pop(f);row=f.result();print(json.dumps(dict(case=j[0],seed=j[1],terminal_pnl=row['terminal_pnl'],seconds=row['elapsed_seconds'])),flush=True)
   submit();status('running')
 for case in cfg['cases']:
  for seed in case['seeds']:assert e.valid_record(case,seed,pid)
 rows=[json.loads(p.read_text()) for p in (R/'data').glob('*/*.json')]
 assert len(rows)==cfg['total_episodes'] and sum(x['events'] for x in rows)==cfg['total_events']
 with (R/'per_episode.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=sorted(set().union(*(x.keys() for x in rows))));w.writeheader();w.writerows(rows)
 status('rendering')
 subprocess.run([sys.executable,'-B',str(R/'render.py')],cwd=R,check=True)
 for model,spec in cfg['models'].items():assert e.r.sha(R/'source_snapshot'/spec['checkpoint'])==spec['sha256'],model
 protected=json.loads((R/'validation/protected_files.json').read_text());changed=[name for name,h in protected.items() if e.r.sha(name)!=h]
 e.r.atomic(R/'completion.json',dict(status='complete',completed_at=now(),episodes=len(rows),events=cfg['total_events'],protocol_sha256=pid,weights_unchanged=True,paper_and_slides_unchanged=not changed,protected_files_changed=changed,per_episode_sha256=e.r.sha(R/'per_episode.csv')))
 status('complete');print('COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  error=traceback.format_exc();e.r.atomic(R/'error.json',dict(at=now(),traceback=error));print(error,flush=True)
  path=R/'status.json'
  if path.exists():
   x=json.loads(path.read_text());x.update(state='failed',error_file=str(R/'error.json'));e.r.atomic(path,x);e.r.atomic(R.parent/'status_avaliacoes_completas.json',x)
  raise
