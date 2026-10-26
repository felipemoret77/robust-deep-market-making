from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys
root=Path(__file__).resolve().parents[1]
assert not (root/'launch.json').exists(),'Already launched; inspect status before resuming.'
with (root/'campaign.log').open('ab',buffering=0) as log:
 p=subprocess.Popen(['/usr/bin/nice','-n','12',sys.executable,'-B','-u',str(root/'code/run_campaign.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
c=subprocess.Popen(['/usr/bin/caffeinate','-dims','-w',str(p.pid)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
x=dict(pid=p.pid,caffeinate_pid=c.pid,launched_at=datetime.now(timezone.utc).isoformat(),workers=2,total_episodes=4810)
(root/'launch.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x,indent=2))
