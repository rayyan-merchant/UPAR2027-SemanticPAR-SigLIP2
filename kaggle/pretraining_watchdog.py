"""Bound ALL GPU pretraining setup, including imports/pip and blocked reads."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--marker',required=True);ap.add_argument('--report',required=True)
    ap.add_argument('--max-seconds',type=float,default=300);ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args()
    if Path(a.marker).exists():raise RuntimeError('Fresh training-start marker required')
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    p=subprocess.Popen(command,start_new_session=os.name!='nt');start=time.monotonic()
    while p.poll() is None:
        if Path(a.marker).exists():sys.exit(p.wait())
        if time.monotonic()-start>a.max_seconds:
            if os.name!='nt':os.killpg(p.pid,signal.SIGTERM)
            else:p.terminate()
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name!='nt':os.killpg(p.pid,signal.SIGKILL)
                else:p.kill()
            Path(a.report).write_text(json.dumps({'passed':False,'training_started':False,
                'error':'Pretraining wall-time exceeded; terminating job to release GPU',
                'elapsed_seconds':time.monotonic()-start,'max_seconds':a.max_seconds},indent=2))
            sys.exit(2)
        time.sleep(.5)
    sys.exit(p.returncode)

if __name__=='__main__':main()
