"""Rebuild and independently verify all filling certificates and holonomy checks."""
from pathlib import Path
import subprocess,sys,json
HERE=Path(__file__).resolve().parent/'window'
for script in ('engine_a.py','engine_b.py'):
 result=subprocess.run([sys.executable,'-B',script],cwd=HERE,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=600)
 print(result.stdout,end='')
 if result.returncode:print(result.stderr);raise SystemExit(result.returncode)
print('All window checks passed')
