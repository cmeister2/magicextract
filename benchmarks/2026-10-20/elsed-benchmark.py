from pathlib import Path
import sys,json,time,subprocess,statistics,platform,hashlib
import cv2,numpy as np
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'research/scripts/extended/learned_lines'))
from line_groups import propose
cv2.setNumThreads(1)
out=ROOT/'target/performance/elsed'
cli=ROOT/'target/research-v2/learned-lines/bin/elsed_cli'
files=['tests/fixtures/scenes/images/baseline-lightning-bolt-background-green.jpg','tests/fixtures/playmats/images/playmat-fantasy-forest-lightning-bolt.jpg','tests/fixtures/camera/images/camera-forest-normal-control.jpg']
records=[]
for file in files:
 source=cv2.imread(file);h,w=source.shape[:2]
 for region in ['full','central80']:
  image=source if region=='full' else source[round(.1*h):round(.9*h),round(.1*w):round(.9*w)]
  gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  for dim in [640,480,320]:
   scale=min(1,dim/max(gray.shape));working=cv2.resize(gray,(round(gray.shape[1]*scale),round(gray.shape[0]*scale)),interpolation=cv2.INTER_AREA) if scale<1 else gray
   path=out/'input.pgm';cv2.imwrite(str(path),working)
   def frontend():
    start=time.perf_counter();result=subprocess.run([str(cli),str(path)],capture_output=True,text=True,check=True);elapsed=time.perf_counter()-start
    return elapsed,json.loads(result.stdout)
   frontend();samples=[]
   for _ in range(5):
    elapsed,segments=frontend();samples.append(elapsed*1000)
   start=time.perf_counter();proposals,stats=propose(np.asarray([r[:4]for r in segments],np.float64).reshape(-1,2,2),working.shape[1],working.shape[0]);group=(time.perf_counter()-start)*1000
   record={'file':file,'region':region,'maximum_dimension':dim,'working_dimensions':[working.shape[1],working.shape[0]],'segments':len(segments),'frontend_process_ms_median':statistics.median(samples),'frontend_process_ms_samples':samples,'grouping_ms':group,'raw_proposals':len(proposals),'stats':stats}
   records.append(record);print(json.dumps(record),flush=True)
result={'hardware':platform.platform(),'cpu':next((r.split(':',1)[1].strip()for r in Path('/proc/cpuinfo').read_text().splitlines()if r.startswith('model name')),''),'scope':'single-thread author frontend through existing PGM CLI: includes process startup/read/JSON emission, excludes source decode, grayscale/resize, PGM write; existing Python grouping single sample; no shape verification or crops','elsed_binary_sha256':hashlib.sha256(cli.read_bytes()).hexdigest(),'records':records}
(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
