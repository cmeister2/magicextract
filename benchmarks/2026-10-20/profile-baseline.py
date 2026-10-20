#!/usr/bin/env python3
"""Reproduce coarse stage timings from the source revision preceding optimization."""
from pathlib import Path
import re
import json
import subprocess
import sys
root=Path(__file__).resolve().parents[2]
out=root/'target/performance/stage-profile'
out.mkdir(parents=True,exist_ok=True)
baseline='63d2bbaf3626f77254a8d9c53d4b2cf5465d96e5'
s=subprocess.check_output(['git','show',baseline+':src/detection.rs'],cwd=root,text=True)
s=s.replace('    edges::canny,','')
s=s.replace('use crate::{CardDetection','use crate::upstream_edges::canny;\nuse crate::{CardDetection')
selected={'detect','working_image','collect_candidates','collect_inferred_candidates','refine_gradient_corners','expand_card_border','recover_inner_artwork','recover_inner_card_border','refine_occluded_outlines','recover_colour_perimeters','recover_guided_colour_perimeters','projective_card_assignments','simplify_closed','convex_hull'}
for m in reversed(list(re.finditer(r'(?m)^(?:pub\(crate\) )?fn (\w+)\(',s))):
 if m[1] in selected:
  at=s.index('{',m.end())+1
  s=s[:at]+'\n let _profile = crate::Timer::new("'+m[1]+'");'+s[at:]
(out/'detection.rs').write_text(s)
metadata=json.loads(subprocess.check_output(['cargo','metadata','--locked','--format-version=1'],cwd=root,text=True))
imageproc=next(p for p in metadata['packages'] if p['name']=='imageproc' and p['version']=='0.27.0')
e=(Path(imageproc['manifest_path']).parent/'src/edges.rs').read_text()
e=e.replace('crate::definitions','imageproc::definitions').replace('crate::filter','imageproc::filter').replace('crate::kernel','imageproc::kernel')
e=e.replace('let blurred = gaussian_blur_f32(image, SIGMA);','let blurred = { let _t = crate::Timer::new("canny_gaussian"); gaussian_blur_f32(image, SIGMA) };')
e=e.replace('let gx = filter_clamped(&blurred, kernel::SOBEL_HORIZONTAL_3X3);','let gx = { let _t = crate::Timer::new("canny_sobel_x"); filter_clamped(&blurred, kernel::SOBEL_HORIZONTAL_3X3) };')
e=e.replace('let gy = filter_clamped(&blurred, kernel::SOBEL_VERTICAL_3X3);','let gy = { let _t = crate::Timer::new("canny_sobel_y"); filter_clamped(&blurred, kernel::SOBEL_VERTICAL_3X3) };')
e=e.replace('let g: Vec<f32> = gx','let _magnitude_timer = crate::Timer::new("canny_magnitude");\n    let g: Vec<f32> = gx')
e=e.replace('let g = Image::from_raw(image.width(), image.height(), g).unwrap();','let g = Image::from_raw(image.width(), image.height(), g).unwrap();\n drop(_magnitude_timer);')
e=e.replace('let thinned = non_maximum_suppression(&g, &gx, &gy);','let thinned = { let _t = crate::Timer::new("canny_nms"); non_maximum_suppression(&g, &gx, &gy) };')
e=e.replace('hysteresis(&thinned, low_threshold, high_threshold)','{ let _t = crate::Timer::new("canny_hysteresis"); hysteresis(&thinned, low_threshold, high_threshold) }')
(out/'upstream_edges.rs').write_text(e)
(out/'main.rs').write_text('''use std::{cell::RefCell, collections::BTreeMap, time::{Duration, Instant}};
pub use magicextract::{CardDetection, ExtractorConfig, Point, TRADING_CARD_ASPECT_RATIO};
mod detection;
mod upstream_edges;
thread_local! { static TIMES: RefCell<(Vec<(Instant,Duration)>, BTreeMap<&'static str,(usize,Duration,Duration)>)> = RefCell::new((Vec::new(), BTreeMap::new())); }
struct Timer { name: &'static str }
impl Timer { fn new(name: &'static str) -> Self { TIMES.with(|t| t.borrow_mut().0.push((Instant::now(), Duration::ZERO))); Self{name} } }
impl Drop for Timer { fn drop(&mut self) { TIMES.with(|t| { let mut t=t.borrow_mut(); let (start,children)=t.0.pop().unwrap(); let elapsed=start.elapsed(); let value=t.1.entry(self.name).or_default(); value.0+=1; value.1+=elapsed; value.2+=elapsed-children; if let Some(parent)=t.0.last_mut() {parent.1+=elapsed;} }); } }
fn main() { let args=std::env::args().skip(1).collect::<Vec<_>>(); let dimension=args[0].parse().unwrap(); for path in &args[1..] { let image=image::open(path).unwrap(); let config=ExtractorConfig{max_detection_dimension:dimension,..Default::default()}; let _=detection::detect(&image,&config); TIMES.with(|t| t.borrow_mut().1.clear()); for _ in 0..4 { std::hint::black_box(detection::detect(&image,&config)); } println!("IMAGE {path} DIM {dimension}"); TIMES.with(|t| {for (name,(calls,total,exclusive)) in &t.borrow().1 {println!("{name:32} {calls:5} total_ms {:9.3} exclusive_ms {:9.3}",total.as_secs_f64()*250.,exclusive.as_secs_f64()*250.);} }); } }
''')

artifacts=subprocess.check_output(['cargo','build','--locked','--release','--example','evaluate_scenes','--message-format=json'],cwd=root,text=True)
libraries={}
for line in artifacts.splitlines():
 event=json.loads(line)
 if event.get('reason')=='compiler-artifact' and event['target']['name'] in {'image','imageproc','magicextract'}:
  libraries[event['target']['name']]=next(Path(f) for f in event['filenames'] if f.endswith('.rlib'))
command=['rustc','--edition=2024','-C','opt-level=3',str(out/'main.rs'),'-o',str(out/'profile'),'-L','dependency='+str(libraries['image'].parent)]
for name,path in libraries.items(): command+=['--extern',name+'='+str(path)]
subprocess.run(command,cwd=root,check=True)
args=sys.argv[1:] or ['640','tests/fixtures/scenes/images/baseline-black-lotus-jpeg.jpg','tests/fixtures/scenes/images/baseline-sol-ring-perspective-1.jpg','tests/fixtures/scenes/images/occlusion-two-cards-1.jpg']
subprocess.run([str(out/'profile'),*args],cwd=root,check=True)
