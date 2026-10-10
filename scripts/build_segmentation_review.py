"""Create one searchable local gallery and a prioritized queue for draft-mask review."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
from statistics import mean, median
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rice_disease.data import ROOT, load_manifest, write_json


PAGE = '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Rice mask review</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font:16px system-ui;margin:24px;background:#f3f5f7;color:#17252d}h1{margin-bottom:8px}
header{max-width:1000px}nav{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}
input,select,button{font:inherit;padding:8px;border:1px solid #b5c1c8;border-radius:6px}
article{background:white;padding:16px;margin:18px 0;border-radius:10px}h2{font-size:16px;overflow-wrap:anywhere}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0}img{width:100%;max-height:440px;object-fit:contain;background:#eee}
small{color:#50616b}a{color:#085c8a}.flag{color:#895200}#count{margin:10px 0}
</style><header><h1>Rice mask review</h1>
<p>One complete dataset. Cyan outlines the estimated leaf. Red shows <b>unconfirmed lesion candidates</b>.</p>
<p>Instance overlays show heuristic drafts. Reviewed semantic corrections are linked separately for each image.
Check missed pale damage, shadows and leaf boundaries.
Large regions are flagged, not automatically discarded. A blank lesion mask does not confirm health.</p>
<p><a href="summary.json">Summary</a> · <a href="validation.json">File validation</a> ·
<a href="review_queue.csv">Review queue</a> · <a href="REPORT.md">Report</a></p></header>
<nav><input id="search" placeholder="Search filename" aria-label="Search filename">
<select id="split" aria-label="Split"><option value="">All splits</option></select>
<select id="disease" aria-label="Disease"><option value="">All classes</option></select>
<select id="flag" aria-label="Review flag"><option value="">All review flags</option></select></nav>
<div id="count"></div><button id="previous">Previous</button> <button id="next">Next</button><main id="cards"></main>
<script id="records" type="application/json">__RECORDS__</script><script>
const rows=JSON.parse(document.getElementById('records').textContent);
const controls=Object.fromEntries(['search','split','disease','flag'].map(k=>[k,document.getElementById(k)]));
for(const [id,values] of [['split',rows.map(r=>r.split)],['disease',rows.map(r=>r.class_name)],['flag',rows.flatMap(r=>r.flags)]]){
 for(const value of [...new Set(values)].sort()){const o=document.createElement('option');o.value=value;o.textContent=value;controls[id].append(o);}}
let page=0;const size=12;const make=(tag,text)=>{const e=document.createElement(tag);e.textContent=text;return e;};
function render(){const filtered=rows.filter(r=>r.source_path.toLowerCase().includes(controls.search.value.toLowerCase())&&
 (!controls.split.value||r.split===controls.split.value)&&(!controls.disease.value||r.class_name===controls.disease.value)&&
 (!controls.flag.value||r.flags.includes(controls.flag.value)));
 const pages=Math.max(1,Math.ceil(filtered.length/size));page=Math.min(page,pages-1);
 document.getElementById('count').textContent=`${filtered.length} of ${rows.length} images · page ${page+1}/${pages} · priority order`;
 document.getElementById('previous').disabled=page===0;document.getElementById('next').disabled=page===pages-1;
 const cards=document.getElementById('cards');cards.replaceChildren();
 for(const r of filtered.slice(page*size,(page+1)*size)){
  const card=document.createElement('article');card.append(make('h2',r.source_path));
  card.append(make('p',`Semantic review: ${r.review_status} (${r.review_method})`));
  card.append(make('p',`${r.split} · candidate pixel coverage ${r.candidate_percent.toFixed(2)}% of estimated leaf (not measured severity)`));
  const flags=make('p',r.flags.join(' · '));flags.className='flag';card.append(flags);
  const pair=document.createElement('div');pair.className='pair';
  for(const [caption,url] of [['Original photo',r.source_url],['Draft overlay',r.overlay_url]]){
   const figure=document.createElement('figure');const link=document.createElement('a');link.href=url;link.target='_blank';link.rel='noopener';
   const img=document.createElement('img');img.src=url;img.loading='lazy';img.alt=caption+' '+r.source_path;link.append(img);
   figure.append(link,make('figcaption',caption));pair.append(figure);}
  card.append(pair);const link=make('a','Instance metadata and mask paths');link.href=r.metadata;card.append(link);
  if(r.semantic_url){card.append(make('span',' | '));const a=make('a','Current semantic mask');a.href=r.semantic_url;card.append(a);}
  if(r.review_url){card.append(make('span',' | '));const a=make('a','Reviewed correction gallery');a.href=r.review_url;card.append(a);}
  cards.append(card);
 }}
for(const e of Object.values(controls))e.addEventListener('input',()=>{page=0;render();});
document.getElementById('previous').onclick=()=>{page--;render();};document.getElementById('next').onclick=()=>{page++;render();};render();
</script></html>'''


def build(output, artifacts):
    output = Path(output).resolve()
    _, split = load_manifest(artifacts)
    summary = json.loads((output/'summary.json').read_text())
    if summary['status'] != 'complete' or summary['failed']:
        raise ValueError('Only complete exports can become a review gallery')
    with (output/'instances.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    annotation_manifest = Path(artifacts)/'segmentation_annotations/annotations.csv'
    annotations = {}
    if annotation_manifest.is_file():
        with annotation_manifest.open(newline='', encoding='utf-8') as stream:
            annotations = {r['image_path']: r for r in csv.DictReader(stream)}
    def url(path):
        return quote(Path(os.path.relpath(path, output)).as_posix())
    records = []
    for row in rows:
        flags = row['review_flags'].split(';')
        percent = 100 * int(row['candidate_lesion_pixels']) / max(1, int(row['leaf_pixels']))
        priority = sum({'complex_foreground_review': 100, 'no_leaf_detected': 100,
                        'high_candidate_coverage': 80, 'large_connected_candidate_review': 60,
                        'candidate_on_healthy_image': 50, 'no_candidate_on_diseased_image': 40}.get(f, 0)
                       for f in flags)
        relative = Path(row['split'])/row['class_name']/(Path(row['source_path']).name+'.jpg')
        annotated = annotations.get(row['source_path'], {})
        review = annotated.get('review_record')
        records.append(dict(row, flags=flags, priority=priority, candidate_percent=percent,
            review_status=annotated.get('review_status', 'unreviewed'),
            review_method=annotated.get('review_method') or 'none',
            semantic_url=url(annotation_manifest.parent/annotated['mask_path']) if annotated else None,
            review_url=url((ROOT/review).parent/'index.html') if review else None,
            source_url=quote(Path(os.path.relpath(Path(split['dataset'])/row['source_path'], output)).as_posix()),
            overlay_url=quote((Path('overlays')/relative).as_posix())))
    records.sort(key=lambda r: (-r['priority'], r['source_path']))
    payload = json.dumps(records).replace('<', '\\u003c')
    (output/'index.html').write_text(PAGE.replace('__RECORDS__', payload), encoding='utf-8')
    fields = ['priority', 'source_path', 'split', 'class_name', 'candidate_percent', 'review_flags',
              'review_status', 'review_method', 'semantic_url', 'review_url', 'metadata']
    with (output/'review_queue.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    print(f'Gallery and review queue: {len(records)} images in {output}')


def compare(output, previous):
    def read(folder):
        with (folder/'instances.csv').open(newline='', encoding='utf-8') as stream:
            return {r['source_path']: r for r in csv.DictReader(stream)}
    old, new = read(previous), read(output)
    if old.keys() != new.keys():
        raise ValueError('Before/after exports must contain the same images')
    def stats(rows):
        ratios = [100*int(r['candidate_lesion_pixels'])/max(1, int(r['leaf_pixels'])) for r in rows]
        return {'images': len(rows), 'mean_candidate_percent': mean(ratios),
                'median_candidate_percent': median(ratios), 'max_candidate_percent': max(ratios),
                'with_candidates': sum(int(r['lesion_candidates']) > 0 for r in rows),
                'above_50_percent': sum(p > 50 for p in ratios)}
    result = {'previous': str(previous), 'current': str(output), 'accuracy_evaluated': False,
              'note': 'Candidate coverage statistics, not accuracy. Lower coverage can also indicate missed pale damage. '
                      'All masks remain unreviewed; thresholds were chosen on training examples only.',
              'by_class': {}}
    for name in sorted({r['class_name'] for r in new.values()}):
        result['by_class'][name] = {version: stats([r for r in rows.values() if r['class_name'] == name])
                                    for version, rows in [('before', old), ('after', new)]}
    write_json(output/'repair_comparison.json', result)
    # Visual evidence uses training images only: original / old overlay / new overlay.
    from PIL import Image, ImageDraw, ImageOps
    selected = []
    for name in result['by_class']:
        group = [r for r in old.values() if r['split'] == 'train' and r['class_name'] == name]
        group.sort(key=lambda r: int(r['candidate_lesion_pixels'])/max(1, int(r['leaf_pixels'])), reverse=True)
        selected.append(group[0])
    dataset = Path(json.loads((output/'summary.json').read_text())['dataset'])
    sheet = Image.new('RGB', (1080, 300*len(selected)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(selected):
        relative = Path(row['split'])/row['class_name']/(Path(row['source_path']).name+'.jpg')
        paths = [dataset/row['source_path'], previous/'overlays'/relative, output/'overlays'/relative]
        for j, path in enumerate(paths):
            with Image.open(path) as im:
                tile = ImageOps.contain(im.convert('RGB'), (355, 265))
                sheet.paste(tile, (j*360, i*300+30))
            draw.text((j*360+4, i*300+4), [row['class_name'], 'Before (v2)', 'After (v3): still a draft'][j], fill='black')
    sheet.save(output/'repair_preview.jpg', quality=90)
    print('Saved repair comparison statistics and training-image preview.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/instance_segmentation')
    parser.add_argument('--artifacts', type=Path, default=ROOT/'artifacts')
    parser.add_argument('--previous', type=Path, help='Archived full export to compare with current drafts')
    args = parser.parse_args()
    build(args.output, args.artifacts)
    if args.previous:
        compare(args.output.resolve(), args.previous.resolve())
