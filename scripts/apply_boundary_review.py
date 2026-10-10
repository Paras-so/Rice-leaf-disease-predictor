"""Render traceable AI visual boundary edits, then commit explicitly accepted masks."""
import argparse
import csv
from html import escape
import json
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage as ndi
from rice_disease.data import ROOT, load_manifest, write_json
from rice_disease.segmentation_data import sha256
from rice_disease.instance_segmentation import components


def polygon_mask(size, polygons):
    image = Image.new('L', size)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        draw.polygon([(round(x*size[0]/512), round(y*size[1]/512)) for x, y in polygon], fill=1)
    return np.array(image, dtype=bool)


def correct(rgb, old, edit):
    leaf = polygon_mask(rgb.size, [edit['leaf_polygon']]) if 'leaf_polygon' in edit else ndi.binary_fill_holes(old > 0)
    if edit.get('leaf_holes'):
        leaf &= ~polygon_mask(rgb.size, edit['leaf_holes'])
    lesion = np.zeros_like(leaf)
    if edit.get('colour', True):
        pixels = np.array(rgb, dtype=np.float32)/255
        r, g, b = np.moveaxis(pixels, -1, 0)
        scope = leaf.copy()
        if 'rois' in edit:
            boxes = [[[x1,y1],[x2,y1],[x2,y2],[x1,y2]] for x1,y1,x2,y2 in edit['rois']]
            scope &= polygon_mask(rgb.size, boxes)
        lesion = (r > g*edit.get('ratio', 1.12)) & (r > b*1.08) & (r > 0.08)
        if edit.get('include_pale'):
            maximum, minimum = pixels.max(2), pixels.min(2)
            lesion |= (maximum > .45) & ((maximum-minimum)/np.maximum(maximum, 1e-6) < .25)
        if edit.get('dark_max'):
            lesion |= pixels.max(2) < edit['dark_max']
        lesion &= scope
        lesion = ndi.median_filter(lesion, size=3)
        lesion = components(ndi.binary_fill_holes(lesion), 20) > 0
    lesion |= polygon_mask(rgb.size, edit.get('lesion_polygons', []))
    lesion &= leaf
    result = leaf.astype(np.uint8)
    result[lesion] = 2
    return result


def render(work):
    if (work/'manifest_before.csv').exists():
        raise ValueError('Review already committed; use a new workspace for further edits')
    selection = json.loads((work/'selection.json').read_text())
    edits = json.loads((work/'edits.json').read_text())
    _, split = load_manifest()
    dataset = Path(split['dataset'])
    annotations = ROOT/'artifacts/segmentation_annotations'
    records, tiles = [], {}
    for row in selection:
        edit = edits[str(row['review_id'])]
        if edit.get('exclude'):
            continue
        source = dataset/row['image_path']
        if sha256(source) != row['source_sha256']:
            raise ValueError('Source changed since selection')
        old_path = annotations/row['mask_path']
        with Image.open(source) as im:
            rgb = ImageOps.exif_transpose(im).convert('RGB')
        with Image.open(old_path) as im:
            old = np.array(im)
        corrected = correct(rgb, old, edit)
        target = work/'corrected'/row['mask_path']
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(corrected).save(target)
        record = dict(row, old_mask_sha256=sha256(old_path), corrected_mask_sha256=sha256(target),
                      changed_pixels=int(np.count_nonzero(old != corrected)),
                      leaf_pixels=int(np.count_nonzero(corrected)), affected_pixels=int(np.count_nonzero(corrected == 2)),
                      review_method='ai_visual', review_resolution=512,
                      review_notes=edit['notes'], expert_validated=False)
        records.append(record)
        preview = rgb.resize((512,512))
        mask = np.array(Image.fromarray(corrected).resize((512,512), Image.Resampling.NEAREST))
        pixels = np.array(preview)
        chosen = mask == 2
        pixels[chosen] = (pixels[chosen]*.6 + np.array([255,25,25])*.4).astype(np.uint8)
        edge = (mask > 0) & ~ndi.binary_erosion(mask > 0)
        pixels[edge] = (0,210,255)
        tile = Image.new('RGB',(512,550),'white')
        tile.paste(Image.fromarray(pixels),(0,38))
        ImageDraw.Draw(tile).text((4,4),f"{row['review_id']} {row['split']} | corrected draft | {100*record['affected_pixels']/max(1,record['leaf_pixels']):.1f}%",fill='black')
        tiles.setdefault(row['class_name'],[]).append(tile)
    write_json(work/'correction_records.json', records)
    for name, group in tiles.items():
        for page in range((len(group)+4)//5):
            sheet = Image.new('RGB',(1536,1100),'#ddd')
            for i,tile in enumerate(group[page*5:(page+1)*5]):
                sheet.paste(tile,(i%3*512,i//3*550))
            sheet.save(work/f'{name}_corrected_{page}.jpg',quality=95)
    print(f'Rendered {len(records)} corrected proposals. Review overlays before committing.')


def commit(work):
    records = json.loads((work/'correction_records.json').read_text())
    edits = json.loads((work/'edits.json').read_text())
    annotations = ROOT/'artifacts/segmentation_annotations'
    manifest = annotations/'annotations.csv'
    with manifest.open(newline='',encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames)
        rows = list(reader)
    indexed = {row['image_path']: row for row in rows}
    for field in ('review_method', 'review_record'):
        if field not in fields:
            fields.append(field)
    if (work/'manifest_before.csv').exists():
        raise ValueError('Review already committed; preserve the previous manifest')
    for record in records:
        if edits[str(record['review_id'])].get('decision') != 'accepted':
            raise ValueError('Every committed proposal must have an explicit visual-review decision')
        if sha256(annotations/record['mask_path']) != record['old_mask_sha256']:
            raise ValueError('Original annotation changed since rendering')
        if sha256(work/'corrected'/record['mask_path']) != record['corrected_mask_sha256']:
            raise ValueError('Correction changed since rendering')
        if indexed[record['image_path']]['review_status'] != 'unreviewed':
            raise ValueError('Do not replace an already reviewed annotation')
    shutil.copy2(manifest,work/'manifest_before.csv')
    for record in records:
        destination = annotations/record['mask_path']
        backup = work/'before'/record['mask_path']
        backup.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(destination,backup)
        shutil.copy2(work/'corrected'/record['mask_path'],destination)
        row = indexed[record['image_path']]
        row.update(review_status='reviewed',reviewed_by='Codex AI visual review (not an expert annotation)',
                   review_method='ai_visual',review_record=str((work/'correction_records.json').relative_to(ROOT)))
    temporary = manifest.with_suffix('.tmp.csv')
    with temporary.open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(manifest)
    write_json(work/'summary.json',{'status':'committed','reviewed_masks':len(records),
        'train_images':sum(r['split']=='train' for r in records),
        'validation_images':sum(r['split']=='validation' for r in records),'test_images':0,
        'review_method':'ai_visual','expert_validated':False,'review_resolution':512,
        'note':'Image-specific approximate boundaries reviewed at 512-pixel display resolution. '
               'Not expert ground truth; a balanced pilot subset, not full-dataset review.'})
    print(f'Committed {len(records)} AI-reviewed masks with backups and provenance.')


def report(work):
    """Link native corrected masks to their source and visually inspected overlay."""
    records = json.loads((work/'correction_records.json').read_text())
    edits = json.loads((work/'edits.json').read_text())
    _, split = load_manifest()
    cards = []
    for row in records:
        source = Path(split['dataset'])/row['image_path']
        mask_path = work/'corrected'/row['mask_path']
        with Image.open(source) as im:
            rgb = ImageOps.exif_transpose(im).convert('RGB')
        with Image.open(mask_path) as im:
            labels = np.array(im)
        # Preserve the photo's aspect ratio in the browser.
        scale = min(1, 1024/max(rgb.size))
        size = tuple(max(1, round(d*scale)) for d in rgb.size)
        pixels = np.array(rgb.resize(size))
        mask = np.array(Image.fromarray(labels).resize(size, Image.Resampling.NEAREST))
        selected = mask == 2
        pixels[selected] = (pixels[selected]*.6 + np.array([255,25,25])*.4).astype(np.uint8)
        edge = (mask > 0) & ~ndi.binary_erosion(mask > 0)
        pixels[edge] = (0,210,255)
        overlay = work/'overlays'/f"{row['review_id']}.jpg"
        overlay.parent.mkdir(exist_ok=True)
        Image.fromarray(pixels).save(overlay, quality=95)
        def url(path):
            return quote(Path(os.path.relpath(path, work)).as_posix())
        title = escape(f"{row['review_id']} | {row['split']} | {row['image_path']}")
        cards.append(f'<article><h2>{title}</h2><p>AI visual review; approximate boundaries. '
                     f"Affected reference area: {100*row['affected_pixels']/row['leaf_pixels']:.2f}%</p>"
                     f'<div><a href="{url(source)}"><img loading="lazy" src="{url(source)}" alt="Original photo"></a>'
                     f'<a href="{url(overlay)}"><img loading="lazy" src="{url(overlay)}" alt="Corrected reference overlay"></a></div>'
                     f'<a href="{url(mask_path)}">Native semantic mask</a></article>')
    deferred = ', '.join(key for key, edit in edits.items() if edit.get('exclude'))
    html = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Boundary corrections</title>'
            '<style>body{font:16px system-ui;margin:24px;background:#f3f5f7}article{background:white;padding:16px;margin:20px 0}'
            'h2{font-size:16px;overflow-wrap:anywhere}article div{display:flex}img{width:100%;height:420px;object-fit:contain}'
            'article div a{width:50%}</style><h1>Reviewed boundary corrections</h1>'
            f'<p>{len(records)} corrected semantic masks; deferred review IDs: {deferred}. '
            'Cyan = leaf boundary; red = approximate affected tissue. Sources on the left, corrections on the right.</p>'
            '<p>AI visual review at 512-pixel display resolution, not expert ground truth. '
            'The remaining dataset masks are unreviewed. These references support an experimental U-Net pilot.</p>'
            '<p><a href="summary.json">Review summary</a> | <a href="correction_records.json">Provenance</a> | '
            '<a href="edits.json">Reproducible edits</a></p>' + ''.join(cards) + '</html>')
    (work/'index.html').write_text(html, encoding='utf-8')
    print(f'Boundary correction gallery: {work / "index.html"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['render','commit','report'])
    parser.add_argument('--work',type=Path,default=ROOT/'artifacts/boundary_review_20261010')
    args = parser.parse_args()
    {'render': render, 'commit': commit, 'report': report}[args.action](args.work.resolve())
