import json, sys, argparse

sys.stdout.reconfigure(encoding='utf-8')

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=30)
parser.add_argument('--out', type=str, default='scratch/cells_dump.txt')
args = parser.parse_args()

nb = open('notebooks/Breast_cancer_subtype_Transcriptomics_Pipeline.ipynb', encoding='utf-8').read()
cells = json.loads(nb)['cells']

with open(args.out, 'w', encoding='utf-8') as f:
    for i in range(args.start, min(args.end, len(cells))):
        c = cells[i]
        src = ''.join(c['source'])
        f.write(f'\n===== CELL [{i}] ({c["cell_type"]}) =====\n')
        f.write(src[:4000])
        f.write('\n')
        if c['cell_type'] == 'code' and c.get('outputs'):
            out = c['outputs']
            f.write(f'  --- OUTPUTS ({len(out)}) ---\n')
            for o in out[:5]:
                if o.get('text'):
                    f.write('  OUT: ' + ''.join(o['text'])[:800] + '\n')
                if o.get('data', {}).get('text/plain'):
                    f.write('  PLN: ' + str(o['data']['text/plain'])[:400] + '\n')
                if o.get('traceback'):
                    f.write('  ERR: ' + str(o['traceback'])[:400] + '\n')

print(f'Written cells {args.start}-{args.end} to {args.out}')
