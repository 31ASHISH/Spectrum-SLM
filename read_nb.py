import json, sys, re

sys.stdout.reconfigure(encoding='utf-8')

with open('new-oelp-slm.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

def clean(t):
    # Remove unicode special chars for safe printing
    return re.sub(r'[^\x00-\x7F]+', '?', t)

for idx in [9, 14, 15]:
    cell = nb['cells'][idx]
    print(f'\n{"="*60}')
    print(f'CELL {idx} FULL OUTPUT')
    print('='*60)
    for o in cell.get('outputs', []):
        otype = o.get('output_type','')
        if otype == 'stream':
            t = o.get('text','')
            if isinstance(t, list): t = ''.join(t)
            print(clean(t))
        elif otype == 'error':
            print('ERROR:', o.get('ename',''), o.get('evalue',''))
