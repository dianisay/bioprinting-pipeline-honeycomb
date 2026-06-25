import json, sys
nb = json.load(open(sys.argv[1], encoding='utf-8'))
print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    src = c['source']
    first = ''.join(src[:1]).strip()[:80] if src else '(empty)'
    print(f"  [{i}] {c['cell_type']}: {first}")
