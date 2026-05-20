# encode_block: batch-encode a directory of GLB files → block PBR NPZ dataset.
#
# Usage:
#   python -m gpainter.dataset_toolkits.encode_block \
#       --input_dir /data/glbs --output_dir /data/npz \
#       [--workers 1] [--batch_blocks 32] [--device cuda] [--preserve_source]
import argparse
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

from .mesh2pbrblock import mesh2pbrblock


def _encode_one(glb_path, output_dir, device, batch_blocks, preserve_source):
    name = os.path.splitext(os.path.basename(glb_path))[0]
    npz_path = os.path.join(output_dir, f'{name}.npz')
    mesh2pbrblock(
        glb_path,
        npz_path,
        device=device,
        batch_blocks=batch_blocks,
        preserve_source=preserve_source,
        verbose=False,
    )
    size_kb = os.path.getsize(npz_path) / 1024
    return name, npz_path, size_kb


def encode_dir(input_dir, output_dir, device='cuda', batch_blocks=32, workers=1,
               skip_existing=True, preserve_source=False):
    glbs = sorted(glob(os.path.join(input_dir, '**', '*.glb'), recursive=True) +
                  glob(os.path.join(input_dir, '**', '*.GLB'), recursive=True))
    if not glbs:
        print(f'No GLB files found in {input_dir}')
        return

    os.makedirs(output_dir, exist_ok=True)

    if skip_existing:
        pending = []
        for g in glbs:
            name = os.path.splitext(os.path.basename(g))[0]
            if not os.path.exists(os.path.join(output_dir, f'{name}.npz')):
                pending.append(g)
        skipped = len(glbs) - len(pending)
        if skipped:
            print(f'Skipping {skipped} already-encoded files.')
        glbs = pending

    print(
        f'Encoding {len(glbs)} GLBs → {output_dir}  '
        f'(workers={workers}, preserve_source={preserve_source})'
    )

    ok, fail = 0, 0

    def _job(g):
        return _encode_one(g, output_dir, device, batch_blocks, preserve_source)

    if workers == 1:
        for i, g in enumerate(glbs):
            try:
                name, npz_path, size_kb = _job(g)
                ok += 1
                print(f'[{i+1}/{len(glbs)}] {name} → {size_kb:.0f} KB')
            except Exception as e:
                fail += 1
                print(f'[{i+1}/{len(glbs)}] FAIL {os.path.basename(g)}: {e}')
                traceback.print_exc()
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_job, g): g for g in glbs}
            for i, fut in enumerate(as_completed(futs)):
                g = futs[fut]
                try:
                    name, npz_path, size_kb = fut.result()
                    ok += 1
                    print(f'[{i+1}/{len(glbs)}] {name} → {size_kb:.0f} KB')
                except Exception as e:
                    fail += 1
                    print(f'[{i+1}/{len(glbs)}] FAIL {os.path.basename(g)}: {e}')

    print(f'\nDone: {ok} succeeded, {fail} failed.')
    return ok, fail


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch-encode GLB → block PBR NPZ')
    parser.add_argument('--input_dir', required=True, help='Directory containing GLB files')
    parser.add_argument('--output_dir', required=True, help='Output directory for NPZ files')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_blocks', type=int, default=32)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--no_skip', action='store_true', help='Re-encode existing NPZ files')
    parser.add_argument('--preserve_source', action='store_true',
                        help='Store source GLB bytes for byte-exact decode; increases NPZ size.')
    args = parser.parse_args()

    encode_dir(
        args.input_dir,
        args.output_dir,
        device=args.device,
        batch_blocks=args.batch_blocks,
        workers=args.workers,
        skip_existing=not args.no_skip,
        preserve_source=args.preserve_source,
    )
