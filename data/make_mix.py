from pathlib import Path
import soundfile as sf
import numpy as np
from tqdm import tqdm

def mix_track(track_dir: Path, out_dir, out_name="mix.wav"):
    stems = []
    sr0 = None
    for wav in sorted(track_dir.glob("**/*.wav")):
        x, sr = sf.read(wav)
        if sr0 is None:
            sr0 = sr
        assert sr == sr0, f"SR mismatch in {track_dir}"
        if x.ndim == 2:  # stereo to mono if needed
            x = x.mean(axis=1)
        stems.append(x)

    if not stems:
        return

    # pad to same length
    L = max(len(s) for s in stems)
    stems = [np.pad(s, (0, L - len(s))) for s in stems]

    mix = np.sum(stems, axis=0)

    # simple peak normalization to avoid clipping
    peak = np.max(np.abs(mix)) + 1e-9
    mix = mix / peak * 0.95

    sf.write(out_dir / out_name, mix, sr0)

def mix_track_simple(track_dir: Path, out_dir, out_name="mix_simple.wav"):
    """Create a mix by strictly summing stems (no padding or normalization)."""
    stems = []
    sr0 = None
    for wav in sorted(track_dir.glob("**/*.wav")):
        x, sr = sf.read(wav)
        if sr0 is None:
            sr0 = sr
        assert sr == sr0, f"SR mismatch in {track_dir}"
        if x.ndim == 2:  # stereo to mono if needed
            x = x.mean(axis=1)
        stems.append(x)

    if not stems:
        return

    # Expect all stems to be aligned already (as in MSDM).
    length0 = len(stems[0])
    for s in stems[1:]:
        assert len(s) == length0, f"Length mismatch in {track_dir}"

    mix = np.sum(np.stack(stems, axis=0), axis=0)
    sf.write(out_dir / out_name, mix, sr0)

def main(root, out_dir):
    root = Path(root)
    # track_dirs = [p for p in root.glob("*") if p.is_dir()]
    track_dirs = [root]
    for td in tqdm(track_dirs):
        mix_track_simple(td, out_dir)

if __name__ == "__main__":
    # import argparse
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--root", required=True, help="dir containing track folders")
    # args = ap.parse_args()
    track_name = 'Track00001'
    track_name = 'Track00082'
    root = f'/private/schwartz-lab/omriker/multi-source-diffusion-models/data/slakh2100/train/{track_name}'
    out_dir = f'/private/schwartz-lab/omriker/data/music_datasets/msdm_datasets/mix_22050/{track_name}'
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    main(root, out_dir)
