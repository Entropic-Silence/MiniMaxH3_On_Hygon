"""Check decoded media against the fixed POD validation profile."""
import argparse
import json
from pathlib import Path

import av

def inspect(path, profile):
    errors = []
    with av.open(str(path)) as container:
        assert len(container.streams.video) == 1, 'Expected exactly one video stream'
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0
        width, height = stream.width, stream.height
        frames = 0
        times = []
        for frame in container.decode(stream):
            frames += 1
            if (frame.width, frame.height) != (profile['width'], profile['height']):
                errors.append(f'Frame {frames} size {frame.width}x{frame.height}')
            if frame.pts is not None:
                times.append(float(frame.pts * frame.time_base))
        duration = times[-1] - times[0] + 1 / fps if times and fps else None
    with av.open(str(path)) as container:
        audio = list(container.streams.audio)
        audio_samples = sum(frame.samples for frame in container.decode(audio[0])) if audio else 0
    if (width, height) != (profile['width'], profile['height']): errors.append('Video dimensions differ')
    if frames != profile['frames']: errors.append(f'Frame count {frames}')
    if abs(fps - profile['fps']) > 0.001: errors.append(f'Frame rate {fps}')
    if duration is None or abs(duration - profile['duration_seconds']) > 1 / profile['fps']:
        errors.append(f'Video duration {duration}')
    if times and any(abs((b-a)-1/profile['fps']) > 0.002 for a,b in zip(times,times[1:])):
        errors.append('Frame timestamps differ from the expected cadence')
    if profile['audio_required'] and not audio_samples: errors.append('No decodable audio samples')
    return dict(file=str(path),width=width,height=height,frames=frames,fps=fps,
                duration_seconds=duration,audio_samples=audio_samples,passed=not errors,errors=errors[:10])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    profile = json.loads((Path(__file__).resolve().parent/'configs/pod-validation.json').read_text())
    result = inspect(args.video,profile)
    payload = json.dumps(result,indent=2)+'\n'
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(payload)
    raise SystemExit(0 if result['passed'] else 1)
