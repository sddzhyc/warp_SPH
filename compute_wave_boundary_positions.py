import argparse
import csv
import math
import os

# ---------------------------------------------------------------------------
# Hardcoded scene configuration — mirrors data/scenes/fish_on_water_new.json
# ("Configuration" block). Edit here if the scene JSON changes.
# ---------------------------------------------------------------------------
TIME_STEP_SIZE = 0.0016                     # "timeStepSize"
NUM_STEPS_PER_RENDER_UPDATE = 1             # "numberOfStepsPerRenderUpdate"
DOMAIN_START = [-2.1, 0.0, -2.1]            # "domainStart"
DOMAIN_END = [2.1, 4.0, 2.1]               # "domainEnd"

WAVE_BOUNDARY_ENABLE = True                 # "waveBoundaryEnable"
WAVE_BOUNDARY_AXIS = "z"                    # "waveBoundaryAxis"
WAVE_BOUNDARY_AMPLITUDE = 0.3               # "waveBoundaryAmplitude"
WAVE_BOUNDARY_FREQUENCY = 1.0               # "waveBoundaryFrequency"
WAVE_BOUNDARY_PHASE = 0.0                   # "waveBoundaryPhase"

DEFAULT_OUTPUT_INTERVAL = 10                # matches build_export_settings() in utils.py
DEFAULT_NUM_FRAMES = 400                    # number of exported frames to compute by default


def resolve_axis_index(axis):
    """Match SimSPH: 'x' -> 0, anything else (e.g. 'z') -> 2."""
    return 0 if str(axis) == "x" else 2


# Match SimSPH: axis 'x' -> 0, anything else (e.g. 'z') -> 2.
AXIS_INDEX = resolve_axis_index(WAVE_BOUNDARY_AXIS)


def compute_board_state(sim_time):
    """Reproduce SimSPH._get_wave_boundary_state for a single board axis.

    Reads the hardcoded module-level configuration directly.
    Returns (lower_axis_pos, upper_axis_pos, velocity, shift, clamped).
    """
    lower = DOMAIN_START[AXIS_INDEX]
    upper = DOMAIN_END[AXIS_INDEX]
    velocity = 0.0
    shift = 0.0
    clamped = False

    if WAVE_BOUNDARY_ENABLE and WAVE_BOUNDARY_AMPLITUDE > 0.0 and WAVE_BOUNDARY_FREQUENCY > 0.0:
        omega = 2.0 * math.pi * WAVE_BOUNDARY_FREQUENCY
        theta = omega * sim_time + WAVE_BOUNDARY_PHASE
        shift = 0.5 * WAVE_BOUNDARY_AMPLITUDE * (1.0 - math.cos(theta))
        max_shift = 0.2 * float(DOMAIN_END[AXIS_INDEX] - DOMAIN_START[AXIS_INDEX])
        if shift > max_shift:
            shift = max_shift
            clamped = True
        lower = float(DOMAIN_START[AXIS_INDEX] + shift)
        upper = float(DOMAIN_END[AXIS_INDEX] + shift)
        # NOTE: SimSPH does NOT clamp the velocity when shift is clamped.
        velocity = 0.5 * WAVE_BOUNDARY_AMPLITUDE * omega * math.sin(theta)

    return lower, upper, velocity, shift, clamped


def main():
    parser = argparse.ArgumentParser(
        description="Compute wave-boundary (挡板) positions per exported frame "
        "using the hardcoded fish_on_water_new configuration."
    )
    parser.add_argument("--output_interval", type=int, default=DEFAULT_OUTPUT_INTERVAL,
                        help="Steps between exported frames (default 10, matches export pipeline).")
    parser.add_argument("--num_frames", type=int, default=DEFAULT_NUM_FRAMES,
                        help="Number of exported frames to compute (default 400).")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional path to write the per-frame table as CSV.")
    args = parser.parse_args()

    time_step_size = float(TIME_STEP_SIZE)

    output_interval = max(1, int(args.output_interval))
    num_frames = max(1, int(args.num_frames))

    axis_idx = AXIS_INDEX
    axis_name = {0: "x", 1: "y", 2: "z"}.get(axis_idx, str(axis_idx))

    # Period / frequency summary
    freq = float(WAVE_BOUNDARY_FREQUENCY)
    period_s = (1.0 / freq) if freq > 0.0 else float("inf")
    period_steps = (period_s / time_step_size) if math.isfinite(period_s) else float("inf")
    pool_size_axis = DOMAIN_END[axis_idx] - DOMAIN_START[axis_idx]

    last_step = (num_frames - 1) * output_interval
    total_time = last_step * time_step_size

    print("=" * 78)
    print("Config            : hardcoded (fish_on_water_new.json)")
    print(f"timeStepSize (dt) : {time_step_size} s  (sim_time(step) = step * dt)")
    print(f"Wave enable       : {WAVE_BOUNDARY_ENABLE}")
    print(f"Wave axis         : '{WAVE_BOUNDARY_AXIS}' -> index {axis_idx} ({axis_name})")
    print(f"Amplitude A       : {WAVE_BOUNDARY_AMPLITUDE}")
    print(f"Frequency f       : {freq} Hz")
    print(f"Phase phi         : {WAVE_BOUNDARY_PHASE} rad")
    if math.isfinite(period_s):
        print(f"Period T          : {period_s:.6f} s  =  {period_steps:.3f} sim steps "
              f"=  {period_steps / output_interval:.3f} export frames")
    print(f"Pool size ({axis_name}-axis): {pool_size_axis:.6f}  (constant: both boards share the same shift)")
    print(f"Output interval   : {output_interval} steps/frame")
    print(f"Exported frames   : {num_frames}  (last sim step {last_step}, t = {total_time:.4f} s)")
    print("=" * 78)

    header = ["frame", "sim_step", "sim_time_s",
              f"lower_{axis_name}", f"upper_{axis_name}",
              "velocity"]
    print("{:>5}  {:>8}  {:>10}  {:>12}  {:>12}  {:>11}".format(*header))

    rows = []
    for frame in range(num_frames):
        step = frame * output_interval
        sim_time = step * time_step_size
        lower, upper, velocity, _shift, _clamped = compute_board_state(sim_time)
        rows.append([frame, step, sim_time, lower, upper, velocity])
        print("{:>5}  {:>8}  {:>10.5f}  {:>12.6f}  {:>12.6f}  {:>11.6f}".format(
            frame, step, sim_time, lower, upper, velocity))

    if args.csv:
        csv_dir = os.path.dirname(args.csv)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"\nCSV written to: {args.csv}")


if __name__ == "__main__":
    main()
