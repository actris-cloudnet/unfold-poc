import argparse
import datetime
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import requests
from numpy import ma
from rpgpy import read_rpg
from rpgpy.spcutil import radar_moment_calculation
from tqdm import tqdm


@dataclass
class Profile:
    vels: list[npt.NDArray]
    alts: list[npt.NDArray]
    data: list[npt.NDArray]


def read_profiles(filename: PathLike | str) -> list[Profile]:
    header, data = read_rpg(filename)
    profiles = []
    for time_ind, t in enumerate(data["Time"]):
        spec = []
        alts = []
        vels = []
        for sequ in range(header["SequN"]):
            vel = header["velocity_vectors"][sequ, :]
            sgn = np.where(vel != 0)[0]
            ind = slice(sgn[0], sgn[-1] + 1)
            sequ_ind = slice(
                header["RngOffs"][sequ],
                header["RngOffs"][sequ + 1] if sequ < header["SequN"] - 1 else None,
            )
            s = data["TotSpec"][time_ind, sequ_ind, ind]
            s = filter_hot(s)
            spec.append(s)
            alts.append(header["RAlts"][sequ_ind])
            vels.append(vel[ind])
        profile = Profile(vels, alts, spec)
        profiles.append(profile)
    return profiles


def calc_mean_vel(p: Profile):
    pd = []
    for vel, spec in zip(p.vels, p.data):
        for rng in spec:
            pd.append(radar_moment_calculation(rng, vel)[1])
    return np.array(pd)


def filter_hot(data: npt.NDArray) -> npt.NDArray:
    mask = data > 0
    above = np.vstack((np.zeros_like(mask[0]), mask[:-1]))
    below = np.vstack((mask[1:], np.zeros_like(mask[0])))
    is_hot = mask & ~above & ~below
    return np.where(is_hot, 0, data)


def plot_all(data: list[Profile], ax: Axes):
    mean_vel = np.array([calc_mean_vel(p) for p in data])
    ax.pcolormesh(
        np.arange(len(data)),
        np.concat(data[0].alts),
        mean_vel.T,
        cmap="RdBu_r",
        vmin=-10,
        vmax=10,
    )
    # ax.clim(-10, 10)


def plot_profile(p: Profile, ax: Axes):
    max_vel = np.max([np.max(v) for v in p.vels])
    min_vel = np.min([np.min(v) for v in p.vels])
    for vel, alt, spec in zip(p.vels, p.alts, p.data):
        spec = ma.masked_where(spec == 0, spec)
        spec = 10 * ma.log10(spec)
        ax.pcolormesh(vel, alt, spec)
        ax.fill_between([min_vel, vel[0]], alt[0], alt[-1], color="silver", alpha=0.5)
        ax.fill_between([vel[-1], max_vel], alt[0], alt[-1], color="silver", alpha=0.5)
    # ax.plot(calc_mean_vel(p), np.concat(p.alts), "r")


def dealias_by_mean(p: Profile) -> list[npt.NDArray[np.int32]]:
    offsets = []
    means = []
    for i, spec in enumerate(p.data):
        n_alt, n_bins = spec.shape
        tau = 2 * np.pi
        a = np.linspace(0, tau, n_bins, endpoint=False)
        b = np.sum(spec * np.sin(a), axis=1)
        c = np.sum(spec * np.cos(a), axis=1)
        mean = (np.atan2(b, c) % tau) / tau * n_bins
        is_range_empty = np.all(spec == 0, axis=1)
        mean[is_range_empty] = 0
        offset = np.round(mean).astype(np.int32) - n_bins // 2
        diff = np.diff(offset, prepend=offset[0])
        jump_offset = np.zeros_like(offset)
        threshold = 0.75 * n_bins
        jump_offset[diff > threshold] = -n_bins
        jump_offset[diff < -threshold] = n_bins
        jump_offset = np.cumsum(jump_offset)
        mean += jump_offset
        offset += jump_offset
        # plt.figure()
        # plt.title(f"sequ {i}")
        # plt.pcolor(10*np.log10(spec))
        # plt.plot(mean, np.arange(n_alt))
        # plt.plot(offset, np.arange(n_alt))
        # plt.plot(offset+n_bins, np.arange(n_alt))
        offsets.append(offset)
        means.append(mean)
    for i in range(1, len(offsets)):
        last_n_bins = p.data[i - 1].shape[1]
        curr_n_bins = p.data[i].shape[1]
        last_vel_res = p.vels[i - 1][-2] - p.vels[i - 1][-1]
        curr_vel_res = p.vels[i][-2] - p.vels[i][-1]
        if means[i - 1][-1] == 0 or means[i][0] == 0:
            continue
        last_mean = (means[i - 1][-1] - last_n_bins / 2) * last_vel_res
        curr_mean = (means[i][0] - curr_n_bins / 2) * curr_vel_res
        # print(f"last mean {last_mean} ({p.alts[i-1][-1]}m), curr mean {curr_mean} ({p.alts[i][0]}m)")
        cands = np.array(
            [
                -3 * curr_n_bins,
                -2 * curr_n_bins,
                -curr_n_bins,
                0,
                curr_n_bins,
                2 * curr_n_bins,
                3 * curr_n_bins,
            ]
        )
        offset = cands[np.argmin(np.abs(curr_mean + cands * curr_vel_res - last_mean))]
        # print(f"shift sequ {i} by {offset}")
        offsets[i] += offset
        means[i] += offset
    return offsets


def shift_profiles(p: Profile, offsets: list[npt.NDArray[np.int32]]):
    new_vels = []
    new_data = []
    for vel, alt, spec, off in zip(p.vels, p.alts, p.data, offsets):
        n_bins = spec.shape[1]
        n_alt = len(alt)
        min_ind = np.min(off)
        max_ind = np.max(off) + n_bins
        new_n_bins = max_ind - min_ind
        new_spec = np.zeros((n_alt, new_n_bins))
        for i in range(n_alt):
            x1 = -min_ind + off[i]
            x2 = -min_ind + off[i] + n_bins
            new_spec[i, x1:x2] = np.roll(spec[i, :], -off[i])
        vel_res = vel[-2] - vel[-1]
        min_vel = (n_bins // 2 - min_ind) * vel_res
        max_vel = (n_bins // 2 - max_ind) * vel_res
        new_vel = np.linspace(min_vel, max_vel, new_n_bins)
        new_vels.append(new_vel)
        new_data.append(new_spec)
        # plt.figure()
        # plt.pcolor(new_vel, alt, ma.masked_where(new_spec == 0, new_spec))
    return Profile(new_vels, p.alts, new_data)


RAW_DIR = Path("data")
RAW_DIR.mkdir(exist_ok=True)


def download_data(site: str, date: datetime.date, hour: int) -> Path:
    prefix = f"{date:%y%m%d}_{hour:02}00"
    suffix = "_ZEN.LV0"
    params = {
        "site": site,
        "instrument": "rpg-fmcw-94",
        "date": date.isoformat(),
        "filenamePrefix": prefix,
        "filenameSuffix": suffix,
    }
    res = requests.get("https://cloudnet.fmi.fi/api/raw-files", params)
    res.raise_for_status()
    metadata = res.json()
    if len(metadata) == 0:
        raise ValueError("No matching file found")
    elif len(metadata) != 1:
        raise ValueError("Multiple matching files found")
    metadata = metadata[0]
    short_pid = metadata["instrumentPid"].removeprefix(
        "https://hdl.handle.net/21.12132/3."
    )[:8]
    out_dir = RAW_DIR / (metadata["siteId"] + "-" + short_pid)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / metadata["filename"]
    if out_path.exists():
        print("Already downloaded!")
        return out_path
    res = requests.get(metadata["downloadUrl"], stream=True)
    res.raise_for_status()
    total_bytes = int(res.headers["content-length"])
    with (
        open(out_path, "wb") as out_file,
        tqdm(
            desc=metadata["filename"],
            total=total_bytes,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in res.iter_content(chunk_size=1024):
            size = out_file.write(data)
            bar.update(size)
    return out_path


def valid_hour(s: str) -> int:
    try:
        hour = int(s)
        if 0 <= hour < 24:
            return hour
        raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid hour: '{s}'. Must be an integer between 0 and 23."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--date",
        type=datetime.date.fromisoformat,
        help="Date in YYYY-MM-DD format",
    )
    parser.add_argument("-H", "--hour", type=valid_hour, help="Hour (0-23)")
    parser.add_argument("-s", "--site", type=str, help="Site name")
    parser.add_argument(
        "-p",
        "--profiles",
        type=int,
        nargs="+",
        default=[],
        help="Plot profiles by index",
    )
    args = parser.parse_args()
    print("Download file...")
    filename = download_data(args.site, args.date, args.hour)
    print("Reading file...")
    profiles = read_profiles(filename)

    if args.profiles:
        profiles = [profiles[i] for i in args.profiles]
        tindex = args.profiles
    else:
        tindex = list(range(len(profiles)))

    print("Dealiasing...")
    dealiased = []
    for i, p in zip(tindex, tqdm(profiles)):
        offsets = dealias_by_mean(p)
        pda = shift_profiles(p, offsets)
        if args.profiles:
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 10))
            fig.suptitle(f"spectra t={i}")
            ax1.set_title("original")
            ax1.set_xlabel("Velocity (m s$^{-1}$)")
            ax1.set_ylabel("Range (m)")
            plot_profile(p, ax1)
            ax2.set_title("dealiased")
            ax2.set_xlabel("Velocity (m s$^{-1}$)")
            plot_profile(pda, ax2)
        else:
            dealiased.append(pda)

    if dealiased:
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True, figsize=(10, 10))
        fig.suptitle("mean velocity")
        ax1.set_title("original")
        ax1.set_xlabel("Time index")
        ax1.set_ylabel("Range (m)")
        plot_all(profiles, ax1)
        ax2.set_title("dealiased")
        ax2.set_ylabel("Range (m)")
        plot_all(dealiased, ax2)

    plt.show()


if __name__ == "__main__":
    main()
