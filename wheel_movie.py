import glob
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.lines import Line2D
from tqdm import tqdm


def ax_format_func(value, tick_number):
    # find number of multiples of pi/2
    N = int(np.round(2 * value / np.pi))
    if N == 0:
        return "0"
    elif N == 1:
        return r"$\pi/2$"
    elif N == 2:
        return r"$\pi$"
    elif N % 2 > 0:
        return r"${0}\pi/2$".format(N)
    else:
        return r"${0}\pi$".format(N // 2)


class MakeMovie(object):
    """
    Class for making movies using matplotlib animation and multiprocessing via joblib
    :param output_path: string specifying path where movie will be saved
    :param output_filename: string specifying filename of movie (without extension)

    Child class needs to define the following three functions:
    -setup_figure: defines fig, fps and frames
    -init_func: init function used by matplotlib.animation
    -animate_func: animate function used by matplotlib.animation


    """

    def __init__(
        self,
        output_path,
        output_filename,
        n_chunks=4,
        n_jobs=4,
        delete_chunks=True,
        overwrite_chunks=False,
    ):
        self.output_path = Path(output_path)
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)
        self.output_filename = output_filename

        self.n_chunks = n_chunks
        self.n_jobs = n_jobs
        self.delete_chunks = delete_chunks
        self.overwrite_chunks = overwrite_chunks

        # these vars need to be set by Child class
        self.fig = None
        self.fps = None
        self.frames = None

    def make_movie_parallel(self):
        # nb_stdout = sys.stdout
        # sys.stdout = open('/dev/stdout', 'w')
        t0 = time.time()
        # variable names are confusing, change
        frames_chunks = np.array_split(self.frames, self.n_chunks)
        # make_chunk = partial(
        #     self.make_chunk,
        #     )
        results = []
        with Pool(self.n_jobs) as p:
            # with tqdm(total=len(frames_chunks)) as pbar:
            for result in p.starmap(
                self.make_chunk,
                tqdm(enumerate(frames_chunks), total=len(frames_chunks)),
                chunksize=1,
            ):
                results.append(result)
                # pbar.update()

        # Parallel(n_jobs=self.n_jobs, verbose=0, backend="loky")(
        #     delayed(self.make_chunk)(chunk, frames)
        #     for chunk, frames in enumerate(frames_chunks)
        # )

        self.stitch_chunks()
        t1 = time.time()
        print("total time: " + str(t1 - t0))
        # sys.stdout = nb_stdout

    def make_movie(self):
        # non parallel implementation, use for profiling
        # nb_stdout = sys.stdout
        # sys.stdout = open('/dev/stdout', 'w')
        t0 = time.time()
        self.make_chunk(0, self.frames)
        t1 = time.time()
        print("total time: " + str(t1 - t0))
        # sys.stdout = nb_stdout

    def make_chunk(self, chunk, frames):
        # progress bar printed in terminal, not working as expected
        # text = f"chunk { chunk + 1:05}"
        # self.pbar = tqdm(total=len(frames), position=chunk + 1, desc=text)

        # should check if self.fig,fps, animate_func and init_func are defined (they are all None by default)
        if self.fig is None:
            raise ValueError("self.fig is None, please define self.fig in child class")
        if self.fps is None:
            raise ValueError("self.fps is None, please define self.fps in child class")
        # if self.animate_func is None:
        #     raise ValueError("self.animate_func is None, please define self.animate_func in child class")
        # if self.init_func is None:
        #     raise ValueError("self.init_func is None, please define self.init_func in child class")
        if not self.overwrite_chunks:
            if (
                self.output_path
                / (self.output_filename + f"_chunk_{chunk + 1:05}" + ".mp4")
            ).exists():
                print(f"chunk {chunk + 1:05} already exists, skipping")
                return

        writer = animation.FFMpegWriter(
            fps=self.fps,
            metadata=dict(artist="Me"),
            bitrate=1080,
            extra_args=[
                "-vcodec",
                "libx264",
                # '-preset', 'ultrafast',
            ],
        )

        ani = animation.FuncAnimation(
            self.fig, self.animate_func, frames=frames, cache_frame_data=False
        )

        ani.save(
            self.output_path
            / (self.output_filename + f"_chunk_{chunk + 1:05}" + ".mp4"),
            writer=writer,
        )
        # self.pbar.close()

    def stitch_chunks(self):
        # stitches all *.mp4 files with word chunk in output_path and saves with output_filename
        current_path = os.getcwd()
        os.chdir(self.output_path)

        # creates list of files to stitch
        text_file = open("list.txt", "w")
        files_to_stitch = sorted(glob.glob("*chunk*.mp4"))
        for file in files_to_stitch:
            text_file.write("file '" + file + "'\n")
        text_file.close()
        # creates stitched movie
        os.system(
            "ffmpeg -y -f concat -safe 0 -i list.txt -c copy "
            + self.output_filename
            + ".mp4"
        )
        # deletes chunked files and temporary list.txt
        if self.delete_chunks:
            for file in files_to_stitch:
                os.remove(file)
            os.remove("list.txt")
        os.chdir(current_path)

    def setup_figure(self):
        pass

    def init_func(self):
        pass

    def animate_func(self, i):
        pass


class WheelMovie(MakeMovie):
    def __init__(
        self,
        wheel_data_file,
        movie_file,
        movie_timestamp_file,
        fly_name,
        plot_width,
        output_path,
        output_filename,
        fps=30,
        start_time=None,
        total_frames=None,
        # end_time=None, #broken
        zero_pos_frac=0.5,
        show_nth_frame=1,
        pos_ax_size=10,
        burn_frame_number=True,
        burn_timestamp=False,
        **kwargs,
    ):
        MakeMovie.__init__(self, output_path, output_filename, **kwargs)

        self.plot_width = pd.Timedelta(plot_width)
        self.pos_ax_size = pos_ax_size / 2
        self.zero_pos_frac = zero_pos_frac
        self.fig = plt.figure(constrained_layout=False, figsize=(15, 10), dpi=140)
        self.fps = fps
        self.burn_frame_number = burn_frame_number
        self.burn_timestamp = burn_timestamp

        # load wheel data
        wheel_df = pd.read_csv(wheel_data_file)
        wheel_df.columns = wheel_df.columns.str.strip().str.replace(" ", "_")
        wheel_df = wheel_df.filter(regex="timestamp|absolute_rotation_cam_[01]")
        wheel_df.rename(
            columns={
                "absolute_rotation_cam_0": "yaw",
                "absolute_rotation_cam_1": "roll",
            },
            inplace=True,
        )
        wheel_df["time"] = (
            pd.to_datetime(wheel_df["timestamp"], unit="ns")
            .dt.tz_localize("UTC")
            .dt.tz_convert("US/Eastern")
        )
        wheel_df.set_index("time", inplace=True)
        self.wheel_df = wheel_df

        # load movie and timestamps
        self.movie_file = movie_file
        self.wheel_movie = None
        movie_timestamps = pd.read_csv(movie_timestamp_file)
        movie_timestamps.columns = movie_timestamps.columns.str.strip().str.replace(
            " ", "_"
        )
        movie_timestamps["time"] = (
            pd.to_datetime(movie_timestamps["timestamp"], unit="ns")
            .dt.tz_localize("UTC")
            .dt.tz_convert("US/Eastern")
        )

        movie_timestamps.set_index("time", inplace=True)
        self.movie_timestamps = movie_timestamps
        self.start_time = start_time
        if self.start_time is None:
            self.start_time = self.movie_timestamps.index[0]

        self.exp_start_time = self.movie_timestamps.index[0]
        # self.movie_timestamps = movie_timestamps[:::skip_nth_frame]
        # if end_time is None:
        end_time = self.movie_timestamps.index[-1]
        # print(len(self.movie_timestamps))
        self.movie_timestamps = self.movie_timestamps.loc[
            : end_time - (self.plot_width * self.zero_pos_frac)
        ]
        # print(len(self.movie_timestamps))
        self.frames = self.movie_timestamps.index[::show_nth_frame]
        # print(len(self.frames))
        if total_frames is not None:
            self.frames = self.frames[:total_frames]

        gsh, gsw = [100, 150]
        timeseries_height = 14
        timeseries_space = 1
        n_timeseries = 2
        mid_space = 5
        gs = self.fig.add_gridspec(gsh, gsw)

        timeseries_tops = (
            gsh
            - (np.arange(1, n_timeseries + 1) * timeseries_height)
            - np.arange(n_timeseries) * timeseries_space
        )
        timeseries_bottoms = timeseries_tops + timeseries_height

        self.image_ax = self.fig.add_subplot(gs[0 : timeseries_tops[-1] - 10, :])
        self.image_ax.set_xticks([])
        self.image_ax.set_yticks([])
        self.image_ax.set_title(fly_name)

        # Eye Axes
        self.yaw_ax = self.fig.add_subplot(
            gs[timeseries_tops[0] : timeseries_bottoms[0], :]
        )

        self.roll_ax = self.fig.add_subplot(
            gs[timeseries_tops[1] : timeseries_bottoms[1], :]
        )
        plt.setp(self.roll_ax.get_xticklabels(), visible=False)

        # position timeseries lines.
        self.roll_line = Line2D([0], [0], c="black")
        self.yaw_line = Line2D([0], [0], c="black")

        # set vertical marker lines, and add lines to axes.
        vlinekwargs = {"color": "red", "linestyle": "--"}
        self.vlines = []
        # self.time_ax_list = [self.roll_ax, self.yaw_ax]
        # self.time_line_list = [self.roll_line, self.yaw_line]

        for line, ax, var in zip(
            [self.roll_line, self.yaw_line],
            [self.roll_ax, self.yaw_ax],
            ["roll", "yaw"],
        ):
            ax.add_line(line)
            ax.set_xlim(0, self.plot_width.total_seconds())
            self.vlines.append(ax.axvline(x=0, **vlinekwargs))

            ax.set_ylim(-np.pi, np.pi)
            ax.set_ylabel(f"{var} (rad)")
            ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(ax_format_func))
            despine(ax)
        # self.fig.suptitle(self.time.strftime("%b %d %H:%M:%S"), size="xx-large")
        # self.animate_func(self.start_time)

        if burn_frame_number or burn_timestamp:
            self.burned_text = self.fig.text(0.01, 0.98, "", ha="left", va="top")

    def animate_func(self, start_time):
        if self.wheel_movie is None:
            self.wheel_movie = cv2.VideoCapture(str(self.movie_file))
        self.start_time = start_time
        wheel_chunk = self.wheel_df.loc[
            self.start_time : self.start_time + self.plot_width
        ].copy()
        for var in ["roll", "yaw"]:
            wheel_chunk.loc[wheel_chunk[var].diff().abs() > np.pi, var] = np.nan

        self.current_time = self.start_time + self.plot_width * self.zero_pos_frac

        current_frame = self.get_movie_frame(self.current_time)
        if current_frame is not None:
            self.image_ax.imshow(current_frame)

        x_time = (wheel_chunk.index - self.exp_start_time).total_seconds()
        for line, ax, var in zip(
            [self.roll_line, self.yaw_line],
            [self.roll_ax, self.yaw_ax],
            ["roll", "yaw"],
        ):
            # line.set_data(wheel_chunk.index - self.current_time, wheel_chunk[var])

            line.set_data(x_time, wheel_chunk[var])
            ax.set_xlim(x_time[0], x_time[-1])

        for line in self.vlines:
            line.set_data(
                (self.current_time - self.exp_start_time).total_seconds(), [0, 1]
            )

        burned_text = ""
        if self.burn_frame_number:
            idx = self.movie_timestamps.index.get_loc(
                self.current_time, method="nearest"
            )
            burned_text += (
                f"F: {self.movie_timestamps.iloc[idx].frame_id}"
                f" / {self.movie_timestamps.iloc[-1].frame_id}\n"
                f"{(self.current_time - self.exp_start_time).total_seconds():.3f} s\n"
            )
        if self.burn_timestamp:
            burned_text += f"{self.current_time.strftime('%b %d %H:%M:%S')}"
        if self.burn_frame_number or self.burn_timestamp:
            self.burned_text.set_text(burned_text)
        # self.fig.suptitle(self.start_time.strftime("%b %d %H:%M:%S"), size="xx-large")

        # return (list(self.eye_x_lines.values()) + list(self.eye_pos_points.values()) +
        #         + list(self.eye_pos_points.values()) + list(self.eye_hist_lines.values()) +
        #         [self.forw_line, self.heading_line,
        #         self.traj_pos, self.pos_point, self.pos_heading])

    def get_movie_frame(self, current_time):
        idx = self.movie_timestamps.index.get_loc(current_time, method="nearest")
        frame_id = self.movie_timestamps.iloc[idx].frame_id
        self.wheel_movie.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.wheel_movie.read()
        if ret:
            return frame
        else:
            return None

    def frame_times(self, start_times=pd.DataFrame(), chunk_length="0s"):
        times = []

        start_times = start_times.index
        chunk_length = pd.Timedelta(chunk_length)

        for start_time in start_times:
            time = start_time
            end_time = time + chunk_length

            while time < end_time:
                times.append(time)
                time += self.interval

        return times


# def load_data_make_movie(data_folder):


def despine(ax: plt.Axes, bottom_left: bool = False):
    """Removes spines (frame)from matplotlib Axes object.
    By default only removes the right and top spines, and removes
    all 4 if bottom_left=True

    Args:
        ax (plt.Axes): Axes object to despine
        bottom_left (bool, optional): If True, removes all 4 spines. Defaults to False.
    """
    sides = ["top", "right"]
    if bottom_left:
        sides += ["bottom", "left"]
    for side in sides:
        ax.spines[side].set_visible(False)
