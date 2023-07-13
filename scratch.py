def load_fictrac_data(filename):
    """Read and minimally process fictrac data.

    Args:
        filename (pathlike): filename to processe.

    Returns:
        pd.DataFrame: fictrac dataframe.
    """
    fictrac_data = pd.read_csv(filename)
    ros_rename_dict = {
        "timestamp": "unix_time",
        "frame_id": "frame",
        "frame_counter": "frame_count",
        "delta_rotation_cam_1": "delta_cam_y",
        "delta_rotation_cam_2": "delta_cam_z",
        "delta_rotation_cam_0": "delta_cam_x",
        "delta_rotation_error": "error",
        "delta_rotation_lab_0": "delta_lab_x",
        "delta_rotation_lab_1": "delta_lab_y",
        "delta_rotation_lab_2": "delta_lab_z",
        "absolute_rotation_cam_0": "cam_x",
        "absolute_rotation_cam_1": "cam_y",
        "absolute_rotation_cam_2": "cam_z",
        "absolute_rotation_lab_0": "lab_x",
        "absolute_rotation_lab_1": "lab_y",
        "absolute_rotation_lab_2": "lab_z",
        "integrated_position_lab_0": "pos_x",
        "integrated_position_lab_1": "pos_y",
        "integrated_heading_lab": "heading",
        "animal_movement_direction_lab": "direction",
        "animal_movement_speed": "speed",
        "integrated_motion_0": "integrated_forward",
        "integrated_motion_1": "integrated_sideways",
        "sequence_counter": "sequence_num",
    }
    fictrac_data = fictrac_data.rename(columns=ros_rename_dict)
    fictrac_data["time"] = (
        pd.to_datetime(fictrac_data["unix_time"], unit="ns")
        .dt.tz_localize("UTC")
        .dt.tz_convert("US/Eastern")
    )
    fictrac_data.heading = ((fictrac_data.heading + np.pi) % (2 * np.pi)) - np.pi
    return fictrac_data


def read_frames(video_file, frame_indices=None, start_frame=0, n_frames=10):
    """Read frames from a video file.

        Must either give frame_indices or start_frame and n_frames.
    Args:
        video_file (pathlike): file to read from
        frame_indices (arraylike, optional): frame indicies to read . Defaults to None.
        start_frame (int, optional): start frame to read from. Defaults to 0.
        n_frames (int, optional): number of frames to read. Defaults to 10.

    Returns:
        List: list of frames
    """
    if frame_indices is None:
        frame_indices = range(start_frame, start_frame + n_frames)
    cap = cv2.VideoCapture(str(video_file))

    frames = []
    for frame_i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    return frames


def combine_frames(
    main_frame,
    sub_frame,
    sub_frame_crop=None,
    sub_frame_center=None,
    sub_frame_width=None,
):
    """Combine two frames, picture in picture style.

    Args:
        main_frame (np.Array): Main / outer frame
        sub_frame (np.Array): Inner frame to be placed in main frame
        sub_frame_crop (slice, optional): Tuple of slices to crop sub frame to.
            Defaults to None.
        sub_frame_center (tuple, optional): x, y position of sub frame center.
            Defaults to None -- center of main frame.
        sub_frame_width (int, optional): width of final subframe in pixels.
            Defaults to None -- 1/3 of main frame width.

    Returns:
        np.Array: combined frame
    """
    main_frame = main_frame.copy()
    sub_frame = sub_frame.copy()
    if sub_frame_crop is not None:
        sub_frame = sub_frame[sub_frame_crop]
    if sub_frame_center is None:
        sub_frame_center = (main_frame.shape[1] // 2, main_frame.shape[0] // 2)
    if sub_frame_width is None:
        sub_frame_width = main_frame.shape[1] // 3

    sub_frame_height = int(sub_frame.shape[0] * sub_frame_width / sub_frame.shape[1])
    sub_frame = cv2.resize(sub_frame, (sub_frame_width, sub_frame_height))

    sub_rows = slice(
        int(sub_frame_center[1] - sub_frame_height / 2),
        int(sub_frame_center[1] + sub_frame_height / 2),
    )
    sub_cols = slice(
        int(sub_frame_center[0] - sub_frame_width / 2),
        int(sub_frame_center[0] + sub_frame_width / 2),
    )

    main_frame[sub_rows, sub_cols] = sub_frame
    return main_frame


def animate_bar(
    headings,
    filename,
    bar_width=25,
    screen_width=880,
    screen_height=100,
    out_fps=30,
    extension=".mp4",
):
    """Animate a bar on a cone, rotating with given headings.

    Args:
        headings (arraylike): heading angles for bar. should be in radians, and (-pi, pi)
        filename (str): filename to save animation to
        bar_width (int, optional): width of bar in pixels. Defaults to 25.
        screen_width (int, optional): number of pixels to plot around screen.
            Defaults to 880.
        screen_height (int, optional): screen height in pixels . Defaults to 100.
        out_fps (int, optional): output animation FPS. Defaults to 30.
        extension (str, optional): extension of output file. must be ".gif", or ".mp4"
            Defaults to ".mp4".

    Raises:
        Exception: _description_
    """
    # set animation parameters
    out_duration = len(headings) / out_fps

    pattern_list = bar_pattern_list(
        bar_width=bar_width,
        screen_height=screen_height,
        screen_width=screen_width,
        angle_list=headings,
    )

    # create an ArenaAnimation object with animation parameters
    arena_animation = ConeArenaAnimation(
        pattern_list,
        fps=out_fps,
        elevation=55,
        camera_distance=7.5,
        # projector_ratio=0.9,
        # height=2,
        background_color=(0.3, 0.3, 0.3),
    )

    # Create a moviepy VideoClip object
    animation = mpy.VideoClip(arena_animation.make_frame, duration=out_duration)

    # Create a video with your specified extension
    if extension == ".mp4":
        animation.write_videofile(filename + ".mp4", fps=arena_animation.fps)
    elif extension == ".gif":
        animation.write_gif(
            filename + ".gif", fps=arena_animation.fps, program="ffmpeg"
        )
    else:
        raise Exception("extension not recognized, set to '.gif' or '.mp4'")


def combine_and_save(
    fly_frames,
    bar_frames,
    fly_crop,
    video_name,
    sub_frame_width=330,
    sub_frame_center=(630, 630),
    out_fps=30,
):
    """Combine two sets of frames, and save to video.

    Args:
        fly_frames (List): frames to be placed in main frame
        bar_frames (List): Main frames to place fly frames in
        fly_crop (_type_): crop to apply to fly frames
        video_name (_type_): output flie name
        sub_frame_width (int, optional): final width of fly frame in px . Defaults to 330.
        sub_frame_center (tuple, optional): center position of fly frame in
            main frame. Defaults to (630, 630).
        out_fps (int, optional): output FPS of video. Defaults to 30.
    """
    combined_frames = []
    for fly_frame, bar_frame in zip(fly_frames, bar_frames):
        combined = combine_frames(
            bar_frame,
            fly_frame,
            sub_frame_crop=fly_crop,
            sub_frame_width=sub_frame_width,
            sub_frame_center=sub_frame_center,
        )
        combined_frames.append(combined)

    height, width, layers = combined.shape

    fourcc = cv2.VideoWriter_fourcc(*"X264")

    video = cv2.VideoWriter(video_name, fourcc, out_fps, (width, height))

    for image in combined_frames:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()
