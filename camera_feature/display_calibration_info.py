import pyrealsense2 as rs

def run():
    pipeline = rs.pipeline()
    pipeline.start()

    # Get the active profile and depth stream profile
    profile = pipeline.get_active_profile()
    depth_stream = profile.get_stream(rs.stream.depth)
    depth_video_profile = rs.video_stream_profile(depth_stream)

    # Get intrinsics object
    intrinsics = depth_video_profile.get_intrinsics()

    fx = intrinsics.fx  # focal length x
    fy = intrinsics.fy  # focal length y
    cx = intrinsics.ppx # principal point x
    cy = intrinsics.ppy # principal point y
    width = intrinsics.width
    height = intrinsics.height
    distortion = intrinsics.coeffs # distortion coefficients

    print("INTRINSIC PARAMETER of Depth Camera")
    print(f"fx={fx}\nfy={fy}\ncx={cx}\ncy={cy}\nwidth={width}\nheight={height}")
    print(f"Distortion coefficients: {distortion}")

if __name__ == "__main__":
    run()