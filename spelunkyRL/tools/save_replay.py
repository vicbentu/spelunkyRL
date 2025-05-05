# import os
# from gymnasium.wrappers import RecordVideo
# from stable_baselines3.common.vec_env import VecVideoRecorder   # <- new
# import numpy as np

# def save_replay(
#     env, # can be a VecEnv or a plain gym.Env
#     predictor, # a callable with two functionalities: reset() and predict(obs)
#     out_path: str,
#     max_steps: int = 1000,
#     fps: int = 60,
# ):
#     predictor.reset()

#     # try:
#     #     old_speedup = env.speedup
#     #     env.speedup = False
#     #     set_speedup = lambda v: setattr(env, "speedup", v)

#     #     old_frames_per_step = env.frames_per_step
#     #     env.frames_per_step = 1
#     #     set_frames_per_step = lambda v: setattr(env, "frames_per_step", v)
#     # except AttributeError:
#     old_speedup = env.get_attr("speedup")[0]
#     env.set_attr("speedup", False)
#     set_speedup = lambda v: env.set_attr("speedup", v)
#     old_frames_per_step = env.get_attr("frames_per_step")[0]
#     env.set_attr("frames_per_step", 1)
#     set_frames_per_step = lambda v: env.set_attr("frames_per_step", v)
#     max_steps = max_steps * old_frames_per_step

#     folder, prefix = os.path.split(out_path)
#     folder = folder or "./"
#     prefix = os.path.splitext(prefix)[0]

#     wrapped = VecVideoRecorder(
#         env,
#         video_folder=folder,
#         record_video_trigger=lambda step: step == 0,
#         name_prefix=prefix,
#         video_length=max_steps,
#     )

#     if fps is not None:
#         wrapped.metadata["render_fps"] = fps

#     reset_out = wrapped.reset()
#     if isinstance(reset_out, tuple):   # Gymnasium ≥0.26  (obs, info)
#         obs, _ = reset_out
#     else:                              # Stable‑Baselines3 VecEnv (obs only)
#         obs = reset_out

#     repeat = max(int(old_frames_per_step), 1)
#     action = predictor.predict(obs)
#     repeat_left = repeat

#     steps = 0
#     while steps < max_steps:
#         step_result = wrapped.step(action)
#         if len(step_result) == 5:            # Gymnasium
#             obs, reward, terminated, truncated, _ = step_result
#             done = terminated or truncated
#         else:                                # SB3 VecEnv
#             obs, reward, done, _ = step_result

#         steps      += 1
#         repeat_left -= 1

#         if done or steps >= max_steps:
#             break

#         if repeat_left == 0:
#             action       = predictor.predict(obs)
#             repeat_left  = repeat

#     if isinstance(wrapped, VecVideoRecorder) and wrapped.recording:
#         video_path = wrapped.video_path
#         wrapped._stop_recording()

#     if os.path.exists(video_path) and video_path != out_path:
#         os.replace(video_path, out_path)

#     print(f"Replay saved ➜  {out_path}")

#     set_speedup(old_speedup)
#     set_frames_per_step(old_frames_per_step)



import os
from gymnasium.wrappers import RecordVideo

def save_replay(
    env, # a plain gym.Env
    predictor, # a callable with two functionalities: reset() and predict(obs)
    out_path: str,
    max_steps: int = 1000,
    fps: int = 60,
):
    predictor.reset()

    print(type(env))
    old_frames_per_step = env.frames_per_step
    env.frames_per_step = 1
    max_steps = max_steps * old_frames_per_step

    folder, prefix = os.path.split(out_path)
    folder = folder or "./"
    prefix = os.path.splitext(prefix)[0]

    rec = RecordVideo(
        env,
        video_folder=folder,
        name_prefix=prefix,
        video_length=max_steps,
        **{"fps": fps
    })


    # reset_out = rec.reset()
    # if isinstance(reset_out, tuple):   # Gymnasium ≥0.26  (obs, info)
    #     obs, _ = reset_out
    # else:                              # Stable‑Baselines3 VecEnv (obs only)
    #     obs = reset_out
    obs, _ = rec.reset()

    repeat = max(int(old_frames_per_step), 1)
    action = predictor.predict(obs)
    repeat_left = repeat

    steps = 0
    while steps < max_steps:
        # step_result = rec.step(action)
        # if len(step_result) == 5:            # Gymnasium
        #     obs, reward, terminated, truncated, _ = step_result
        #     done = terminated or truncated
        # else:                                # SB3 VecEnv
        #     obs, reward, done, _ = step_result
        obs, reward, terminated, truncated, _ = rec.step(action)

        steps      += 1
        repeat_left -= 1

        if terminated or truncated or steps >= max_steps:
            break

        if repeat_left == 0:
            action       = predictor.predict(obs)
            repeat_left  = repeat

    rec.video_recorder.close()
    # if os.path.exists(video_path) and video_path != out_path:
    #     os.replace(video_path, out_path)

    print(f"Replay saved ➜  {out_path}")

    env.frames_per_step = old_frames_per_step
