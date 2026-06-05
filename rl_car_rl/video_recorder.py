"""
Video recording utility for evaluation episodes.
Records frames from the renderer and encodes them as MP4 video.

Usage:
    from video_recorder import VideoRecorder
    recorder = VideoRecorder("output.mp4", fps=30)
    recorder.add_frame(frame_array)
    recorder.close()
"""

import os
import numpy as np


class VideoRecorder:
    """Records rendered frames and exports as MP4 video.

    Uses imageio for encoding if available; falls back to saving frames
    as individual PNG files with a note to use ffmpeg for conversion.
    """

    def __init__(self, output_path: str, fps: int = 30, crf: int = 23):
        self.output_path = output_path
        self.fps = fps
        self.crf = crf
        self.frames: list[np.ndarray] = []
        self._closed = False

        # Auto-append extension
        if not output_path.endswith(".mp4"):
            self.output_path = output_path.rsplit(".", 1)[0] + ".mp4"

    def __del__(self):
        if not self._closed and len(self.frames) > 0:
            try:
                self.close()
            except Exception:
                pass  # Don't raise during garbage collection

    def add_frame(self, frame: np.ndarray) -> None:
        """Add a frame to the recording buffer.

        Args:
            frame: RGB image as numpy array, shape (H, W, 3), dtype uint8.
        """
        self.frames.append(frame.copy())

    def close(self) -> str:
        """Encode and save the video. Returns the output path."""
        if self._closed:
            return self.output_path

        self._closed = True

        if len(self.frames) == 0:
            print("VideoRecorder: no frames recorded.")
            return ""

        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        try:
            import imageio
            writer = imageio.get_writer(
                self.output_path,
                fps=self.fps,
                codec="libx264",
                quality=None,
                ffmpeg_params=["-crf", str(self.crf), "-preset", "medium"],
            )
            for frame in self.frames:
                writer.append_data(frame)
            writer.close()
            file_size = os.path.getsize(self.output_path) / (1024 * 1024)
            print(f"Video saved to {self.output_path} ({file_size:.1f} MB, {len(self.frames)} frames @ {self.fps} fps)")
        except ImportError:
            self._save_frames_as_png()
            print("imageio not installed. Install with: pip install imageio[ffmpeg]")
        except Exception as e:
            print(f"Video encoding failed: {e}")
            self._save_frames_as_png()

        return self.output_path

    def _save_frames_as_png(self):
        fallback_dir = self.output_path.rsplit(".", 1)[0] + "_frames"
        os.makedirs(fallback_dir, exist_ok=True)
        try:
            from PIL import Image
            for i, frame in enumerate(self.frames):
                img = Image.fromarray(frame)
                img.save(os.path.join(fallback_dir, f"frame_{i:06d}.png"))
        except ImportError:
            print(f"PIL not available. Frames not saved.")
            return
        print(f"Saved {len(self.frames)} frames to {fallback_dir}/")
        print("To create video: ffmpeg -framerate {} -i {}/frame_%06d.png -c:v libx264 -crf {} {}.mp4".format(
            self.fps, fallback_dir, self.crf, self.output_path[:-4]
        ))
