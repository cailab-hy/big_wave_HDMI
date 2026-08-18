import numpy as np


class ImageWindow:
    """A dockable omni.ui window that displays a numpy RGB image inside the IsaacLab UI.

    Used to show the robot's ego camera (tiled_camera) feed without an external
    OpenCV window. Only usable when the app runs with a GUI.
    """

    def __init__(self, title: str = "Robot Camera", width: int = 640, height: int = 480):
        import omni.ui as ui

        self._ui = ui
        self._provider = ui.ByteImageProvider()
        self._window = ui.Window(title, width=width, height=height)
        with self._window.frame:
            ui.ImageWithProvider(
                self._provider,
                fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT,
            )

    def update(self, frame: np.ndarray):
        """frame: (H, W, 3) uint8 RGB or (H, W, 4) uint8 RGBA."""
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        h, w = frame.shape[:2]
        if frame.shape[2] == 3:
            rgba = np.empty((h, w, 4), dtype=np.uint8)
            rgba[..., :3] = frame
            rgba[..., 3] = 255
        else:
            rgba = frame
        try:
            self._provider.set_data_array(rgba, [w, h])
        except AttributeError:
            # older omni.ui without set_data_array
            self._provider.set_bytes_data(rgba.reshape(-1).tolist(), [w, h])

    def close(self):
        if self._window is not None:
            self._window.visible = False
            self._window.destroy()
            self._window = None
