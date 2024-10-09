import time

import numpy as np
from numpy.typing import NDArray
import napari
from napari.layers import Image, Points
from napari.layers.utils.plane import (
    SlicingPlane,
    Point3D,
    ClippingPlane,
    ClippingPlaneList,
)
from napari.qt.threading import thread_worker, GeneratorWorker

from qtpy.QtWidgets import (  # pylint: disable=no-name-in-module
    QSlider,
    QGridLayout,
    QLabel,
    QPushButton,
    QWidget,
)

N_PLANE_TICKS = 500

DIAGONALS = np.array(
    [
        [[1, 1, 1], [0, 0, 0]],
        [[0, 1, 1], [1, 0, 0]],
        [[1, 0, 1], [0, 1, 0]],
        [[1, 1, 0], [0, 0, 1]],
    ]
)

def get_clipping_planes(
    normal: Point3D, thickness: int, slider_lookup_table: np.ndarray, slider_value: int
) -> ClippingPlaneList:
    """
    returns two clipping planes thickness away from slicing_plane
    """
    normal_array = np.array(normal)
    try:
        position1 = slider_lookup_table[slider_value + thickness]
    except IndexError:
        position1 = None
    if slider_value - thickness < 0:
        position2 = None
    else:
        position2 = slider_lookup_table[slider_value - thickness]
    plane_list: list[ClippingPlane] = []
    if position1 is not None:
            plane_list.append(ClippingPlane(position=position1, normal=normal_array))
    if position2 is not None:
            plane_list.append(ClippingPlane(position=position2, normal=-normal_array))
    return ClippingPlaneList(plane_list)


def vectors_from_diagonals(diagonals: NDArray) -> NDArray:
    """
    Gets the unit length vector describing the angle of each diagonal returns a
    numpy array of vectors
    """
    out = np.zeros((4, 3))
    for i, coords_pair in enumerate(diagonals):
        zeroed_vector = coords_pair[0] - coords_pair[1]
        out[i] = zeroed_vector / np.linalg.norm(zeroed_vector)
    return out


DIAGONAL_VECTORS = vectors_from_diagonals(DIAGONALS)


def get_slider_lookup_table(normal: Point3D, shape: tuple) -> NDArray:
    """
    gets a lookup table that has a point for each slider position
    """
    cosines = [(np.dot(normal, vector)) for vector in DIAGONAL_VECTORS]
    abs_cosines = np.abs(cosines)
    index = np.argmax(abs_cosines)
    corners = DIAGONALS[index] * shape
    if cosines[index] > 0:
        return np.linspace(corners[0], corners[1], N_PLANE_TICKS)
    else:
        return np.linspace(corners[1], corners[0], N_PLANE_TICKS)



def init_layers(
    viewer, points_thickness: int, image_layers: list[Image], points_layers: list[Points]
) -> tuple[SlicingPlane, ClippingPlaneList, NDArray]:
    """
    makes all layers be rendered as plane, returning SlicingPlane and
    """
    first_layer = image_layers[0]
    slider_lookup_table = get_slider_lookup_table((0, 1, 0), first_layer.data.shape)
    assert all(l.data.shape == first_layer.data.shape for l in image_layers)
    slicing_plane = SlicingPlane(
        position=slider_lookup_table[N_PLANE_TICKS // 2], normal=(0, 1, 0), thickness=1
    )
    for image in image_layers:
        image.plane = slicing_plane
        image.depiction = "plane"
    clipping_planes = get_clipping_planes(slicing_plane.normal, points_thickness, slider_lookup_table, N_PLANE_TICKS // 2)
    for points in points_layers:
        points.experimental_clipping_planes = clipping_planes

    return (
        slicing_plane,
        clipping_planes,
        slider_lookup_table,
    )




class ObliqueSlice(QWidget):  # type: ignore
    """
    Slices all image layers
    """

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.plane_angle = (0.0, 0.0)
        self.clipping_thickness = 15
        # init GUI
        # self.setLayout(QVBoxLayout())
        self.slicer, self.slice_button, self.rotator, self.rotate_button = self.init_gui()
        # connect callbacks
        self.viewer.camera.events.angles.connect((self, "update_angle_slider"))
        self.slicer.valueChanged.connect(self.update_slice)
        self.rotator.valueChanged.connect(self.rotate_camera)
        self.slice_button.clicked.connect(
            self.orient_plane_normal_along_view_direction
        )
        self.rotate_button.clicked.connect(self.reset_angle)
        # get a slicer for all layers
        self.image_layers = [l for l in self.viewer.layers if isinstance(l, Image)]
        self.points_layers = [l for l in self.viewer.layers if isinstance(l, Points)]
        self.slicing_plane, self.clipping_planes, self.slider_lookup_table = (
            init_layers(self.viewer, self.clipping_thickness, self.image_layers, self.points_layers)
        )
        self.update_angle_slider(None, angles=self.viewer.camera.angles)
        self.async_clip_worker: GeneratorWorker | None = None

    def init_gui(self) -> tuple[QSlider, QPushButton, QSlider, QPushButton]:
        # get widgets
        slicer_label = QLabel("Slice")
        slicer = QSlider()
        slicer.setRange(0, N_PLANE_TICKS - 1)
        slicer.setValue(N_PLANE_TICKS // 2)
        slice_button = QPushButton(text="reorient\nslice")
        rotator_label = QLabel("Roatate")
        rotator = QSlider()
        rotator.setRange(-180, 180)
        rotate_button = QPushButton(text="reorient\ncamera")
        # add to layout
        self.setLayout(QGridLayout())
        self.layout().addWidget(slicer_label, 0, 0)
        self.layout().addWidget(slicer, 1, 0)
        self.layout().addWidget(slice_button, 2, 0)
        self.layout().addWidget(rotator_label, 0, 1)
        self.layout().addWidget(rotator, 1, 1)
        self.layout().addWidget(rotate_button, 2, 1)
        return slicer, slice_button, rotator, rotate_button

    def reset_angle(self):
        roll, _, _ = self.viewer.camera.angles
        self.viewer.camera.angles = (roll, ) + self.plane_angle

    def update_slice(self, value: int):
        """
        upates the slice with value and updates layers

        also is a qt callback
        """
        if self.async_clip_worker is not None:
            self.async_clip_worker.quit()

        @thread_worker
        def async_clip(normal: Point3D, thickness: int, slider_lookup_table: np.ndarray, slider_value: int):
            time.sleep(.05)
            return get_clipping_planes(normal, thickness, slider_lookup_table, slider_value)

        out_plane = self.slicing_plane.copy()
        px, py, pz = self.slider_lookup_table[value]
        out_plane.position = px, py, pz
        self.slicing_plane = out_plane
        for image in self.image_layers:
            image.plane = self.slicing_plane
        self.async_clip_worker = async_clip(
            self.slicing_plane.normal, self.clipping_thickness, self.slider_lookup_table, self.slicer.value()
        )
        self.async_clip_worker.returned.connect(self.update_points)
        self.async_clip_worker.start()

    def update_points(self, clipping_planes):
        self.clipping_planes = clipping_planes
        assert self.async_clip_worker is not None
        for points in self.points_layers:
            if self.async_clip_worker.abort_requested:
                # stop updateing
                return
            points.experimental_clipping_planes = self.clipping_planes

    def update_angle_slider(self, event, angles: tuple | None = None):
        if angles is None:
            angles = event.value
        assert angles is not None
        self.rotator.setValue(int(angles[0]))

    def rotate_camera(self, value: int):
        _, a2, a3 = self.viewer.camera.angles
        self.viewer.camera.angles = value, a2, a3

    def orient_plane_normal_along_view_direction(self):
        if self.viewer is None or self.viewer.dims.ndisplay != 3:
            return
        layers = self.image_layers
        if len(layers) == 0:
            print("no layers to orient")
            return
        layer = layers[0]
        n1, n2, n3 = layer._world_to_displayed_data_ray(
            self.viewer.camera.view_direction, [-3, -2, -1]
        )
        out_plane = layer.plane.copy()
        out_plane.normal = (n1, n2, n3)
        self.slider_lookup_table = get_slider_lookup_table(
            (n1, n2, n3), layer.data.shape
        )
        self.slicing_plane = out_plane
        self.update_slice(self.slicer.value())
        self.plane_angle = self.viewer.camera.angles[1:]

