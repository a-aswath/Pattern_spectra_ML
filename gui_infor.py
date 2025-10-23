import numpy as np
from typing import Optional

class GUIInfo:
    def __init__(
        self,
        diatom_pgm: np.ndarray,
        shape_pgm: np.ndarray,
        width: int,
        height: int,
        tree: "MaxTree",
        filter_value: int,
        k: int,
        min_grey_level: int,
        num_attrs: int,
        procs: "ProcSet",
        dims: np.ndarray,
        dls: np.ndarray,
        dhs: np.ndarray,
        shared_cm: bool = False
    ):
        """
        Python equivalent of GUIInfoCreate().
        Stores all metadata, computed results, and GUI-related handles.
        """

        # --- Core algorithmic data ---
        self.tree = tree
        self.filter = filter_value
        self.k = k
        self.min_grey_level = min_grey_level
        self.num_attrs = num_attrs
        self.procs = procs
        self.dims = dims
        self.dls = dls
        self.dhs = dhs

        # --- Image data ---
        self.diatom_pgm = diatom_pgm
        self.diatom_width = width
        self.diatom_height = height
        self.shape_pgm = shape_pgm

        # --- Computed results ---
        self.result = self.allocate_result_array(num_attrs, dims)

        # --- Derived images ---
        self.gran_pgm: Optional[np.ndarray] = None
        self.compute_granulometry()
        self.create_granulometry_image()

        # --- GUI Handles ---
        self.display = self.create_display(shared_cm)
        self.control_window = None
        self.diatom_window = None
        self.diatom_image = None
        self.gran_window = None
        self.gran_image = None
        self.noise_info = None
        self.about_window = None

    # ---------------------------------------------------------------------
    # 🔧 Methods corresponding to the original C functions
    # ---------------------------------------------------------------------

    def gui_alloc_result_array(num_attrs: int, dims: list[int]) -> np.ndarray:
        """
        Python equivalent of GUIAllocResultArray().
        
        Args:
            num_attrs: Number of attributes (dimensions).
            dims: List or array of dimensions (length = num_attrs).
        
        Returns:
            A NumPy array of zeros (dtype=float64), with shape:
            (dims[0]+1, dims[1]+1, ..., dims[num_attrs-1]+1)
        """
        if len(dims) != num_attrs:
            raise ValueError("Length of 'dims' must match 'num_attrs'")

        # Compute shape as each dimension + 1
        shape = tuple(d + 1 for d in dims)

        # Allocate and zero-initialize
        result = np.zeros(shape, dtype=np.float64)
        return result

    def compute_granulometry(self):
        """Placeholder for GUIComputeGranulometry(info)."""
        # TODO: implement granulometry calculation based on self.tree
        print("✅ Computing granulometry... (placeholder)")

    def create_granulometry_image(self):
        """Placeholder for GUICreateGranulometryImage(info)."""
        # TODO: build visualization image from self.result
        self.gran_pgm = np.random.randint(0, 255, (self.dims[1], self.dims[0]), dtype=np.uint8)
        print("✅ Granulometry image created.")

    def create_display(self, shared_cm: bool):
        """Placeholder for TLCreateDisplay(sharedcm)."""
        # In Python, this might initialize a QApplication or PyQt display context
        return {"shared_colormap": shared_cm}

    # ---------------------------------------------------------------------
    # Utility functions for later integration with PyQt UI
    # ---------------------------------------------------------------------

    def refresh_gran_image(self):
        """Recompute and refresh the granulometry visualization."""
        self.compute_granulometry()
        self.create_granulometry_image()
        print("🔄 Granulometry visualization refreshed.")


# Example placeholder classes for MaxTree and ProcSet
class MaxTree:
    def __init__(self, nodes=None):
        self.nodes = nodes or []

class ProcSet:
    def __init__(self, processes=None):
        self.processes = processes or []
