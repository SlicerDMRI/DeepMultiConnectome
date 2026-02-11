"""Reference from https://github.com/zhangfanmark/DeepWMA"""
import numpy as np
import whitematteranalysis as wma
import sys
import vtk
import os
import matplotlib.pyplot as plt

sys.path.append('..')



class CustomFiberArray(wma.fibers.FiberArray):
    def convert_from_polydata(self, input_vtk_polydata, points_per_fiber=None, distribution='equidistant', decay_factor=2.0):

        """Convert input vtkPolyData to the fixed length fiber
        representation of this class.

        The polydata should contain the output of tractography.

        The output is downsampled fibers in array format and
        hemisphere info is also calculated.

        """

        # parameters for custom distance between points in downsampled streamline
        # distribution=self.distribution
        # decay_factor=self.decay_factor
        
        # points used in discretization of each trajectory
        if points_per_fiber is not None:
            self.points_per_fiber = points_per_fiber

        
        # line count. Assume all input lines are from tractography.
        self.number_of_fibers = input_vtk_polydata.GetNumberOfLines()

        if self.verbose:
            print(f"<{os.path.basename(__file__)}> Converting polydata to array representation. Lines: {self.number_of_fibers}")

        # allocate array number of lines by line length
        self.fiber_array_r = np.zeros((self.number_of_fibers,
                                            self.points_per_fiber))
        self.fiber_array_a = np.zeros((self.number_of_fibers,
                                            self.points_per_fiber))
        self.fiber_array_s = np.zeros((self.number_of_fibers,
                                            self.points_per_fiber))

        # loop over lines
        input_vtk_polydata.GetLines().InitTraversal()
        line_ptids = vtk.vtkIdList()
        inpoints = input_vtk_polydata.GetPoints()
        
        for lidx in range(0, self.number_of_fibers):

            input_vtk_polydata.GetLines().GetNextCell(line_ptids)
            line_length = line_ptids.GetNumberOfIds()

            if self.verbose:
                if lidx % 100 == 0:
                    print(f"<{os.path.basename(__file__)}> Line: {lidx} / {self.number_of_fibers}")
                    print(f"<{os.path.basename(__file__)}> number of points: {line_length}")

            pidx = 0
            for line_index in self._calculate_line_indices(line_length, self.points_per_fiber, distribution, decay_factor):
                # Get the lower and upper indices for interpolation
                lower_idx = int(np.floor(line_index))  # Floor index
                upper_idx = int(np.ceil(line_index))   # Ceil index

                # If the line_index is an integer, just use the point at that index
                if distribution == 'equidistant' or lower_idx == upper_idx:
                    ptidx = line_ptids.GetId(lower_idx)
                    point = inpoints.GetPoint(ptidx)
                else: # Interpolate between points
                    # Interpolation factor (how far line_index is between lower_idx and upper_idx)
                    t = line_index - lower_idx
                    
                    # Get points at lower_idx and upper_idx
                    ptidx_lower = line_ptids.GetId(lower_idx)
                    ptidx_upper = line_ptids.GetId(upper_idx)

                    point_lower = inpoints.GetPoint(ptidx_lower)
                    point_upper = inpoints.GetPoint(ptidx_upper)

                    # Linearly interpolate between the two points
                    point = [
                        (1 - t) * point_lower[0] + t * point_upper[0],
                        (1 - t) * point_lower[1] + t * point_upper[1],
                        (1 - t) * point_lower[2] + t * point_upper[2]
                    ]

                # Store the interpolated point in the fiber arrays
                self.fiber_array_r[lidx, pidx] = point[0]
                self.fiber_array_a[lidx, pidx] = point[1]
                self.fiber_array_s[lidx, pidx] = point[2]

                pidx += 1

        # initialize hemisphere info
        if self.hemispheres:
            self.calculate_hemispheres()

    def _calculate_line_indices(self, input_line_length, output_line_length, distribution='equidistant', decay_factor=2.0):
        """ Figure out indices for downsampling of polyline data.

        The indices include the first and last points on the line,
        plus points either spaced evenly (equidistant) or with more
        points toward the endpoints, based on a custom distribution.

        Parameters:
        - input_line_length: Number of points in the input line.
        - output_line_length: Desired number of output points.
        - distribution: 'equidistant' or 'exponential', defines how points are distributed.
        - decay_factor: Controls the exponential decay towards endpoints (for 'exponential' mode).
            0 will equal the equidistant distribution, higher values result in smaller distances in endpoints

        Returns:
        - ptlist: List of selected point indices.
        """

        if distribution == 'equidistant':
            # Equidistant spacing
            step = (input_line_length - 1.0) / (output_line_length - 1.0)
            ptlist = [ptidx * step for ptidx in range(output_line_length)]
        
        elif distribution == 'exponential':
            # Exponential decay towards the endpoints
            normalized_dist = np.linspace(-1, 1, output_line_length-1)
            
            # Apply exponential decay based on the decay factor
            weights = np.exp(-decay_factor * np.abs(normalized_dist))
            weights /= np.sum(weights)  # Normalize to sum to 1
            ptlist = np.cumsum(weights) * (input_line_length - 1)
            ptlist = np.insert(ptlist, 0, 0)
     
        # Test to ensure we hit the last point on the line
        if __debug__:
            test = (round(ptlist[-1]) == input_line_length - 1)
            if not test:
                print(f"<{os.path.basename(__file__)}> ERROR: fiber numbers don't add up.")
                print(ptlist)
                raise AssertionError

        return ptlist

    def visualize_fiber_distributions(self, input_line_length, output_line_length, path_plots, decay_factor=2.0):
        """Visualize the difference between equidistant and exponential spacing along a fiber."""
        equidistant_points = self._calculate_line_indices(input_line_length, output_line_length, 'equidistant')
        exponential_points = self._calculate_line_indices(input_line_length, output_line_length, 'exponential', decay_factor)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot equidistant spacing
        ax1.plot(equidistant_points, np.zeros_like(equidistant_points), 'ro-', markersize=10)
        ax1.set_title('Equidistant Spacing Along the Fiber', fontsize=14)
        ax1.set_yticks([])
        ax1.set_xlabel('Point Index Along the Fiber', fontsize=12)
        ax1.grid(True)

        for i, eq_point in enumerate(equidistant_points):
            ax1.text(eq_point, 0.05, f'{i}', color='red', fontsize=10, ha='center')

        # Plot exponential spacing
        ax2.plot(exponential_points, np.zeros_like(exponential_points), 'bo-', markersize=10)
        ax2.set_title(f'Exponential Spacing Along the Fiber (Decay Towards Endpoints with factor {decay_factor})', fontsize=14)
        ax2.set_yticks([])
        ax2.set_xlabel('Point Index Along the Fiber', fontsize=12)
        ax2.grid(True)

        for i, exp_point in enumerate(exponential_points):
            ax2.text(exp_point, 0.05, f'{i}', color='blue', fontsize=10, ha='center')

        plt.tight_layout()
        plt.savefig(f"{path_plots}streamline_sampling{round(decay_factor)}")
        
def read_tractography(tractography_path):
    """Read tractography data and convert it to a feature array."""
    pd_tractography = wma.io.read_polydata(tractography_path)
    fiber_array = CustomFiberArray()
    fiber_array.convert_from_polydata(pd_tractography, points_per_fiber=15, distribution='exponential', decay_factor=2.0)
    feat = np.dstack((fiber_array.fiber_array_r, fiber_array.fiber_array_a, fiber_array.fiber_array_s))
    return feat, fiber_array

def feat_RAS(pd_tract, number_of_points=15, decay_factor=0):
    """The most simple feature for initial test"""
    fiber_array = CustomFiberArray()
    fiber_array.convert_from_polydata(pd_tract, points_per_fiber=number_of_points, distribution='exponential', decay_factor=decay_factor)
    # fiber_array_r, fiber_array_a, fiber_array_s have the same size: [number of fibers, points of each fiber]
    feat = np.dstack((fiber_array.fiber_array_r, fiber_array.fiber_array_a, fiber_array.fiber_array_s))

    return feat, fiber_array

if __name__ == "__main__":
    # Visualize how the streamline sampling is performed with exponential mode
    fiber_array = CustomFiberArray()
    fiber_array.visualize_fiber_distributions(input_line_length=100, output_line_length=15, decay_factor=8, path_plots="../plots/")
