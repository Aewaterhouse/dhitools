"""DHI MIKE21 dfsu functions

Authors: 
Robert Wall (original dhitools package),
Alex Waterhouse (expanded functionality)

"""

from . import dfsu
from . import mesh
from . import plot
from . import _utils
from . import config
from . import units

import os
import clr
import numpy as np
from scipy.spatial import cKDTree
import datetime as dt

# Set path to MIKE SDK
sdk_path = config.MIKE_SDK
dfs_dll = config.MIKE_DFS
eum_dll = config.MIKE_EUM
clr.AddReference(os.path.join(sdk_path, dfs_dll))
clr.AddReference(os.path.join(sdk_path, eum_dll))
clr.AddReference("System")

# Import .NET libraries
import System
from System import Array
from DHI.Generic.MikeZero import eumQuantity
import DHI.Generic.MikeZero.DFS as dfs


class Dfsu(mesh.Mesh):
    """
    MIKE21 dfsu class. Contains many attributes read in from the input `.dfsu`
    file. Uses :class:`dhitools.mesh.Mesh` as a base class and inherits its
    methods.

    Parameters
    ----------
    filename : str
        Path to .dfsu

    Attributes
    ----------
    filename : str
        Path to .dfsu
    items : dict
        List .dfsu items (ie. surface elevation, current speed), item index
        to lookup in .dfsu, item units and counts of elements, nodes and
        time steps.
    projection : str
        .dfsu spatial projection string in WKT format
    element_table : ndarray, shape (num_ele, 3)
        Defines for each element the nodes that define the element.
    node_table : ndarray, shape (num_nodes, n)
        Defines for each node the element adjacent to this node. May contain
        padded zeros
    nodes : ndarray, shape (num_nodes, 3)
        (x,y,z) coordinate for each node
    elements : ndarray, shape (num_ele, 3)
        (x,y,z) coordinate for each element
    start_datetime_str : str
        Start datetime (as a string)
    start_datetime : datetime
        Start datetime (datetime object)
    end_datetime : datetime
        End datetime (datetime object)
    timestep : float
        Timestep delta in seconds
    number_tstep : int
        Total number of timesteps
    time : ndarray, shape (number_tstep,)
        Sequence of datetimes between start and end datetime at delta timestep

    See Also
    ----------
    * Many of these methods have been adapated from the `DHI MATLAB Toolbox <https://github.com/DHI/DHI-MATLAB-Toolbox>`_
    """

    def __init__(self, filename=None):
        super(Dfsu, self).__init__(filename)
        if self.filename is not None:
            self.read_dfsu(self.filename)

    def read_dfsu(self, filename):
        """
        Read in .dfsu file and read attributes

        Parameters
        ----------
        filename : str
            File path to .dfsu file
        """
        self.read_mesh(filename)

        dfsu_object = dfs.DfsFileFactory.DfsuFileOpen(self.filename)

        mesh_in_3d = _3d_element_geo(dfsu_object, self.element_table)

        self.geo2d = _calc_2d_geo(
            dfsu_object,
            self.elements,
            self.nodes,
            self.node_ids,
            self.element_table,
            mesh_in_3d[2],
        )

        self.element_ids_horz = mesh_in_3d[0]
        self.element_ids_vert = mesh_in_3d[1]
        self.element_ids_surf = mesh_in_3d[2]

        # Time attributes
        self.start_datetime_str = dfsu_object.StartDateTime.Date.ToString()
        dt_start_obj = dfsu_object.StartDateTime
        self.start_datetime = dt.datetime(
            year=dt_start_obj.Year,
            month=dt_start_obj.Month,
            day=dt_start_obj.Day,
            hour=dt_start_obj.Hour,
            minute=dt_start_obj.Minute,
            second=dt_start_obj.Second,
        )
        self.timestep = dfsu_object.TimeStepInSeconds
        self.number_tstep = dfsu_object.NumberOfTimeSteps - 1
        self.end_datetime = self.start_datetime + dt.timedelta(
            seconds=self.timestep * self.number_tstep
        )
        self.time = np.arange(
            self.start_datetime,
            self.end_datetime + dt.timedelta(seconds=self.timestep),
            dt.timedelta(seconds=self.timestep),
        ).astype(dt.datetime)

        self.items = _dfsu_info(dfsu_object)
        dfsu_object.Close()

    def summary(self):
        """
        Prints a summary of the dfsu
        """
        print("Input .dfsu file: {}".format(self.filename))
        print("Num. Elmts = {}".format(self.num_elements))
        print("Num. Nodes = {}".format(self.num_nodes))
        if self.num_layers > 0:
            print("Num. Layers = {}".format(self.num_layers))
            print("Num. Sigma Layers = {}".format(self.num_siglayers))
        print("Mean elevation = {}".format(np.mean(self.nodes[:, 2])))
        print("Projection = \n {}".format(self.projection))
        print("\n")
        print("Time start = {}".format(self.start_datetime_str))
        print("Number of timesteps = {}".format(self.number_tstep))
        print("Timestep = {}".format(self.timestep))
        print("\n")
        print("Number of items = {}".format(len(self.items) - 3))
        print("Items:")
        for k in self.items.keys():
            if k not in ["num_elements", "num_nodes", "num_timesteps"]:
                print(
                    "{}, unit = {}, index = {}".format(
                        k, self.items[k]["unit"], self.items[k]["index"]
                    )
                )

    def item_element_data(
        self, item_name, tstep_start=None, tstep_end=None, element_list=None,
    ):
        """
        Get element data for specified item with option to specify range of
        timesteps.

        Parameters
        ----------
        item_name : str
            Specified item to return element data. Item names are found in
            the `Dfsu.items` attribute.
        tstep_start : int or None, optional
            Specify time step for element data. Timesteps begin from 0.
            If `None`, returns data from 0 time step.
        tstep_end : int or None, optional
            Specify last time step for element data. Allows for range of time
            steps to be returned, where `tstep_end` is included.Must be
            positive int <= number of timesteps
            If `None`, returns single time step specified by `tstep_start`
            If `-1`, returns all time steps from `tstep_start`:end
        element_list : list, optional
            Provide list of elements. Element numbers are as seen by MIKE
            programs and adjusted for Python indexing.

        Returns
        -------
        ele_data : ndarray, shape (num_elements,[tstep_end-tstep_start])
            Element data for specified item and time steps
            `element_list` will change num_elements returned in `ele_data`

        """
        dfsu_object = dfs.DfsFileFactory.DfsuFileOpen(self.filename)
        ele_data = _element_data(
            dfsu_object=dfsu_object,
            item_name=item_name,
            item_info=self.items,
            tstep_start=tstep_start,
            tstep_end=tstep_end,
            element_list=element_list,
        )
        dfsu_object.Close()

        return ele_data

    def item_node_data(self, item_name, tstep_start=None, tstep_end=None):
        """
        Get node data for specified item with option to specify range of
        timesteps.

        Parameters
        ----------
        item_name : str
            Specified item to return node data. Item names are found in
            the `Dfsu.items` attribute.
        tstep_start : int or None, optional
            Specify time step for node data. Timesteps begin from 0.
            If `None`, returns data from 0 time step.
        tstep_end : int or None, optional
            Specify last time step for node data. Allows for range of time
            steps to be returned, where `tstep_end` is included.Must be
            positive int <= number of timesteps
            If `None`, returns single time step specified by `tstep_start`
            If `-1`, returns all time steps from `tstep_start`:end

        Returns
        -------
        node_data : ndarray, shape (num_nodes,[tstep_end-tstep_start])
            Node data for specified item and time steps
        """

        dfsu_object = dfs.DfsFileFactory.DfsuFileOpen(self.filename)
        node_data = _node_data(
            dfsu_object=dfsu_object,
            item_name=item_name,
            item_info=self.items,
            ele_cords=self.elements,
            node_cords=self.nodes,
            node_table=self.node_table,
            tstep_start=tstep_start,
            tstep_end=tstep_end,
        )
        dfsu_object.Close()

        return node_data

    def ele_to_node(self, z_element):
        """
        Convert data at element coordinates to node coordinates

        Parameters
        ----------
        z_element : ndarray, shape (num_elements,)
            Data corresponding to order and coordinates of elements

        Returns
        -------
        z_node : ndarray, shape (num_nodes,)
            Data corresponding to order and coordinates of nodes

        """

        z_node = _map_ele_to_node(
            node_table=self.node_table,
            elements=self.elements,
            nodes=self.nodes,
            element_data=z_element,
        )
        return z_node

    def max_item(
        self, item_name, tstep_start=None, tstep_end=None, current_dir=False, node=False
    ):
        """
        Calculate maximum element value for specified item over entire model or
        within specific range of timesteps.

        Parameters
        ----------
        item_name : str
            Specified item to return element data. Item names are found in
            the `Dfsu.items` attribute.
        tstep_start : int or None, optional
            Specify time step for data considered in determining maximum.
            Timesteps begin from 0.
            If `None`, returns data from 0 time step.
        tstep_end : int or None, optional
            Specify last time step for data considered in determining maximum
            Must be positive int <= number of timesteps
            If `None`, returns all time steps from `tstep_start`:end
        current_dir : boolean
            If True, returns corresponding current direction value occuring at
            the maxmimum of specified `item_name`.
        node : boolean, optional
            If True, returns item data at node rather than element

        Returns
        -------
        If `current_dir` is False:
        max_ele : ndarray, shape (num_elements,)
            Maximum elements values for specified item

        If `current_dir` is True
        max_ele : ndarray, shape (num_elements,)
            Maximum elements values for specified item
        max_current_dir : ndarray, shape (num_elements,)
            Current direction corresponding to `max_ele`

        if `node` is True
        min_node : ndarray, shape (num_nodes,)
            Minimum node values for specified item

        If `node` and `current_dir` are True
        min_node : ndarray, shape (num_nodes,)
            Minimum node values for specified item
        min_current_dir : ndarray, shape (num_elements,)
            Current direction corresponding to `min_node`

        """

        dfsu_object = dfs.DfsFileFactory.DfsuFileOpen(self.filename)
        max_ele = _item_aggregate_stats(
            dfsu_object,
            item_name,
            self.items,
            tstep_start=tstep_start,
            tstep_end=tstep_end,
            current_dir=current_dir,
        )
        dfsu_object.Close()

        # Return either element data or convert to node if specified
        if node:
            if current_dir:
                me = max_ele[0]
                cd = max_ele[1]
            else:
                me = max_ele

            # Max element item to node
            max_node = _map_ele_to_node(
                node_table=self.node_table,
                elements=self.elements,
                nodes=self.nodes,
                element_data=me,
            )
            # Current at element to node
            if current_dir:
                cd_node = _map_ele_to_node(
                    node_table=self.node_table,
                    elements=self.elements,
                    nodes=self.nodes,
                    element_data=cd,
                )

                return max_node, cd_node
            else:
                return max_node

        else:
            if current_dir:
                return max_ele[0], max_ele[1]
            else:
                return max_ele

    def min_item(
        self, item_name, tstep_start=None, tstep_end=None, current_dir=False, node=False
    ):
        """
        Calculate minimum element value for specified item over entire model or
        within specific range of timesteps.

        Parameters
        ----------
        item_name : str
            Specified item to return element data. Item names are found in
            the `Dfsu.items` attribute.
        tstep_start : int or None, optional
            Specify time step for data considered in determining minimum.
            Timesteps begin from 0.
            If `None`, returns data from 0 time step.
        tstep_end : int or None, optional
            Specify last time step for data considered in determining minimum
            Must be positive int <= number of timesteps
            If `None`, returns all time steps from `tstep_start`:end
        current_dir : boolean
            If True, returns corresponding current direction value occuring at
            the maxmimum of specified `item_name`.
        node : boolean, optional
            If True, returns item data at node rather than element

        Returns
        -------
        If `current_dir` is False:
            min_ele : ndarray, shape (num_elements,)
                Minimum elements values for specified item

        If `current_dir` is True
            min_ele : ndarray, shape (num_elements,)
                Minimum elements values for specified item
            min_current_dir : ndarray, shape (num_elements,)
                Current direction corresponding to `min_ele`

        if `node` is True
            min_node : ndarray, shape (num_nodes,)
                Minimum node values for specified item

        If `node` and `current_dir` are True
            min_node : ndarray, shape (num_nodes,)
                Minimum node values for specified item
            min_current_dir : ndarray, shape (num_elements,)
                Current direction corresponding to `min_node`

        """

        dfsu_object = dfs.DfsFileFactory.DfsuFileOpen(self.filename)
        min_ele = _item_aggregate_stats(
            dfsu_object,
            item_name,
            self.items,
            tstep_start=tstep_start,
            tstep_end=tstep_end,
            return_max=False,
            current_dir=current_dir,
        )
        dfsu_object.Close()

        # Return either element data or convert to node if specified
        if node:
            if current_dir:
                me = min_ele[0]
                cd = min_ele[1]
            else:
                me = min_ele

            # Max element item to node
            min_node = _map_ele_to_node(
                node_table=self.node_table,
                elements=self.elements,
                nodes=self.nodes,
                element_data=me,
            )
            # Current at element to node
            if current_dir:
                cd_node = _map_ele_to_node(
                    node_table=self.node_table,
                    elements=self.elements,
                    nodes=self.nodes,
                    element_data=cd,
                )

                return min_node, cd_node
            else:
                return min_node

        else:
            if current_dir:
                return min_ele[0], min_ele[1]
            else:
                return min_ele

    def max_ampltiude(self, item_name="Maximum water depth", datum_shift=0, nodes=True):
        """
        Calculate maximum amplitude from MIKE21 inundation output.

        Specifically, takes the MIKE21 output for `Maximum water depth` across
        the model run, adjusted for `datum_shift` and calculates maximum
        amplitude by the difference between the depth and mesh elevation

        Datum shift applies are different water level to a model run and the
        mesh elevation values saved within the `dfsu` file will be adjusted by
        the datum shift. So, providing the datum shift is necessary to
        calculate the correct amplitudes.

        Parameters
        ----------
        item_name : str
            Default is 'Maxium water depth' which is the default output name
            from MIKE21. Can parse an alternative string if a different name
            has been used.
        datum_shift : float
            Adjust for datum_shift value used during model run. Only necessary
            if a datum shift was applied to the model. Default is 0.
        nodes : boolean
            If True, return data at node coordinates
            If False, return data at element coordinates

        Returns
        -------
        if `node` is True
        max_amplitude : ndarray, shape (num_nodes,)
            Max amplitude across entire model run at node coordinates

        if `node` is False
        max_amplitude : ndarray, shape (num_elements,)
            Max amplitude across entire model run at element coordinates
        """

        depth = self.item_element_data(item_name)[:, 0]  # Load max depth
        max_amplitude_ele = self.elements[:, 2] + datum_shift  # Dfsu elevation
        max_amplitude_ele[max_amplitude_ele > 0] = 0  # Set overland values to 0
        # Max amplitdue at elements; add since depths are negative
        max_amplitude_ele += depth

        if nodes:
            max_amplitude = self.ele_to_node(max_amplitude_ele)
            return max_amplitude
        else:
            return max_amplitude_ele

    def plot_item(
        self, layer=None, item_name=None, tstep=None, node_data=None, kwargs=None
    ):
        """
        Plot triangular mesh with tricontourf for input item and timestep

        **Warning**: if mesh is large performance will be poor

        Parameters
        ----------
        layer : int
            Specified layer to plot in 2D. If `None`, will plot surface layer
        item_name : str
            Specified item to return element data. Item names are found in
            the `Dfsu.items` attribute.
        tstep : int
            Specify time step for node data. Timesteps begin from 0.
        node_date : ndarray or None, shape (num_nodes,), optional
            Provide data at node coordinates to plot. Will take precedence
            over `item_name` and `tstep`.
        kwargs : dict
            Additional arguments supported by tricontourf

        Returns
        -------
        fig : matplotlib figure obj
        ax : matplotlib axis obj
        tf : tricontourf obj

        """
        # if node_data is None:
        #     # Get item_data and reshape from (N,1) to (N,) because of single
        #     # timestep. tricontourf prefers (N,)
        #     assert tstep is not None, "Must provided tstep if providing `item_name`"
        #     item_data = self.item_node_data(item_name, tstep)
        #     item_data = np.reshape(item_data, self.num_nodes)

        # else:
        #     item_data = node_data

        # fig, ax, tf = plot.filled_mesh_plot(
        #     self.nodes[:, 0], self.nodes[:, 1], item_data, self.element_table, kwargs
        # )

        # return fig, ax, tf
        print("Not yet implemented")
        

    def plot_mesh(self, fill=False, kwargs=None):
        """
        Plot Horizontal (2D) representative of triangular mesh 
        with triplot or tricontourf.

        See matplotlib kwargs for respective additional plot arguments.

        **Warning**: if mesh is large performance will be poor

        Parameters
        ----------
        fill : boolean
            if True, plots filled contour mesh (tricontourf)
            if False, plots (x, y) triangular mesh (triplot)
        kwargs : dict
            Additional arguments supported by triplot/tricontourf

        Returns
        -------
        fig : matplotlib figure obj
            Figure object
        ax : matplotlib axis obj
            Axis object

        If `fill` is True
            tf : matplotlib tricontourf obj
                Tricontourf object

        See Also
        --------
        * `Triplot <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.triplot.html>`_
        * `Tricontourf <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.tricontourf.html>`_

        """

        nodes = self.geo2d["nodes"]
        element_table = self.geo2d["element_table"]

        if fill:
            fig, ax, tf = plot.filled_mesh_plot(
                nodes[:, 0], nodes[:, 1], nodes[:, 2], element_table, kwargs,
            )
            return fig, ax, tf

        else:
            fig, ax = plot.mesh_plot(nodes[:, 0], nodes[:, 1], element_table, kwargs)
            return fig, ax

    def gridded_item(
        self,
        item_name=None,
        tstep_start=None,
        tstep_end=None,
        res=1000,
        node=True,
        node_data=None,
    ):
        """
        Calculate gridded item data, either from nodes or elements, at
        specified grid resolution and for a range of time steps. Allows
        for downsampling of high resolution mesh's to a more manageable size.

        The method :func:`grid_res() <dhitools.mesh.Mesh.grid_res()>` needs to be run before this to calculate the grid
        parameters needed for interpolation. Pre-calculating these also greatly
        improves run-time. res and node must be consistent between grid_res()
        and gridded_item().

        Parameters
        ----------
        item_name : str
            Specified item to return node data. Item names are found in
            the `Dfsu.items` attribute.
        tstep_start : int or None, optional
            Specify time step for node data. Timesteps begin from 0.
            If `None`, returns data from 0 time step.
        tstep_end : int or None, optional
            Specify last time step for node data. Allows for range of time
            steps to be returned, where `tstep_end` is included.Must be
            positive int <= number of timesteps
            If `None`, returns single time step specified by `tstep_start`
            If `-1`, returns all time steps from `tstep_start`:end
        res : int
            Grid resolution
        node : bool
            If true, interpolate from node data,
            Else, interpolate from element data
        node_data : ndarray or None, shape (num_nodes,), optional
            Provide data at node coordinates to create grid from. Will take
            precedence over `item_name`.

        Returns
        -------
        z_interp : ndarray, shape (num_timsteps, len_xgrid, len_ygrid)
            Interpolated z grid for each timestep
        """

        from . import _gridded_interpolate as _gi

        # Check that grid parameters have been calculated and if they are,
        # that they match the specified res
        assert (
            self._grid_calc is True
        ), "Must calculate grid parameters first using method grid_res(res)"
        assert (
            self._grid_res == res
        ), "Input grid resolution must equal resolution input to grid_res()"
        assert (
            self._grid_node == node
        ), "grid_res(node) must be consistent with gridded_item(node)"

        if node_data is None:
            if node:
                z = self.item_node_data(item_name, tstep_start, tstep_end)
            else:
                z = self.item_element_data(item_name, tstep_start, tstep_end)
        else:
            z = np.reshape(node_data, (node_data.shape[0], 1))

        # Interpolate z to regular grid
        num_tsteps = z.shape[1]
        z_interp = np.zeros(
            shape=(num_tsteps, self.grid_x.shape[0], self.grid_x.shape[1])
        )
        for i in range(num_tsteps):
            z_interp_flat = _gi.interpolate(
                z[:, i], self.grid_vertices, self.grid_weights
            )
            z_interp_grid = np.reshape(
                z_interp_flat, (self.grid_x.shape[0], self.grid_y.shape[1])
            )
            z_interp[i] = z_interp_grid

        return z_interp

    def gridded_stats(
        self, item_name, tstep_start=None, tstep_end=None, node=True, max=True, res=1000
    ):
        """
        Calculate gridded item maximum or minimum across time range,
        either from nodes or elements, at specified grid resolution. Allows
        for downsampling of high resolution mesh's to a more manageable size.

        The method grid_res() needs to be run before this to calculate the grid
        parameters needed for interpolation. Pre-calculating these also greatly
        improves run-time. res and node must be consistent between grid_res()
        and gridded_item().

        Parameters
        ----------
        item_name : str
            Specified item to return element data. Item names are found in
            the `Dfsu.items` attribute.
        tstep_start : int or None, optional
            Specify time step for data considered in determining maximum.
            Timesteps begin from 0.
            If `None`, returns data from 0 time step.
        tstep_end : int or None, optional
            Specify last time step for data considered in determining maximum
            Must be positive int <= number of timesteps
            If `None`, returns all time steps from `tstep_start`:end
        node : boolean, optional
            If True, returns item data at node rather than element
        max : boolean, optional
            If True, returns max (see method max_item()) else returns min

        Returns
        -------
        z_interp : ndarray, shape (len_xgrid, len_ygrid)
            Interpolated z grid
        """

        from . import _gridded_interpolate as _gi

        # Check that grid parameters have been calculated and if they are,
        # that they match the specified res
        assert (
            self._grid_calc is True
        ), "Must calculate grid parameters first using method grid_res(res)"
        assert (
            self._grid_res == res
        ), "Input grid resolution must equal resolution input to grid_res()"
        assert (
            self._grid_node == node
        ), "grid_res(node) must be consistent with gridded_item(node)"

        if max:
            z = self.max_item(item_name, tstep_start, tstep_end, node=node)
        else:
            z = self.min_item(item_name, tstep_start, tstep_end, node=node)
        z_interp_flat = _gi.interpolate(z, self.grid_vertices, self.grid_weights)
        z_interp = np.reshape(
            z_interp_flat, (self.grid_x.shape[0], self.grid_y.shape[1])
        )

        return z_interp

    def boolean_mask(self, mesh_mask, res=1000):
        """
        Create a boolean mask of a regular grid at input resolution indicating
        if gridded points are within the model mesh.

        This is slightly different to the mesh method which will automatically
        create the mask if it isn't provided. This will not automatically
        create the mask and the mask method has been disabled. See mask() for
        further details.

        Parameters
        ----------
        res : int
            Grid resolution
        mesh_mask : shapely Polygon object
            Mesh domain mask output from the :func:`mask() <dhitools.mesh.Mesh.mask()>`
            or any shapely polygon. `mesh_mask` will be used to determine
            gridded points that are within the polygon.

        Returns
        -------
        bool_mask : ndarray, shape (len_xgrid, len_ygrid)
            Boolean mask covering the regular grid for the mesh domain

        """
        from . import _gridded_interpolate as _gi
        from shapely.geometry import Point

        # Create (x,y) grid at input resolution
        X, Y = _gi.dfsu_XY_meshgrid(self.nodes[:, 0], self.nodes[:, 1], res=res)

        # Create boolean mask
        bool_mask = []
        for xp, yp in zip(X.ravel(), Y.ravel()):
            bool_mask.append(Point(xp, yp).within(mesh_mask))
        bool_mask = np.array(bool_mask)
        bool_mask = np.reshape(bool_mask, X.shape)

        return bool_mask

    def mask(self):
        """
        Method disabled for dfsu class since the node boundary codes for
        dfsu files are not consistent with mesh boundary codes particularly
        when dfsu output is a subset of the mesh
        """
        raise AttributeError("'dfsu' object has no attribute 'mask'")

    def find_nearest_element(self, points, tree=None):
        """ Returns 2D element number of the nearest ele to each point"""
        if tree is None:
            xy = self.geo2d["elements"][:, 0:2]
            tree = cKDTree(xy)

        _, idx = tree.query(points, k=1)
        ind = np.column_stack(np.unravel_index(idx, xy.shape[0]))

        return ind + 1  # +1 to align with DHI ele-num

    def extract_2D_element_data(
        self, item_name=None, tstep_start=None, tstep_end=None, layers=None
    ):
        """
        Extract element data and convert to 2D by taking the maximum.
        
        Element data can be combined by taking the maximum throughout all vertical
        layers (default), or for a specified layer or layers if supplied. 
        
        Parameters
        ----------
        item_name : str
            Specified item to return element data. Item names are found in
            the `Dfsu.items` attribute.
        tstep_start : int or None, optional
            Specify time step for data considered in determining maximum.
            Timesteps begin from 0.
            If `None`, returns data from 0 time step.
        tstep_end : int or None, optional
            Specify last time step for data considered in determining maximum
            Must be positive int <= number of timesteps
            If `None`, returns all time steps from `tstep_start`:end
        layers : list, Optional
            List of layers to take the maximum value over.
        """

        dfsu_geo = {}
        dfsu_geo["elements_2d"] = self.geo2d["elements"]
        dfsu_geo["element_ids_horz"] = self.element_ids_horz
        dfsu_geo["element_ids_vert"] = self.element_ids_vert

        dfsu_object = dfs.DfsFileFactory.DfsuFileOpen(self.filename)

        element_data_2d = _extract_2D_element_data(
            dfsu_object,
            item_name,
            self.items,
            dfsu_geo,
            tstep_start=tstep_start,
            tstep_end=tstep_end,
            layers=layers,
        )

        dfsu_object.Close()

        return element_data_2d

    def calc_percentiles_3Dto2D(
        self,
        item_name,
        percentiles,
        tstep_start=None,
        tstep_end=None,
        node=False,
        layers=None,
        element_data_2d=None,
    ):
        """
        Calculate percentile levels for a specified item over entire model domain.
        
        The function will calculate the value based on the maximum value throughout 
        the  water column (i.e. max over all vertical layers for each cell), or 
        based on the maximum value over specified layers (layers = List of Ints), 
        or for a single layer (layer = Int).
        
        Parameters
        ----------
        item_name : str
            Specified item to return element data. Item names are found in
            the `Dfsu.items` attribute.
        percentiles: int or tuple
            Percentile(s) to be calculated
        tstep_start : int or None, optional
            Specify time step for data considered in determining maximum.
            Timesteps begin from 0.
            If `None`, returns data from 0 time step.
        tstep_end : int or None, optional
            Specify last time step for data considered in determining maximum
            Must be positive int <= number of timesteps
            If `None`, returns all time steps from `tstep_start`:end
        node : boolean, optional
            If True, returns item data at node rather than element
        layers : list, Optional
            List of layers to take the maximum value over.
        element_data_2d : numpy.array, Optional
            Pre-extracted element data can be supplied (nelements x ntimesteps)

        Returns
        -------

        if `node` is False
        max_ele : ndarray, shape (num_elements,)
            Maximum elements values for specified item

        if `node` is True
        min_node : ndarray, shape (num_nodes,)
            Minimum node values for specified item

        """

        # User may have extracted element data and applied custom calcs beforehand.
        if element_data_2d is None:
            # Pass required geometry information
            dfsu_geo = {}
            dfsu_geo["elements_2d"] = self.geo2d["elements"]
            dfsu_geo["element_ids_horz"] = self.element_ids_horz
            dfsu_geo["element_ids_vert"] = self.element_ids_vert

            dfsu_object = dfs.DfsFileFactory.DfsuFileOpen(self.filename)

            element_data_2d = _extract_2D_element_data(
                dfsu_object,
                item_name,
                self.items,
                dfsu_geo,
                tstep_start=tstep_start,
                tstep_end=tstep_end,
                layers=layers,
            )

            dfsu_object.Close()

        ele_percentile_data = np.percentile(element_data_2d, percentiles, axis=1).T

        # Return either element data or convert to node if specified
        if node:
            node_percentile_data = _map_ele_to_node(
                node_table=self.node_table,
                elements=self.elements,
                nodes=self.nodes,
                element_data=ele_percentile_data,
            )

            return node_percentile_data
        else:
            return ele_percentile_data

    def calc_percentage_exceedance_3Dto2D(
        self,
        item_name,
        thresholds,
        tstep_start=None,
        tstep_end=None,
        node=False,
        layers=None,
        element_data_2d=None,
    ):
        """
        Calculate percentage of time that a specified item 
        exceeds a threshold value over entire model domain.
        
        The function will calculate the value based on the maximum value throughout 
        the  water column (i.e. max over all vertical layers for each cell), or 
        based on the maximum value over specified layers (layers = List of Ints), 
        or for a single layer (layer = Int).
        
        Parameters
        ----------
        item_name : str
            Specified item to return element data. Item names are found in
            the `Dfsu.items` attribute.
        threholds: int, tuple,
            Percentile(s) to be calculated
        tstep_start : int or None, optional
            Specify time step for data considered in determining maximum.
            Timesteps begin from 0.
            If `None`, returns data from 0 time step.
        tstep_end : int or None, optional
            Specify last time step for data considered in determining maximum
            Must be positive int <= number of timesteps
            If `None`, returns all time steps from `tstep_start`:end
        node : boolean, optional
            If True, returns item data at node rather than element
        layers : list, Optional
            List of layers to take the maximum value over.
        element_data_2d : numpy.array, Optional
            Pre-extracted element data can be supplied (nelements x ntimesteps)
        

        Returns
        -------

        if `node` is False
        max_ele : ndarray, shape (num_elements,)
            Maximum elements values for specified item

        if `node` is True
        min_node : ndarray, shape (num_nodes,)
            Minimum node values for specified item

        """
        # User may have extracted element data and applied custom calcs beforehand.
        if element_data_2d is None:
            # Pass required geometry information
            dfsu_geo = {}
            dfsu_geo["elements_2d"] = self.geo2d["elements"]
            dfsu_geo["element_ids_horz"] = self.element_ids_horz
            dfsu_geo["element_ids_vert"] = self.element_ids_vert

            dfsu_object = dfs.DfsFileFactory.DfsuFileOpen(self.filename)

            element_data_2d = _extract_2D_element_data(
                dfsu_object,
                item_name,
                self.items,
                dfsu_geo,
                tstep_start=tstep_start,
                tstep_end=tstep_end,
                layers=layers,
            )

            dfsu_object.Close()

        # Initialise array
        element_perc_exc_data = np.zeros(
            (element_data_2d.shape[0], len(thresholds)), dtype=float
        )

        # Find and sum occurrences when value exceeds threshold.
        for c, t in enumerate(thresholds):
            abv_thr = np.sum([element_data_2d >= t], axis=2)
            element_perc_exc_data[:, c] = (
                abv_thr.T[:, 0] / element_data_2d.shape[1] * 100
            )

        # Return either element data or convert to node if specified
        if node:
            # Max element item to node
            node_perc_exc_data = _map_ele_to_node(
                node_table=self.node_table,
                elements=self.elements,
                nodes=self.nodes,
                element_data=element_perc_exc_data,
            )

            return node_perc_exc_data
        else:
            return element_perc_exc_data

    def create_3D_dfsu(
        self, items, output_dfsu, start_datetime=None, timestep=None,
    ):
        """
        Create a new 3D `dfsu` file based on the underlying :class:`Dfsu()` for
        some new non-temporal or temporal layer.

        Parameters
        ----------
        items : dictionary
            Dict of items to write to dfsu.
            Item data and info is stored as a sub-dict of the items dict. 
            
            Item and unit information can be assigned to the dictionary. 
            If left blank, items will be assigned as type SurfaceElevation (units: m)
            
            All item data should have the same number of timesteps. 
            
            Example: 
                items = {'Surface Elevation': 
                    {
                        'arr': element_data, 
                        'item_type': `units.get_item("SurfaceElevation")`, 
                        'unit_type': `units.get_unit("meter")`,
                    }

            Dict-keys : str
                Name of items to write to `dfsu`
            
            Sub-Dict Keys and Values:
            arr : ndarray, shape (num_elements, num_timesteps)
                Array to write to dfsu file. Number of rows must equal the number
                of elements in the :class:`Dfsu()` object and the order of the
                array must align with the order of the elements. Can create a
                non-temporal `dfsu` layer of a single dimensional input `arr`, or a
                temporal `dfsu` layer at `timestep` from `start_datetime`.
            item_type: str
                MIKE21 item code. See :func:`get_item() <dhitools.units.get_item>`.
                Default is "SurfaceElevation"
            unit_type : str
                MIKE21 unit code. See :func:`get_unit() <dhitools.units.get_unit>`.
                Default is "meter" unit
        output_dfsu : str
            Path to output .dfsu file
        start_datetime : datetime
            Start datetime (datetime object). If `None`, use the base
            :class:`Dfsu()` `start_datetime`.
        timestep : float
            Timestep delta in seconds. If `None`, use the base
            :class:`Dfsu()` `timestep`.

        Returns
        -------
        Creates a new dfsu file at output_dfsu : dfsu file
        output_dfsu : str
            Path to output .dfsu file
        start_datetime : datetime
            Start datetime (datetime object). If `None`, use the base
            :class:`Dfsu()` `start_datetime`.
        timestep : float
            Timestep delta in seconds. If `None`, use the base
            :class:`Dfsu()` `timestep`.
        item_type : str
            MIKE21 item code. See :func:`get_item() <dhitools.units.get_item>`.
            Default is "Mannings M"
        unit_type : str
            MIKE21 unit code. See :func:`get_unit() <dhitools.units.get_unit>`.
            Default is "Mannings M" unit "cube root metre per second"

        Returns
        -------
        Creates a new dfsu file at output_dfsu : dfsu file

        """
        dim = self.geo2d["elements"].shape
        for v in items.values():
            assert (
                v["arr"].shape[0] == dim[0]
            ), "Rows of input array must equal number of mesh elements"

        dfs_obj = dfs.DfsFileFactory.DfsuFileOpen(self.filename)
        builder = dfs.dfsu.DfsuBuilder.Create(dfs.dfsu.DfsuFileType.Dfsu2D)

        # Create mesh nodes
        node_x = Array[System.Double](self.nodes[:, 0])
        node_y = Array[System.Double](self.nodes[:, 1])
        node_z = Array[System.Single](self.nodes[:, 2])
        node_id = Array[System.Int32](self.node_boundary_codes)

        # Element table
        element_table = Array.CreateInstance(System.Int32, self.num_elements, 3)

        for i in range(self.num_elements):
            for j in range(dim[1]):
                element_table[i, j] = self.element_table[i, j]

        # Set dfsu items
        builder.SetNodes(node_x, node_y, node_z, node_id)
        builder.SetElements(dfs_obj.ElementTable)
        builder.SetProjection(dfs_obj.Projection)

        # Start datetime and time step
        if start_datetime is not None:
            sys_dt = System.DateTime(
                start_datetime.year,
                start_datetime.month,
                start_datetime.day,
                start_datetime.hour,
                start_datetime.minute,
                start_datetime.second,
            )
        else:
            sys_dt = dfs_obj.StartDateTime

        if timestep is None:
            timestep = dfs_obj.TimeStepInSeconds
        builder.SetTimeInfo(sys_dt, timestep)

        # Create items
        for item_name, v in items.items():
            builder.AddDynamicItem(
                item_name, eumQuantity(v["item_type"], v["unit_type"])
            )

        # Create file
        dfsu_file = builder.CreateFile(output_dfsu)

        # Stack arrays by a third dimension (nelements x ntimesteps x nvariables)
        arr = np.dstack([v["arr"] for v in items.values()])

        # Write item data
        ntimesteps = arr.shape[1]
        nvariables = arr.shape[2]
        for j in range(ntimesteps):
            for k in range(nvariables):
                net_arr = Array.CreateInstance(System.Single, dim[0])
                for i, val in enumerate(arr[:, j, k]):
                    net_arr[i] = val
                dfsu_file.WriteItemTimeStepNext(0, net_arr)

        # Close file
        dfsu_file.Close()

    def create_2D_dfsu(
        self, items, output_dfsu, start_datetime=None, timestep=None,
    ):
        """
        Create a new `dfsu` file based on the underlying :class:`Dfsu()` for
        some new non-temporal or temporal layer.

        Parameters
        ----------
        items : dictionary
            Dict of items to write to dfsu.
            Item data and info is stored as a sub-dict of the items dict. 
            
            Item and unit information can be assigned to the dictionary. 
            If left blank, items will be assigned as type SurfaceElevation (units: m)
            
            All item data should have the same number of timesteps. 
            
            Example: 
                items = {'Surface Elevation': 
                    {
                        'arr': element_data, 
                        'item_type': units.get_item("SurfaceElevation"), 
                        'unit_type': units.get_unit("meter"),
                    }

            Dict-keys : str
                Name of items to write to `dfsu`
            
            Sub-Dict Keys and Values:
            arr : ndarray, shape (num_elements, num_timesteps)
                Array to write to dfsu file. Number of rows must equal the number
                of elements in the :class:`Dfsu()` object and the order of the
                array must align with the order of the elements. Can create a
                non-temporal `dfsu` layer of a single dimensional input `arr`, or a
                temporal `dfsu` layer at `timestep` from `start_datetime`.
            item_type: str
                MIKE21 item code. See :func:`get_item() <dhitools.units.get_item>`.
                Default is "SurfaceElevation"
            unit_type : str
                MIKE21 unit code. See :func:`get_unit() <dhitools.units.get_unit>`.
                Default is "meter" unit
        output_dfsu : str
            Path to output .dfsu file
        start_datetime : datetime
            Start datetime (datetime object). If `None`, use the base
            :class:`Dfsu()` `start_datetime`.
        timestep : float
            Timestep delta in seconds. If `None`, use the base
            :class:`Dfsu()` `timestep`.

        Returns
        -------
        Creates a new dfsu file at output_dfsu : dfsu file

        """
        dim = self.geo2d["elements"].shape
        for v in items.values():
            assert (
                v["arr"].shape[0] == dim[0]
            ), "Rows of input array must equal number of mesh elements"

        dfs_obj = dfs.DfsFileFactory.DfsuFileOpen(self.filename)
        builder = dfs.dfsu.DfsuBuilder.Create(dfs.dfsu.DfsuFileType.Dfsu2D)

        # Create mesh nodes
        node_x = Array[System.Double](self.geo2d["nodes"][:, 0])
        node_y = Array[System.Double](self.geo2d["nodes"][:, 1])
        node_z = Array[System.Single](self.geo2d["nodes"][:, 2])
        node_id = Array[System.Int32](self.geo2d["node_boundary_code"].T[:, 0])
        elmt_table = self.geo2d["element_table"].astype(System.Int32)

        # Set dfsu items
        builder.SetNodes(node_x, node_y, node_z, node_id)
        builder.SetElements(elmt_table)
        builder.SetProjection(dfs_obj.Projection)

        # Start datetime and time step
        if start_datetime is not None:
            sys_dt = System.DateTime(
                start_datetime.year,
                start_datetime.month,
                start_datetime.day,
                start_datetime.hour,
                start_datetime.minute,
                start_datetime.second,
            )
        else:
            sys_dt = dfs_obj.StartDateTime

        if timestep is None:
            timestep = dfs_obj.TimeStepInSeconds
        builder.SetTimeInfo(sys_dt, timestep)

        # Create items
        for item_name, v in items.items():
            builder.AddDynamicItem(
                item_name, eumQuantity(v["item_type"], v["unit_type"])
            )

        # Create file
        dfsu_file = builder.CreateFile(output_dfsu)

        # Stack arrays by a third dimension (nelements x ntimesteps x nvariables)
        arr = np.dstack([v["arr"] for v in items.values()])

        # Write item data
        ntimesteps = arr.shape[1]
        nvariables = arr.shape[2]
        for j in range(ntimesteps):
            for k in range(nvariables):
                net_arr = Array.CreateInstance(System.Single, dim[0])
                for i, val in enumerate(arr[:, j, k]):
                    net_arr[i] = val
                dfsu_file.WriteItemTimeStepNext(0, net_arr)

        # Close file
        dfsu_file.Close()


def _dfsu_info(dfsu_object):
    """
    Make a dictionary with .dfsu items and other attributes.

    See class attributes
    """
    itemnames = [[n.Name, n.Quantity.UnitAbbreviation] for n in dfsu_object.ItemInfo]
    items = {}

    for ind, it in enumerate(itemnames):

        # Create key from itemname and add to dictionary
        itemName = str(it[0])
        itemUnit = str(it[1])
        items[itemName] = {}
        items[itemName]["unit"] = itemUnit
        items[itemName]["index"] = ind

    items["num_timesteps"] = dfsu_object.NumberOfTimeSteps
    items["num_nodes"] = dfsu_object.NumberOfNodes
    items["num_elements"] = dfsu_object.NumberOfElements

    dfsu_object.Close()

    return items


"""
Read item node and element data
"""


def _3d_element_geo(dfs_obj, element_table):
    """
    Calculate 3D dfsu geometry properties
    """

    # Check if mesh is triangular or quad/tri mix
    if element_table.shape[0] % 3 == 0:
        cid = 3
    else:
        cid = 4

    # Iterate through each element and determine if the next element is
    # in the same position, or is a new 2D element
    MaxLyr = dfs_obj.NumberOfLayers
    ele_ids_horz = [1]
    layer_count = [MaxLyr]
    ele_ids_surf = [False]
    e_last = element_table[0]

    for c, e in enumerate(element_table[1:], 1):
        if e[0] == e_last[cid]:
            ele_ids_horz.append(ele_ids_horz[c - 1])
            layer_count.append(layer_count[c - 1] - 1)
            ele_ids_surf.append(False)
            e_last = e
        else:
            ele_ids_horz.append(ele_ids_horz[c - 1] + 1)
            layer_count.append(MaxLyr)
            ele_ids_surf.append(False)
            ele_ids_surf[c - 1] = True
            e_last = e
    ele_ids_surf[-1] = True

    # Count layers and assign a vertical layer number
    layer_count = layer_count[::-1]
    ele_ids_vert = [MaxLyr]
    for c, l in enumerate(layer_count[1:], 1):
        diff = l - layer_count[c - 1]
        if diff == 1:
            ele_ids_vert.append(ele_ids_vert[c - 1] - 1)
        elif diff <= 0:
            ele_ids_vert.append(MaxLyr)
    ele_ids_vert = ele_ids_vert[::-1]

    return ele_ids_horz, ele_ids_vert, ele_ids_surf


def _calc_2d_geo(dfs_obj, elements, nodes, node_ids, element_table, ele_ids_surf):
    """
    Generate a dict with equivalent 2D mesh geometry. 
    Used for 2D plotting and for building 2D dfsu files. 
    """

    # Find all unique 2D elements and extract x-y positions.
    if element_table.shape[1] % 3 == 0:
        element_table_surf = element_table[ele_ids_surf][:, 3:] - 1
    else:
        element_table_surf = element_table[ele_ids_surf][:, 4:] - 1

    nodes_top = np.unique(element_table_surf)
    nodes_top[nodes_top == 0] = []

    NodeCols = element_table_surf.shape[1]
    EleRows = element_table_surf.shape[0]

    node_boundary_code = np.zeros((1, nodes_top.shape[0]), dtype=int)
    nodes_2d = nodes[nodes_top]
    nodes_ids_2d = node_ids[nodes_top]
    ele_2d = elements[ele_ids_surf]

    ele_table_2D = np.ones((EleRows, NodeCols), dtype=int)
    for ne in np.arange(0, EleRows, 1):
        for nc in np.arange(0, NodeCols, 1):
            ele_table_2D[ne, nc] = np.where(nodes_top == element_table_surf[ne, nc])[0][
                0
            ]
    ele_table_2D = ele_table_2D + 1

    node_table = mesh._node_table(ele_table_2D)

    geo2d = {
        "nodes": nodes_2d,
        "node_id": nodes_ids_2d,
        "node_boundary_code": node_boundary_code,
        "node_table": node_table,
        "element_table": ele_table_2D,
        "element_ids": np.arange(1, ele_table_2D.shape[0] + 1, 1, dtype=int),
        "elements": ele_2d,
        "proj": str(dfs_obj.Projection.WKTString),
        "zUnitKey": dfs_obj.get_ZUnit(),
    }
    return geo2d


def _2D_element_table(self):
    """
    Calculates an equivalent element table by converting quad elements
    to 2 triangular elements
    
    Used for rendering tricontourf plots
    """

    tn3D_top = self.element_table[self.element_ids_surf]
    tn_top = np.zeros((len(tn3D_top), 4))

    q2D = tn3D_top[:, 6] > 0
    tn_top[q2D, :] = tn3D_top[q2D, 4:]
    tn_top[~q2D, 0:3] = tn3D_top[~q2D, 3:6]

    return tn_top


def _element_data(
    dfsu_object,
    item_name,
    item_info,
    tstep_start=None,
    tstep_end=None,
    element_list=None,
):
    """ Read specified item_name element data"""
    if element_list:
        # Subtract zero to match Python idx'ing
        element_list = [e - 1 for e in element_list]

    item_idx = item_info[item_name]["index"] + 1

    if tstep_start is None:
        tstep_start = 0

    if tstep_end is None:
        # Only get one tstep specified by tstep_start
        tstep_end = tstep_start + 1
    elif tstep_end == -1:
        # Get from tstep_start to the end
        tstep_end = item_info["num_timesteps"]
    else:
        # Add one to include tstep_end in output
        tstep_end += 1

    t_range = range(tstep_start, tstep_end)
    if element_list:
        ele_data = np.zeros(shape=(len(element_list), len(t_range)))
    else:
        ele_data = np.zeros(shape=(item_info["num_elements"], len(t_range)))
    for i, t in enumerate(t_range):
        if element_list:
            ele_data[:, i] = _utils.dotnet_arr_to_ndarr(
                dfsu_object.ReadItemTimeStep(item_idx, t).Data
            )[element_list]
        else:
            ele_data[:, i] = _utils.dotnet_arr_to_ndarr(
                dfsu_object.ReadItemTimeStep(item_idx, t).Data
            )

    return ele_data


def _node_data(
    dfsu_object,
    item_name,
    item_info,
    ele_cords,
    node_cords,
    node_table,
    tstep_start=None,
    tstep_end=None,
):
    """ Read specified item_name node data """

    # Get item_name element data
    ele_data = _element_data(dfsu_object, item_name, item_info, tstep_start, tstep_end)

    # Get item_name node data
    node_data = np.zeros(shape=(len(node_cords), ele_data.shape[1]))
    for i in range(ele_data.shape[1]):
        node_data[:, i] = _map_ele_to_node(
            node_table, ele_cords, node_cords, ele_data[:, i]
        )

    return node_data


def _interp_node_z(nn, node_table, xe, ye, ze, xn, yn):
    """
    Calculate value at node (xn,yn) from element center values (xe, ye, ze).

    Attempts to use Psuedo Laplace procedure by [Holmes, Connel 1989]. If this
    fails, uses an inverse distance average.

    Parameters
    ----------
    nn : int
        Node number to solve node value.
    node_table : ndarray, shape (num_nodes, n)
        Defines for each node the element adjacent to this node. May contain
        padded zeros
    xe : ndarray, shape (num_elements, 1)
        Element x vector
    ye : ndarray, shape (num_elements, 1)
        Element y vector
    ze : ndarray, shape (num_elements, 1)
        Element x vector
    xn : ndarray, shape (num_nodes, 1)
        Node x vector
    yn : ndarray, shape (num_nodes, 1)
        Node x vector

    Returns
    -------
    weights : array, shape (n_components,)

    See Also
    -------
    DHI MIKE MATLAB Toolbox; specifically `mzCalcNodeValuePL.m`

    Holmes, D. G. and Connell, S. D. (1989), Solution of the
        2D Navier-Stokes on unstructured adaptive grids, AIAA Pap.
        89-1932 in Proc. AIAA 9th CFD Conference.
    """
    nelmts = len(np.where(node_table[nn, :] != 0)[0])

    if nelmts < 1:
        zn = np.nan
        return zn

    Rx = 0
    Ry = 0
    Ixx = 0
    Iyy = 0
    Ixy = 0

    for i in range(nelmts):
        el_id = int(node_table[nn, i] - 1)
        dx = xe[el_id] - xn[nn]
        dy = ye[el_id] - yn[nn]
        Rx = Rx + dx
        Ry = Ry + dy
        Ixx = Ixx + dx * dx
        Iyy = Iyy + dy * dy
        Ixy = Ixy + dx * dy

    lamda = Ixx * Iyy - Ixy * Ixy

    # Pseudo laplace procedure
    if abs(lamda) > 1e-10 * (Ixx * Iyy):
        lamda_x = (Ixy * Ry - Iyy * Rx) / lamda
        lamda_y = (Ixy * Rx - Ixx * Ry) / lamda

        omega_sum = float(0)
        zn = float(0)

        for i in range(nelmts):
            el_id = int(node_table[nn, i] - 1)

            omega = 1 + lamda_x * (xe[el_id] - xn[nn]) + lamda_y * (ye[el_id] - yn[nn])
            if omega < 0:
                omega = 0
            elif omega > 2:
                omega = 2
            omega_sum = omega_sum + omega
            zn = zn + omega * ze[el_id]

        if abs(omega_sum) > 1e-10:
            zn = zn / omega_sum
        else:
            omega_sum = float(0)
    else:
        omega_sum = float(0)

    # If not successful use inverse distance average
    if omega_sum == 0:
        zn = 0

        for i in range(nelmts):
            el_id = int(node_table[nn, i] - 1)

            dx = xe[el_id] - xn[nn]
            dy = ye[el_id] - yn[nn]

            omega = float(1) / np.sqrt(dx * dx + dy * dy)
            omega_sum = omega_sum + omega
            zn = zn + omega * ze[el_id]

        if omega_sum != 0:
            zn = zn / omega_sum
        else:
            zn = float(0)

    return zn


def _map_ele_to_node(node_table, elements, nodes, element_data):
    """
    Get node data relating to specific element
    """

    xn = nodes[:, 0]
    yn = nodes[:, 1]
    xe = elements[:, 0]
    ye = elements[:, 1]

    zn = np.zeros(len(xn))

    for i in range(len(xn)):
        zn[i] = _interp_node_z(i, node_table, xe, ye, element_data, xn, yn)

    return zn


def tritables(element_table):
    """
    Build connection tables for tri/quad meshes.
    
    Need to implement connectivitiy tables. 
    
    # Note, element table is already python indexed
    """

    nelmts = element_table.shape[0]
    nnodes = element_table.max() + 1

    if element_table.shape[1] % 3 == 0:
        hasquads = False
        quads = np.zeros((len(element_table), 1)).astype(bool)
    else:
        hasquads = True
        quads = element_table[:, 3] + 1 > 0

    e = np.arange(1, nelmts + 1)
    u = np.ones((nelmts, 1))
    I = np.asarray((e, e, e)).ravel()
    J = element_table[:, :3].ravel("F")
    K = np.asarray((u, u * 2, u * 3)).ravel("F")

    if hasquads:
        I = np.append(I, e[quads])
        J = np.append(J, element_table[quads, 3])
        K = np.append(K, 4 * u[quads])

    # Make Node-to-Element table
    if hasquads:
        NtoE = np.zeros((nnodes, 4), dtype=int)
    else:
        NtoE = np.zeros((nnodes, 3), dtype=int)
    count = np.zeros((nnodes, 1), dtype=int)

    for i in range(len(I)):
        count[J[i]] = count[J[i]] + 1
        NtoE[J[i], count[J[i]] - 1] = I[i]

    return NtoE


"""
dfsu stats
"""


def _item_aggregate_stats(
    dfsu_object,
    item_name,
    item_info,
    return_max=True,
    tstep_start=None,
    tstep_end=None,
    current_dir=False,
):
    """
    Return max or min for input item across entire model or specific time range
    """
    item_idx = item_info[item_name]["index"] + 1
    ele_data = np.zeros((item_info["num_elements"]))

    # If current_dir provided, get current dir at input item_name max/min
    if current_dir:
        cd_index = item_info["Current direction"]["index"] + 1
        cd_ele_data = np.zeros((item_info["num_elements"]))

    # Sort time range
    if tstep_start is None:
        tstep_start = 0

    if tstep_end is None:
        # Get from tstep_start to the end
        tstep_end = item_info["num_timesteps"]
    else:
        # Add one to include tstep_end in output
        tstep_end += 1

    for tstep in range(tstep_start, tstep_end):
        # Iterate tstep in time range
        item_data = _utils.dotnet_arr_to_ndarr(
            dfsu_object.ReadItemTimeStep(item_idx, tstep).Data
        )

        # Determine elements to update
        if return_max:
            comp_boolean = np.greater(item_data, ele_data)
        else:
            comp_boolean = np.less(item_data, ele_data)

        # Update elements which have new extreme
        update_elements = item_data[comp_boolean]
        ele_data[comp_boolean] = update_elements

        # Update current_dir if specified
        if current_dir:
            cd_data = _utils.dotnet_arr_to_ndarr(
                dfsu_object.ReadItemTimeStep(cd_index, tstep).Data
            )
            update_cd_elements = cd_data[comp_boolean]
            cd_ele_data[comp_boolean] = update_cd_elements

    if current_dir:
        # Return both item_name data and current_dir data
        return ele_data, cd_ele_data
    else:
        # Else just item_name data
        return ele_data


def _extract_2D_element_data(
    dfsu_object,
    item_name,
    item_info,
    dfsu_geo,
    tstep_start=None,
    tstep_end=None,
    layers=None,
    ele_data=None,
):
    """
    Function to extract element data and take maximum throughout the specified layers.
    
    If Layers = None, the maximum value is taken throughout the water column.
    """

    if ele_data is not None:
        ele_2d_data = ele_data
    else:
        ele_data = np.zeros((item_info["num_elements"]))

        # Sort time range
        if tstep_start is None:
            tstep_start = 0

        if tstep_end is None:
            # Get from tstep_start to the end
            tstep_end = item_info["num_timesteps"]
        else:
            # Add one to include tstep_end in output
            tstep_end += 1

        # Load the element data
        if layers is not None:

            # Determine which elements belong to the specified layer(s)
            element_list = []
            for l in layers:
                element_list.extend(
                    list(np.where(l == np.asarray(dfsu_geo["element_ids_vert"]))[0])
                )

            if type(layers) is int:
                ele_2d_data = _element_data(
                    dfsu_object,
                    item_name,
                    item_info,
                    tstep_start=tstep_start,
                    tstep_end=tstep_end - 1,
                    element_list=element_list,
                )

            elif type(layers) is list:
                ele_data = _element_data(
                    dfsu_object,
                    item_name,
                    item_info,
                    tstep_start=tstep_start,
                    tstep_end=tstep_end - 1,
                    element_list=element_list,
                )

                # Initialise the 2-D Element data array (1 x ntimesteps)
                ele_2d_data = np.zeros(
                    (dfsu_geo["elements_2d"].shape[0], ele_data.shape[1]), dtype=float
                )

                # stack each layer of data.
                # (Note, change this to np array for faster processing next time)
                nn = []
                for l in layers:
                    nn.extend(
                        np.asarray(dfsu_geo["element_ids_horz"])[
                            (np.asarray(dfsu_geo["element_ids_vert"]) == 1)
                        ].tolist()
                    )

                # For each 2D cell, take the maximum in each stack of data.
                for c, n in enumerate(np.arange(1, ele_2d_data.shape[0] + 1, 1)):
                    ids = n == nn
                    ele_2d_data[c, :] = np.max(ele_data[ids, :], axis=0)

        else:  # No layers specified. Extract all data and take max.
            ele_data = _element_data(
                dfsu_object,
                item_name,
                item_info,
                tstep_start=tstep_start,
                tstep_end=tstep_end - 1,
            )

            # Initialise the 2-D Element data array (1 x ntimesteps)
            ele_2d_data = np.zeros(
                (dfsu_geo["elements_2d"].shape[0], ele_data.shape[1]), dtype=float
            )

            # For each 2D cell, take the maximum in each stack of data.
            nn = np.array(dfsu_geo["element_ids_horz"])
            for c, n in enumerate(np.arange(1, ele_2d_data.shape[0] + 1, 1)):
                ids = n == nn
                ele_2d_data[c, :] = np.max(ele_data[ids, :], axis=0)

    return ele_2d_data
