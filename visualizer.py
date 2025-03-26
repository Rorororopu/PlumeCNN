'''
Using Matplotlib to visualize graphs, output them to PNG image.
'''
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from pathlib import Path # Check if the path exist
import pandas as pd
import pyvista as pv
import scipy.interpolate
import tqdm # Dyamically show the progress
import typing
import vtk

import analyzer


def get_directory_path(param_name:str) -> str:
    '''
    Prompt the user to enter the path of the directory to open repeatedly until a valid path is provided.

    Args:
        param_name: The name of params to plot. Just for the ease of visulization.

    Returns:
        str: The inputted path to the directory which has been verified to exist.
    '''
    while True:
        path = input(f"What is the path of the directory you want to store your image of {param_name}?"
                    "\nFor instance, if you want to store your image as 'image.png' to the curent directory,"
                    "you could input './image.png")
        if Path(path).is_dir():
            print("Success.")
            return path  # Return the valid directory path
        else:
            print("\033[91mInvalid directory path. No such directory exists. Please try again.\033[0m") #In red


def plot_2D_df(df: pd.DataFrame, param_name:str, path:str=None, cmap:str='viridis', range:list=None, transparent:bool=False) -> typing.Tuple[plt.Figure, plt.Axes]:
    '''
    Generate a 2D scatter plot from a specified parameter in a pandas table. The plot
    includes a color gradient to represent values, complete with a title, axis labels, and a color bar.
    The resulting plot is saved to the specified file path.

    Because if you input sliced data of visit, 
    although the slicing direction may not be z, visit will still convert coordinates into x and y,
    so the variable to plot will always be x and y, regardless of slicing direction.

    Args:
    df: The Pandas dataframe containing coordinates and data.
    param_name: name of the param to plot.
    path: The path to store the image. If none, the image won't be saved.
    cmap: The colormap of the image. Choices include:
        'viridis': Ranges from dark blue to bright yellow. Recommended for data ranging from [0,1].
        'coolwarm': Ranges from deep blue, white to deep red. Recommended for data ranging from [-1,1].
    range: Optional. The range of data shown in the plot.
        If specified, values outside this range will be capped to the range limits.
        If None, the full range of the data will be used.
    transparent: Boolean. If True, the space outside the scatter plot will be transparent.
        If False, it will be white.

    Returns:
    fig: The matplotlib Figure object
    ax: The matplotlib Axes object
    '''
    fig, ax = plt.subplots(figsize=(10, 8))

    color_data = df[param_name]
    if range is not None:
        sc = ax.scatter(df['x'], df['y'], c=color_data, cmap=cmap, vmin=range[0], vmax=range[1])
    else:
        sc = ax.scatter(df['x'], df['y'], c=color_data, cmap=cmap)

    cbar = plt.colorbar(sc, label=param_name, ax=ax)
    ax.set_title(f'Plot of {param_name}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Set transparency
    if transparent:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
    else:
        fig.patch.set_alpha(1.0)
        ax.patch.set_facecolor('white')

    if path is not None:
        plt.savefig(path, transparent=transparent)
        plt.close(fig)
        return None
    else:
        return fig, ax


def plot_3D_to_2D_slice_df(df: pd.DataFrame, direction: str, param_name: str, path: str = None, cmap: str = 'viridis', data_range: list = None, axis_limits: list = None, transparent: bool = False) -> typing.Tuple[plt.Figure, plt.Axes]:
    '''
    Generate a 2D scatter plot from a specified parameter in a pandas table. The plot
    includes a color gradient to represent values, complete with a title, axis labels, and a color bar.
    The resulting plot is saved to the specified file path.

    Because if you get data slice from MY PlumeCNN code, the coordinate won't change in the table,
    so we need to specify which two coordinates are needed to plot.

    Args:
    df: The Pandas dataframe containing coordinates and data.
    direction: The slicing direction, it can be 'x', 'y' or 'z'. e.g. if the slicing direction is perpendicular to z axis, please input 'z'.
    param_name: name of the param to plot.
    path: The path to store the image. If none, the image won't be saved.
    cmap: The colormap of the image. Choices include:
        'viridis': Ranges from dark blue to bright yellow. Recommended for data ranging from [0,1].
        'coolwarm': Ranges from deep blue, white to deep red. Recommended for data ranging from [-1,1].
    data_range: Optional. The range of data shown in the plot.
        If specified, values outside this range will be capped to the range limits.
        If None, the full range of the data will be used.
    axis_limits: Optional. A list of axis limits for the plot in the form [x_min, x_max, y_min, y_max].
        If None, the axis limits will be determined automatically from the data.
        This is needed because slices from the edge of the cylinder will only occupy a small section of the whole region calculated, and we want to have a clearer idea of how big they are.
    transparent: Boolean. If True, the space outside the scatter plot will be transparent.
        If False, it will be white.

    Returns:
    fig: The matplotlib Figure object
    ax: The matplotlib Axes object
    '''
    fig, ax = plt.subplots(figsize=(10, 8))

    # Determine which coordinates to plot based on the direction
    if direction == 'x':
        coords = ['y', 'z']
    elif direction == 'y':
        coords = ['x', 'z']
    elif direction == 'z':
        coords = ['x', 'y']
    
    # Extract color data
    color_data = df[param_name]
    
    # Plot data with colormap and range
    if data_range is not None:
        sc = ax.scatter(df[coords[0]], df[coords[1]], c=color_data, cmap=cmap, vmin=data_range[0], vmax=data_range[1])
    else:
        sc = ax.scatter(df[coords[0]], df[coords[1]], c=color_data, cmap=cmap)

    # Set colorbar
    cbar = plt.colorbar(sc, label=param_name, ax=ax)

    # Set axis limits if provided
    if axis_limits is not None:
        ax.set_xlim(axis_limits[0], axis_limits[1])
        ax.set_ylim(axis_limits[2], axis_limits[3])
    else:
        # Use auto-scaling if no axis limits are provided
        ax.relim()
        ax.autoscale()

    # Set axis labels and title
    ax.set_title(f'Plot of {param_name}')
    ax.set_xlabel(coords[0])
    ax.set_ylabel(coords[1])

    # Set transparency
    if transparent:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
    else:
        fig.patch.set_alpha(1.0)
        ax.patch.set_facecolor('white')

    # Save or return the figure
    if path is not None:
        plt.savefig(path, transparent=transparent)
        plt.close(fig)
        return None
    else:
        return fig, ax


def arrange_slice_df(input_file: str, direction:str, output_df: str = None, cubic: bool = True) -> pd.DataFrame:
    '''
    Arrange sliced data to prepare for plotting streamlines.
    Fill empty rows with coordinates to ensure a complete grid of points in a square.
    For NaN points outside the container, set velocity to 0.
    For NaN points inside the container, interpolate velocity as needed.

    Note that because even data for 3D are stored in many slices, so it is necessary to specify a slicing direction.
    Args:
        input_file: The path of the SLICED CSV file (preprocessed by our PlumeCNN).
        Direction: Specify the axis to which the slice is perpendicular ('x', 'y', or 'z'). 
        output_file: If you want to output the arranged data to CSV, please input the desired path of the output file.
            If None(in default), it will not write any file out.
        if_cubic: If True (default), the function will process coordinates and velocity components in all directions.
            If false, Only coordinates and velocity components in the remaining directions will be considered. 
            The reason why this variable is called 'cubic' is because only a slice representing for a whole cubic data needs this option.
           
    Returns:The arranged pandas dataframe, containing coordinate values and velocities in desired directions.
    '''
    df = pd.read_csv(input_file)
    # Read and extract columns to read
    if direction == 'z':
        coord1 = 'x'
        coord2 = 'y'
        coord3 = 'z'
        vel1 = 'x_velocity'
        vel2 = 'y_velocity'
        vel3 = 'z_velocity'
    elif direction == 'y':
        coord1 = 'x'
        coord2 = 'z'
        coord3 = 'y'
        vel1 = 'x_velocity'
        vel2 = 'z_velocity'
        vel3 = 'y_velocity'
    elif direction == 'x':
        coord1 = 'y'
        coord2 = 'z'
        coord3 = 'x'
        vel1 = 'y_velocity'
        vel2 = 'z_velocity'
        vel3 = 'x_velocity'

    if cubic is True:
        df = df[['x', 'y', 'z', 'x_velocity', 'y_velocity', 'z_velocity']].dropna()
    else:
        df = df[[coord1, coord2, vel1, vel2]].dropna()

    # Extract unique coordinate values, sorted in ascending order
    coord1_values = np.sort(df[coord1].unique())
    coord2_values = np.sort(df[coord2].unique())
    # define grid points in the specified plane for interpolation
    grid_coord1, grid_coord2 = np.meshgrid(coord1_values, coord2_values, indexing='ij')
    # Prepare data points and velocity components for interpolation
    points = df[[coord1, coord2]].values
    u_values = df[vel1].values
    v_values = df[vel2].values
    if cubic is True:
        w_values = df[vel3].values
    # Interpolate velocity components onto the grid, to eliminate NaN points lying in the dataset
    '''without this step, NaN points in the grid will make the streamline extremely short'''
    u_grid = scipy.interpolate.griddata(points, u_values, (grid_coord1, grid_coord2), method='linear')
    v_grid = scipy.interpolate.griddata(points, v_values, (grid_coord1, grid_coord2), method='linear')
    if cubic is True:
        w_grid = scipy.interpolate.griddata(points, w_values, (grid_coord1, grid_coord2), method='linear')
    # Replace any NaN values in interpolated data with zeros, to specify the border of the valid data
    u_grid = np.nan_to_num(u_grid, nan=0.0).ravel(order='F')
    v_grid = np.nan_to_num(v_grid, nan=0.0).ravel(order='F')
    if cubic is True:
        w_grid = np.nan_to_num(w_grid, nan=0.0).ravel(order='F')
    
    # Prepare the final DataFrame for return
    if direction == 'z':
        result_df = pd.DataFrame({
            'x': grid_coord1.ravel(order='F'),
            'y': grid_coord2.ravel(order='F'),
            'x_velocity': u_grid,
            'y_velocity': v_grid
        })
        if cubic is True:
            result_df['z'] = df[coord3].unique().mean()
            result_df['z_velocity'] = w_grid
    elif direction == 'y':
        result_df = pd.DataFrame({
            'x': grid_coord1.ravel(order='F'),
            'z': grid_coord2.ravel(order='F'),
            'x_velocity': u_grid,
            'z_velocity': v_grid
        })
        if cubic is True:
            result_df['y'] = df[coord3].unique().mean()
            result_df['y_velocity'] = w_grid
    elif direction == 'x':
        result_df = pd.DataFrame({
            'y': grid_coord1.ravel(order='F'),
            'z': grid_coord2.ravel(order='F'),
            'y_velocity': u_grid,
            'z_velocity': v_grid
        })
        if cubic is True:
            result_df['x'] = df[coord3].unique().mean()
            result_df['x_velocity'] = w_grid
    # Save to CSV if output path is provided
    if output_df is not None:
        result_df.to_csv(output_df, index=False)

    # Return the result DataFrame
    return result_df


def plot_3D_to_2D_slice_streamline(input_file: str, output_file: str, direction: str, seed_points_resolution: list, integration_direction: str = 'both', max_time: float = 0.2, terminal_speed: float = 1e-5, cmap: str = 'viridis', axis_limits: list = None):
    '''
    Generate and visualize 2D streamlines from a (sliced) 3D dataset, projected onto a specified plane,
    and export the visualization as an HTML file. The seed points (starting points) of the streamlines are evenly distributed, with the resolution specified by the user.

    Args:
        input_file: Path to the CSV file containing the 3D dataset.
        output_file: Path to the output HTML file for the visualization, sontaining the extension.
        direction: The direction to which the plane is perpendicular ('x', 'y' or 'z').
        seed_points_resolution: Specifies the resolution for distributing seed points in the format [var1_resolution, var2_resolution], where each element controls the density along the respective variable axis.
        integration_direction (optional): Specify whether the streamline is integrated in the upstream or downstream directions (or both). Options are 'both'(default), 'backward', or 'forward'.
        max_time (optional): What is the maximum integration time of a streamline (0.2 in default).
        terminal_speed (optional): When will the integration stop (1e-5 in default).
        cmap (optional): Colormap to use for the visualization. Default is 'viridis'.
        axis_limits (optional): A list of axis limits for the plot in the form [var1_min, var1_max, var2_min, var2_max].
            If None, the axis limits will be determined automatically from the data.
    '''

    # Suppress all VTK warnings and errors
    '''
    If not, we will get the warning "Unable to factor linear system" for every streamline we plot, although the result is quite good.
    '''
    vtk.vtkObject.GlobalWarningDisplayOff()

    # Define coordinate and velocity components based on the specified direction
    if direction == 'z':
        coord1 = 'x'
        coord2 = 'y'
        vel1 = 'x_velocity'
        vel2 = 'y_velocity'
    elif direction == 'y':
        coord1 = 'x'
        coord2 = 'z'
        vel1 = 'x_velocity'
        vel2 = 'z_velocity'
    elif direction == 'x':
        coord1 = 'y'
        coord2 = 'z'
        vel1 = 'y_velocity'
        vel2 = 'z_velocity'
    else:
        raise ValueError("Invalid direction. Choose from 'x', 'y', 'z'.")
    
    # Load and prepare data
    df = arrange_slice_df(input_file, direction, cubic=False)

    # Extract unique coordinate values, sorted in ascending order
    coord1_values = np.sort(df[coord1].unique())
    coord2_values = np.sort(df[coord2].unique())

    # Calculate the 'dx' between coordinate values for grid step size
    d_coord1 = np.diff(coord1_values).mean()
    d_coord2 = np.diff(coord2_values).mean()

    grid_coord1, grid_coord2 = np.meshgrid(coord1_values, coord2_values, indexing='ij')

    # Create a structured 3D grid for visualization
    # Initialize x, y, z coordinates for the structured grid
    if direction == 'z':
        grid = pv.StructuredGrid(grid_coord1, grid_coord2, np.zeros_like(grid_coord1))
    elif direction == 'y':
        grid = pv.StructuredGrid(grid_coord1, np.zeros_like(grid_coord1), grid_coord2)
    elif direction == 'x':
        grid = pv.StructuredGrid(np.zeros_like(grid_coord1), grid_coord1, grid_coord2)

    # Combine velocity components into a single array and assign to the grid
    if direction == 'z':
        velocity = np.column_stack((df[vel1], df[vel2], np.zeros_like(df[vel1])))
    elif direction == 'y':
        velocity = np.column_stack((df[vel1], np.zeros_like(df[vel1]), df[vel2]))
    elif direction == 'x':
        velocity = np.column_stack((np.zeros_like(df[vel1]), df[vel1], df[vel2]))

    grid.point_data['velocity'] = velocity

    # Generate streamlines from seed points (initial points)
    # Define seed points across the flow domain
    seed_coord1, seed_coord2 = np.meshgrid(
        np.linspace(coord1_values[0], coord1_values[-1], seed_points_resolution[0]),
        np.linspace(coord2_values[0], coord2_values[-1], seed_points_resolution[1])
    )
    seed_coord1 = seed_coord1.ravel()
    seed_coord2 = seed_coord2.ravel()

    if direction == 'z':
        x_seed = seed_coord1
        y_seed = seed_coord2
        z_seed = np.zeros_like(x_seed)
    elif direction == 'y':
        x_seed = seed_coord1
        y_seed = np.zeros_like(seed_coord1)
        z_seed = seed_coord2
    elif direction == 'x':
        x_seed = np.zeros_like(seed_coord1)
        y_seed = seed_coord1
        z_seed = seed_coord2

    # Combine coordinates to form seed points
    seed_points = np.column_stack((x_seed, y_seed, z_seed))
    seed = pv.PolyData(seed_points)

    # Calculate streamlines based on velocity field, starting from the seed points
    streamlines = grid.streamlines_from_source(
        source=seed,
        vectors='velocity',
        integration_direction=integration_direction,
        max_time=max_time,
        initial_step_length=0.5*(d_coord1+d_coord2),
        terminal_speed=terminal_speed
    )

    # Calculate and add velocity magnitude as a scalar field on streamlines for coloring
    velocity_vectors = streamlines['velocity']
    velocity_magnitude = np.linalg.norm(velocity_vectors, axis=1)
    streamlines['velocity_magnitude'] = velocity_magnitude

    # Visualize and export streamlines as HTML
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(grid.outline(), color='k')

    # Color the streamlines by velocity magnitude and set up a scalar bar
    plotter.add_mesh(
        streamlines.tube(radius=0.5 * (d_coord1+d_coord2) * 0.5),
        scalars='velocity_magnitude',
        cmap=cmap,  # Use the colormap specified in the function argument
        scalar_bar_args={'title': 'Velocity Magnitude'}
    )
    if direction == 'z':
        # Set the view based on the specified direction
        plotter.view_xy()
    elif direction == 'y':
        plotter.view_xz()
    elif direction == 'x':
        plotter.view_yz()

    # Show grid with axis labels
    plotter.show_grid(
        xtitle='X',
        ytitle='Y',
        ztitle='Z',
        grid='front'  # Display the grid in front of the scene
    )

    # Adjust axis limits by adding an invisible box
    if axis_limits is not None:
        var1_min, var1_max, var2_min, var2_max = axis_limits

        if direction == 'z':
            # For 'z' direction, create a box with x and y limits
            box = pv.Box(bounds=(
                var1_min, var1_max,  # x bounds
                var2_min, var2_max,  # y bounds
                0, 0                # z bounds (since it's a 2D plane at z=0)
            ))
        elif direction == 'y':
            # For 'y' direction, create a box with x and z limits
            box = pv.Box(bounds=(
                var1_min, var1_max,  # x bounds
                0, 0,               # y bounds (since it's a 2D plane at y=0)
                var2_min, var2_max   # z bounds
            ))
        elif direction == 'x':
            # For 'x' direction, create a box with y and z limits
            box = pv.Box(bounds=(
                0, 0,               # x bounds (since it's a 2D plane at x=0)
                var1_min, var1_max,  # y bounds
                var2_min, var2_max   # z bounds
            ))

        # Add the box to the plotter with zero opacity
        plotter.add_mesh(box, opacity=0.0, show_edges=False)

    plotter.export_html(output_file)


def plot_3D_streamline(input_file: str, output_file: str, seed_points_resolution: list, integration_direction: str = 'both', max_time: float = 0.2, terminal_speed: float = 1e-5, cmap: str = 'viridis'):
    '''
    Generate and visualize 3D streamlines from a 3D dataset,
    and export the visualization as an HTML file.

    Args:
        input_folder: Path to the ARRANGED folder containing sliced CSV files of the 3D dataset.
        output_file: Path to the output HTML file for the visualization.
        seed_points_resolution: Specifies the resolution for distributing seed points in the format [x_resolution, y_resolution, z_resolution], where each element controls the density along the respective variable axis.
        integration_direction (optional): Specify whether the streamline is integrated in the upstream or downstream directions (or both). Options are 'both'(default), 'backward', or 'forward'.
        max_time (optional): What is the maximum integration time of a streamline (100 in default).
        terminal_speed (optional): When will the integration stop (1e-5 in default).
        cmap (optional): Colormap to use for the visualization. Default is 'viridis'.
    '''
    pass

# Not applied in code because FFMPG is not installed on the HPC.
def create_2D_movie(data_frames: typing.List[analyzer.Data], param_name: str, path: str, fps: int = 30):
    '''
    Create a movie directly from a series of data with uneven time intervals.

    Because this function needs FFMpeg, and I can't install it on HPC, so it is currently not in use.

    Args:
        data_frames: List of dataframes, one for each frame.
        param_name: name of the param to plot.
        path: Path where the output movie will be saved.
        fps: Frames per second for the output movie.
    '''
    fig, ax = plt.subplots()
    
    # Extract times from data frames
    times = [data.time for data in data_frames]
    
    # Calculate time intervals
    intervals = np.diff(times)
    max_interval = max(intervals)
    
    def animate(i):
        ax.clear()
        plot_2D_df(data_frames[i].df, param_name)
        ax.set_title(f"Time: {times[i]:.2e}")
        return ax.get_children()
    
    # Create the animation (blit = False means redraw every frame from scratch)
    anim = animation.FuncAnimation(fig, animate, frames=len(data_frames), interval=1000/fps, blit=False)
    
    # Calculate frame durations based on actual time intervals
    frame_durations = [interval / max_interval * 1000/fps for interval in intervals]
    frame_durations.append(frame_durations[-1])  # Use the last interval for the final frame
    
    # Save the animation with variable frame durations
    writer = animation.FFMpegWriter(fps=fps)
    try:
        total_frames = len(frame_durations)
        with writer.saving(fig, path, dpi=100):
            for i, duration in enumerate(frame_durations):
                animate(i)  # Call the animate function directly
                writer.grab_frame()
                plt.pause(duration / 1000)  # Convert ms to seconds
                if (i + 1) % 10 == 0 or i + 1 == total_frames:
                    print(f"Progress: {i+1}/{total_frames} frames processed")
        print(f"Movie saved to {path}")
    except Exception as e:
        print(f"An error occurred while saving the movie: {e}")
    finally:
        plt.close(fig)

# Not applied in model prediction, just for experiments
def plot_relevance(df: pd.DataFrame, param1: str, param2: str, path: str):
    '''
    Create a scatterplot showing the correlation between two parameters and save it to a file.
    The plot includes the Pearson correlation coefficient.
    
    Args:
    df: The dataframe containing the data
    param1: The name of the first parameter (x-axis)
    param2: The name of the second parameter (y-axis)
    path: The file path to save the plot
    '''
    # Calculate the Pearson correlation coefficient
    corr_coef = df[param1].corr(df[param2])
    
    # Create the scatterplot
    plt.figure(figsize=(10, 6))
    plt.scatter(df[param1], df[param2], alpha=0.5)
    
    # Add a title and labels
    plt.title(f"Relevance between {param1} and {param2}")
    plt.xlabel(param1)
    plt.ylabel(param2)
    
    # Add correlation coefficient to the plot
    plt.annotate(f'Correlation Coefficient: {corr_coef:.2f}', 
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=12, ha='left', va='top')
    
    # Save the plot
    plt.savefig(path)
    plt.close()

# Not applied because our 3D data is put in separated files, and it's not filling the whole cubical space. 
def plot_3D_streamline_deprecated(input_file: str, output_file: str, seed_points_resolution: list, integration_direction: str = 'both', max_time: float = 0.2, terminal_speed: float = 1e-5, cmap: str = 'viridis'):
    '''
    This function is for read a csv containing all data points in a cubical space and plotting it.
    '''
    # Suppress all VTK warnings and errors
    '''
    If not, we will get the warning "Unable to factor linear system" for every streamline we plot, although the result is quite good.
    '''
    vtk.vtkObject.GlobalWarningDisplayOff()

    '''
    Following codes are not modified to accomodate the arguments of function!
    '''

    # Read the CSV file into a DataFrame
    df = pd.read_csv('test.csv')

    points = df[['x', 'y', 'z']].values
    velocities = df[['x_velocity', 'y_velocity', 'z_velocity']].values

    # Extract unique coordinate values for each axis (ensure they are sorted)
    x_vals = np.sort(df['x'].unique())
    y_vals = np.sort(df['y'].unique())
    z_vals = np.sort(df['z'].unique())

    # Determine grid dimensions
    nx, ny, nz = len(x_vals), len(y_vals), len(z_vals)

    # Reshape coordinates
    x = df['x'].values.reshape((nx, ny, nz))
    y = df['y'].values.reshape((nx, ny, nz))
    z = df['z'].values.reshape((nx, ny, nz))

    # Create the StructuredGrid
    grid = pv.StructuredGrid(x, y, z)

    # Add the velocity vectors
    grid.point_data['velocity'] = velocities

    seed_x, seed_y, seed_z = np.meshgrid(
        np.linspace(-0.5, 0.5, 3), # min, max, num
        np.linspace(-0.5, 0.5, 3),
        np.linspace(-0.5, 0.5, 3)
        )
    seed_x = seed_x.ravel()
    seed_y = seed_y.ravel()
    seed_z = seed_z.ravel()

    seed_points = np.column_stack((seed_x, seed_y, seed_z))
    seed = pv.PolyData(seed_points)

    streamlines = grid.streamlines_from_source(
        source=seed,
        vectors='velocity',
        integration_direction='both',
        max_time=10,
        initial_step_length=0.01,
        terminal_speed=1e-3
    )

    velocity_vectors = streamlines['velocity']
    velocity_magnitude = np.linalg.norm(velocity_vectors, axis=1)
    streamlines['velocity_magnitude'] = velocity_magnitude

    # Visualize and export streamlines as HTML
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(grid.outline(), color='k')


    plotter.add_mesh(
        streamlines.tube(radius=0.01),
        scalars='velocity_magnitude',
        cmap='viridis',  # Use the colormap specified in the function argument
        scalar_bar_args={'title': 'Velocity Magnitude'}
    )

    plotter.view_isometric()
    # Show grid with axis labels
    plotter.show_grid(
        xtitle='X',
        ytitle='Y',
        ztitle='Z',
        grid='front'  # Display the grid in front of the scene
    )

    plotter.export_html('output_file.html')

