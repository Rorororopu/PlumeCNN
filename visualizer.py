'''
Using Matplotlib to visualize graphs, output them to PNG image.
'''

from pathlib import Path # Check if the path exist
import pandas as pd
import numpy as np
import typing

import matplotlib.pyplot as plt
import matplotlib.animation as animation

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