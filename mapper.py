'''
Map the original data file to a evenly spaced grid with user-specified resolution.

The data itself will be stored to a Pandas dataframe (in simple words, a table), with the following titles:
    x, y, z, temperature, velocity_magnitude, z-velocity
Time column is deleted. If the data is sliced, its corresponding x, y, or z coordinate will not be included.
Points out of the original data range will be converted to NaN.

There is more steps to do interpolation for 3D data, because firectly interpolating large 3D datasets can result in excessive memory usage 
and may let the job takes forever to finish. 
To mitigate this, we propose a more efficient step-by-step I/O approach to handle the data interpolation while conserving memory.


1. Instead of interpolating the entire 3D dataset at once, we will first perform interpolation slice by slice along the z-axis.
1.1 Z-Slice Separation:  
   In our simulation, the z-axis grids are evenly spaced, and the data exported via VisIt follows this structure:

   x1, y1, z1\n x2, y2, z1\n ...\n xm, ym, z1\n
   x1, y1, z2\n x2, y2, z2\n ...\n xm, ym, z2\n ...\n
   x1, y1, zn\n x2, y2, zn\n ...\n xm, ym, zn\n
   xa, yn, z1\n xb, yb, z1\n ...\n xc, yc, z1\n
   xa, ya, z2\n xb, yb, z2\n ...\n xc, yc, z2\n ...\n
   xa, ya, zn\n xb, yb, zn\n ...\n xc, yc, zn\n
   (a series of same z coordinates will appear periodically)
   
   The following steps outline the process:

   - For each series of same z-coordinate(not including all data points of that coordinate), export the corresponding 2D grid of data points into an okc file.

1.2 Merging separated z-slices
    Merge z-slices with same z-coordinates together.

There will also be a dictionary organizing these files, in format {<prefix>_0:Data(path_0), <prefix>_1:Data(path_1), ...}
The prefix is named by user.
'''

import numpy as np
import os
import pandas as pd
import scipy.interpolate
import typing
import preliminary_processing


def get_time(var_ranges: dict) -> float:
    '''
    Extracts the time value from the var_ranges dictionary at the key.
    The min and max values should be the same.
    Args:
    var_ranges: dictionary in format {varname:[min, max], ...}.
    Returns:
    time: The time value extracted from the dictionary.
    If the key 'time_derivative/conn_based/mesh_time' doesn't exist, return None.
    '''
    key = 'time_derivative/conn_based/mesh_time'
    if key not in var_ranges:
        return None
    
    time_range = var_ranges[key]
    
    if time_range[0] != time_range[1]:
        print("\033[91mError: The min and max values for time in one file are not the same.\033[0m")
        return None
    
    time = time_range[0]
    print("Obtain the time for this data.")
    return time


def get_resolution_2D(datas_object: preliminary_processing.Datas) -> list:
    '''
    Args:
        a datas_object. Only slicing and grid_num attribute is needed.
    
    Returns:
        resolution: a list in format [res1, res2], in accordance to vars_plotted
    '''
    while True:
        resolution_list = input(f"\nWhat is your desired resolution for your graph?\n"
                   "Please input two numbers in the format 'num1,num2' or 'num1 num2' to corresponding variables.\n"
                   f"Note that there are {datas_object.grid_num} grid points in your simulation, "
                   "and they are sliced.\n"
                   "So don't set a resolution too high or too low compared to your original one.\n"
                   "If it's too low, the information in your data will be lost.\n"
                   "If it's too high, the program can't correctly calculate derivatives.\n"
                   f"Also, the aspect ratio(diameter/depth) of your simulation is {datas_object.aspect_ratio}.\n"
                   "Choose a resolution whose aspect ratio is close to your original one.\n")
        resolution_list = resolution_list.replace(',', ' ').split()  # Replace commas with spaces and split the input into a list
        if len(resolution_list) != 2:
            print("\033[91mInvalid input. Please input 2 numbers!\033[0m")
            continue # Go to the next loop
        try:
            res1 = int(resolution_list[0])
            res2 = int(resolution_list[1])
            return [res1, res2]
        except ValueError:
            print("\033[91mInvalid input. Please input 2 numbers!\033[0m")
    

def get_resolution_3D(datas_object: preliminary_processing.Datas) -> list:
    '''
    It could automatically calculate the resolution.

    Args:
        a datas_object. Only grid_num attribute is needed.
    
    Returns:
        resolution: a list in format [resol1, resol2, resol3], in accordance to vars_plotted
    '''
    while True:
        resolution_list = input(f"\nWhat is your desired resolution for your 'x', 'y' and 'z' graph?\n"
                   "Please input two numbers in the format 'num1,num2,num3' or 'num1 num2 num3' to corresponding variables.\n"
                   f"Note that there are {datas_object.grid_num} grid points in your simulation, "
                   "So don't set a resolution too high or too low compared to your original one.\n"
                   "Choose a resolution similar to your original one.\n"
                   "If it's too low, the information in your data will be lost.\n"
                   "If it's too high, the program can't correctly calculate derivatives.\n"
                   f"Also, the aspect ratio(diameter/depth) of your simulation is {datas_object.aspect_ratio}.\n"
                   "Choose a resolution whose aspect ratio is close to your original one.\n"
                   "If you want to let this program to automatically calculate resolution,\n"
                   "Please replace that value to '-'(hyphen), let the program determine what is its exact value.\n"
                   "For instance, your aspect ratio is 0.1, and you want your z-axis have 1000 grid points,\n"
                   "you could input '-,-,1000', and the program will know that you actually want the resolution be '100,100,1000'.\n"
                   "Note that resolution for x abd y SHOULD NORMALLY BE EQUAL!")
        resolution_list = resolution_list.replace(',', ' ').split()  # Replace commas with spaces and split the input into a list
        if len(resolution_list) != 3:
            print("\033[91mInvalid input. Please input at least 1 number!\033[0m")
            continue # Go to the next loop
        try:
            res1 = int(resolution_list[0]) if resolution_list[0] not in ['-', '_', '*', '#', '.', '..', '...'] else None
            res2 = int(resolution_list[1]) if resolution_list[1] not in ['-', '_', '*', '#', '.', '..', '...'] else None
            res3 = int(resolution_list[2]) if resolution_list[2] not in ['-', '_', '*', '#', '.', '..', '...'] else None
            if (res1 is not None) and (res2 is not None) and (res3 is not None): # No need to calculate
                if res1 != res2:
                    print("\033[91mInvalid input. Resolution for x and y should be equal.\033[0m")
                    continue
                return [res1, res2, res3]
            elif (res3 is not None) and (res1 is None) and (res2 is None): # input is' -, -, res3'
                res1 = int(res3 * datas_object.aspect_ratio)
                res2 = res1
                return [res1, res2, res3]
            elif (res3 is None) and (res1 is not None) and (res2 is not None):  # input is 'res1, res2, -'
                if res1 != res2:
                    print("\033[91mInvalid input. Resolution for x and y should be equal.\033[0m")
                    continue
                res3 = int(res1 / datas_object.aspect_ratio)
                return [res1, res2, res3]
            elif ((res1 is not None) and (res2 is None) and (res3 is None)) or ((res2 is not None) and (res1 is None) and (res3 is None)): # input is 'res1, -, -' or '-, res2, -'
                res1 = res2 if res1 is None else res1
                res2 = res1 if res2 is None else res2
                res3 = int(res1 / datas_object.aspect_ratio)
                return [res1, res2, res3]
            elif (res1 is None) and (res2 is None) and (res3 is None): # Inpiut is '-,-,-'
                print("\033[91mInvalid input. Please input at least 1 number!\033[0m")
            else: # Input is 'res1, -, res3' or '-, res2, res3'
                res1 = res2 if res1 is None else res1
                res2 = res1 if res2 is None else res2
                return [res1, res2, res3]
        except ValueError:
            print("\033[91mInvalid input. Please input at least 1 number!\033[0m")
            continue


def get_resolution(datas_object: preliminary_processing.Datas) -> list:
    '''
    Args:
        a datas_object. Only slicing and grid_num attribute is needed.
    
    Returns:
        resolution: a list in format [res1, res2], or [res1, res2, res3] in accordance to vars_plotted.

    Error:
        if datas_object.slicing doesn't exist, prompt the user and kill the program.
    '''

    if not datas_object.slicing:
        resolution = get_resolution_3D(datas_object)
    else:
        resolution = get_resolution_2D(datas_object)

    return resolution


def mapper_2D(filepath: str, resolution: list, use_comma: bool = False) -> pd.DataFrame:
    '''
    Process a sliced data file by interpolating the variables onto a regularly spaced grid defined by the resolution.
    Time and z coordinate columns will be dropped, because it is sliced and VisIt will always make z column be 0. 
    NaN values will be converted to 0. 
    Points out of the original data range will be converted to NaN.

    (We have tried to store the data into a tensor, but it proves to not be the best choice, because it is really hard to drop points.)

    Args:
        filepath: Path of that data.
        resolution: Resolution in format [res1, res2].
        use_comma: Whether change the delimiter to comma rather than whitespace. False in default.
    
    Returns: 
        interpolated_df: the Pandas dataframe of that data.
    '''
    # From slicing, know what vars to be plotted
    var_ranges, _ = preliminary_processing.get_info(filepath)
    
    # Prepare mesh grid
    grid_1, grid_2 = (np.linspace(var_ranges[var][0], var_ranges[var][1], res) for var, res in zip(['x', 'y'], resolution)) # Generate 2 arrays of coords
    mesh_grid_1, mesh_grid_2 = np.meshgrid(grid_1, grid_2) # Repeat elements in grid_1 and grid_2
    coordinates = np.stack((mesh_grid_1.ravel(), mesh_grid_2.ravel()), axis=-1) # Flatten mesh_grid_1 and 2 and stack them.

    # Define columns to read
    columns_to_read = [col for col in var_ranges.keys() if col not in ['time_derivative/conn_based/mesh_time', 'z']] # Excluding time, sliced coordinate
    indices = [i for i, var in enumerate(var_ranges.keys()) if var in columns_to_read]
    interpolated_results = []
    
    # I planned to make the data process be in batches, but it shows that then it can't deal with the boundry of batches,
    # making the data blurred and behaving strangely. So I deleted the batch part. Still, I kept the name `batch`.
    if not use_comma:
        batch = pd.read_csv(filepath, skiprows=1+2*len(var_ranges), sep=r'\s+', header=None, names=columns_to_read, usecols=indices, engine='c')     
    else:
        batch = pd.read_csv(filepath, skiprows=1+2*len(var_ranges), header=None, names=columns_to_read, usecols=indices, engine='c')     
    '''
    Skip header lines.
    Delimiters are whitespace(default setting of my code) or comma(default setting of pandas).
    No header present in the CSV.
    Provide column names for the data.
    Only read data in indicated indices.
    Use c engine to increase the speed.
    '''
    batch.fillna(0, inplace=True) # Replace NaNs with zero
    points = batch[['x','y']].values
    interpolated_batch = np.full((coordinates.shape[0], len(columns_to_read)), np.nan) # Create an empty array to store interpolated values.
    
    for i, col in enumerate(columns_to_read):
        values = batch[col].values
        # For grid points out of the range, add NaN as their value (For instance, when slicing direction is z.)
        # Filling value with NaNs may cause many problems to fix later, 
        # but it will have bugs, if you fill in other values, for reason unknown.
        interpolator = scipy.interpolate.LinearNDInterpolator(points, values)
        grid_data = interpolator(coordinates)
        grid_data[np.isnan(grid_data)] = np.nan
        interpolated_batch[:, i] = grid_data
    interpolated_results.append(interpolated_batch)

    # Concatenate all interpolated results and convert them into a DataFrame
    final_data = np.concatenate(interpolated_results, axis=0)
    interpolated_df = pd.DataFrame(final_data, columns=columns_to_read)
    interpolated_df = interpolated_df.reset_index(drop=True)
    
    # Return the final interpolated DataFrame
    print("Finished mapping this data to your specified resolution.")
    return interpolated_df


# Codes below are functions tools for interpolation of 3D data, because if we interpolate the 3D data directly, the Bletchley will be out of memory and will take forever to finish this task.
def okc_split(file: typing.IO, index: int, dir_path: str, z_index: int, row_count: int = 0, first_line: str = None) -> typing.Tuple[bool, str, int]:
    '''
    Splits the original 3D data file into separate files, each containing data for one z-coordinate slice (not unique, many files are having same z-coordinate).
    The function processes of one section with same z-coordinate per iteration, saves it in a temp file.
    
    Args:
        file: Opened file object of the original 3D data file.
        index: This is appended to the filename, indicating this is which iteration.
        dir_path: Path to the directory where the temporary files (z-slices) will be saved.
        z_index: The index of the z-coordinate column, e.g. if z_index = 2, the z-coordinate is at the 3rd column.
        row_count: number of rows that have been processed
        first_line: If provided, use this line as the first line for this iteration. Without this parameter, the line to cause the break will not be recorded in any file, that's not what we want.
    
    Returns:
        bool: True if the end of the file is reached, otherwise False.
        str: The line that caused the break, so it can be passed to the next iteration.
        int: number of rows that have been processed
    '''
    # Prepare the output file path
    base_filename = os.path.basename(file.name).rsplit('.', 1)[0]
    output_filename = f"{base_filename}_{str(index)}.tmp"
    output_filepath = os.path.join(dir_path, output_filename)

    # Create a new file in this iteration
    with open(output_filepath, 'w') as outfile:
        if first_line is None:
            first_line = file.readline().strip()  # Read the first line if not provided

        if not first_line:  # If the file is empty or we reached EOF
            return True, None, row_count

        outfile.write(first_line + '\n')  # Write the first line to the new file
        row_count += 1
        z_value = first_line.split()[z_index]  # Get the z-coordinate from the first line

        # Process subsequent lines
        for line in file:
            line = line.strip()
            current_z_value = line.split()[z_index]

            if current_z_value == z_value:
                outfile.write(line + '\n')
                row_count += 1
            else:
                return False, line, row_count  # Return the current line as it caused the break

    # If we processed all lines with the same z-coordinate, return success
    return True, None, row_count


def okc_merge(folder_path: str, z_index: int):
    '''
    Merges files that contain the same z-coordinate in a specified column. 
    Each file is assumed to represent a partial z-slice of data, and multiple files may have the same z-coordinate. 
    This function consolidates these files into a single file per z-coordinate, appending the contents of files with the same z-coordinate into the first file and then deleting the others.

    Args:
        folder_path: Path to the directory containing the partial z-slice files.
        z_index: The index (0-based) of the column that contains the z-coordinate in each file.
    '''

    z_coord_file = os.path.join(folder_path, 'z_coord_map.txt')  # File to store z-coordinate mapping

    # Ensure z_coord_map.txt exists, or create it if it doesn't
    if not os.path.exists(z_coord_file):
        with open(z_coord_file, 'w'):
            pass  # Create an empty file

    # Step 1: First, scan all files and create the map
    files = os.listdir(folder_path)
    index = 0

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        
        # Check if the path is indeed a file and not the z_coord_map.txt itself
        if os.path.isfile(file_path) and filename != 'z_coord_map.txt':
            with open(file_path, 'r') as file:
                # Read the first line and extract the z coordinate at the given index
                z_coord = file.readline().strip().split()[z_index]
                
                # Step 2: Update the z_coord_map.txt with the z_coord and filename

                # Read the z_coord_map.txt file to find if the z_coord already exists
                if os.path.getsize(z_coord_file) > 0:
                    with open(z_coord_file, 'r') as directory:
                        lines = directory.readlines()
                else:
                    lines = []

                found = False
                for i, line in enumerate(lines):
                    existing_z, file_list = line.strip().split(':', 1)
                    if existing_z == z_coord:
                        file_list = file_list.strip('[]')  # Clean the brackets
                        file_paths = [p.strip() for p in file_list.split(',')] if file_list else []
                        if file_path not in file_paths:
                            file_paths.append(file_path)  # Add the new file to the list
                        new_line = f"{z_coord}:[{', '.join(file_paths)}]\n"
                        lines[i] = new_line
                        found = True
                        # REPLACE that old line with new line. Don't create a dictionary to save memory.
                        break  # We found the z_coord, no need to continue

                if not found:
                    # If z_coord was not found, add a new entry for it
                    new_line = f"{z_coord}:[{file_path}]\n"
                    lines.append(new_line)
                    # Write the new line into the file

                # Now, write the updated lines back to z_coord_map.txt
                with open(z_coord_file, 'w') as directory:
                    directory.writelines(lines)
                    index += 1

        print(f"\rChecking the z coordinate of splitted files... {index}/{len(files)-1}", end='', flush=True) # exclude z_coord_file
    
    print("\nDone!")
    
    # Step 3: Now, merge files based on z-coordinates
    index = 0
    with open(z_coord_file, 'r') as directory:
        lines = directory.readlines()

    for line in lines:
        z_coord, file_list = line.strip().split(':', 1)
        file_list = file_list.strip('[]').split(', ')
        file_list = [f.strip() for f in file_list]
        
        # If there is more than one file with the same z coordinate, merge them
        if len(file_list) > 1:
            primary_file = file_list[0]
            for secondary_file in file_list[1:]:
                with open(secondary_file, 'r') as sec_file, open(primary_file, 'a') as pri_file:
                    for line in sec_file:
                        pri_file.write(line)

                # Delete the secondary file after merging
                os.remove(secondary_file)
        index += 1
        print(f"\rMerging files... {index}/{len(lines)}", end='', flush=True)

    os.remove(z_coord_file)
    print("\nDone!")


def mapper_z_slice(filepath:str, resolution:list, var_ranges: dict, use_comma:bool = False) -> pd.DataFrame:
    '''
        Interpolates the z-slice data (output from okc_splitter()) into a pandas DataFrame, using regular grid spacing.

        Args:
            filepath: The path to the file containing the z-slice data.
            resolution: A list specifying the grid resolution as [x_res, y_res].
            use_comma: A boolean indicating whether the original file uses commas as delimiters. Defaults to False.
            var_ranges: A dictionary containing the column titles and their ranges, used to identify which columns represent the coordinates.
                Because the program only care the range of x and y coordinates, although ranges of other variables are different from the 3D data, it doesn't matter.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the interpolated data for the z-slice.
    '''
    
    # Prepare mesh grid
    grid_1, grid_2 = (np.linspace(var_ranges[var][0], var_ranges[var][1], res) for var, res in zip(['x', 'y'], resolution)) # Generate 2 arrays of coords
    mesh_grid_1, mesh_grid_2 = np.meshgrid(grid_1, grid_2) # Repeat elements in grid_1 and grid_2
    coordinates = np.stack((mesh_grid_1.ravel(), mesh_grid_2.ravel()), axis=-1) # Flatten mesh_grid_1 and 2 and stack them.

    # Define columns to read
    columns_to_read = [col for col in var_ranges.keys() if col not in ['time_derivative/conn_based/mesh_time']] # Excluding time
    indices = [i for i, var in enumerate(var_ranges.keys()) if var in columns_to_read]
    interpolated_results = []
    
    # I planned to make the data process be in batches, but it shows that then it can't deal with the boundry of batches,
    # making the data blurred and behaving strangely. So I deleted the batch part. Still, I kept the name `batch`.
    if not use_comma:
        batch = pd.read_csv(filepath, sep=r'\s+', header=None, names=columns_to_read, usecols=indices, engine='c')     
    else:
        batch = pd.read_csv(filepath, header=None, names=columns_to_read, usecols=indices, engine='c')     
    batch.fillna(0, inplace=True) # Replace NaNs with zero
    points = batch[['x','y']].values
    interpolated_batch = np.full((coordinates.shape[0], len(columns_to_read)), np.nan) # Create an empty array to store interpolated values.

    for i, col in enumerate(columns_to_read):
        values = batch[col].values
        # For grid points out of the range, add NaN as their value (For instance, when slicing direction is z.)
        # Filling value with NaNs may cause many problems to fix later, 
        # but it will have bugs, if you fill in other values, for reason unknown.
        interpolator = scipy.interpolate.LinearNDInterpolator(points, values)
        grid_data = interpolator(coordinates)
        grid_data[np.isnan(grid_data)] = np.nan
        interpolated_batch[:, i] = grid_data
    interpolated_results.append(interpolated_batch)

    # Concatenate all interpolated results and convert them into a DataFrame
    final_data = np.concatenate(interpolated_results, axis=0)
    interpolated_df = pd.DataFrame(final_data, columns=columns_to_read)
    interpolated_df = interpolated_df.reset_index(drop=True)
    
    # Return the final interpolated DataFrame
    print("Finished mapping this z slice to the specified resolution.")
    return interpolated_df


def mapper_3D(filepath: str, resolution: list, use_comma: bool = False):
    '''
    This function processes a large 3D data file by splitting it into z slices, interpolating each slice, and saving the results.
    The original slices are saved as .tmp files, and the interpolated data is saved as .csv files.

    Args:
        filepath: Path to the original file.
        resolution: Resolution of the interpolation in the form [x_res, y_res, z_res].
        use_comma: Whether the original file uses commas as delimiters. Defaults to False.

    Returns:
        None
    '''
    # Get the base directory and filename (without the .okc extension)
    base_dir = "/".join(filepath.split('/')[:-1])
    filename_without_ext = filepath.split('/')[-1].replace('.okc', '')

    # Create directories for temporary files and interpolated CSVs
    tmp_dir = f"{base_dir}/{filename_without_ext}_tmp"
    '''csv_dir = f"{base_dir}/{filename_without_ext}_tmp_csv" '''
    os.makedirs(tmp_dir, exist_ok=True)
    ''' os.makedirs(csv_dir, exist_ok=True)'''

    # Get the variable ranges from the file, find which column represents z coordinate
    var_ranges, grid_num = preliminary_processing.get_info(filepath)
    z_index = list(var_ranges.keys()).index('z')

    # Split the original file to many little files, each file containing same z coordinates, although different file may have same coordinates
    with open(filepath, 'r') as file:
        start_line = 1 + 2 * len(var_ranges)  # Start after the metadata and ranges
        index = 0
        row_count = 0
        first_line = None # "lookahead" approach
        for _ in range(start_line):
            next(file)
        while True:
            # Split one z slice into one temp file
            end_of_file, first_line, row_count = okc_split(file, index, tmp_dir, z_index, row_count, first_line)
            print(f"\rsplitting the original file to z slices... {row_count}/{grid_num}", end='', flush=True)
            if end_of_file:  # Ensure we break if we reached the end
                break
            index += 1 # Update start_line and index for the next slice
        print("\nDone!")

    # Merge files with same z coordinates
    okc_merge(tmp_dir, z_index)


def mapper(datas_object:preliminary_processing.Datas, filepath:str, resolution:list) -> pd.DataFrame:
    '''
    Process an data file by interpolating the variables onto a regularly spaced grid defined by the resolution.
    Time column and if the data is sliced, z values (it will always be 0 if it is sliced) will be dropped. NaN values will be converted to 0. 
    Points out of the original data range will be converted to NaN.

    Args:
        datas_object: just to jnow the slicing.
        filepath: Path of that data.
        resolution: Resolution in format [res1, res2] ot [x_res, y_res, z_res].
    
    Returns: 
        interpolated_df: the interpolated Pandas dataframe.
    '''
    if not datas_object.slicing:
        interpolated_df = mapper_3D(filepath, resolution)
    else:
        interpolated_df = mapper_2D(filepath, resolution)
    return interpolated_df


