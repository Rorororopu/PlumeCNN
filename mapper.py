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
    1.1 Z-slice separation:  
        In our simulation, the z-axis grids are evenly spaced, and the data exported via VisIt follows this structure:

        x1, y1, z1\n x2, y2, z1\n ...\n xm, ym, z1\n
        x1, y1, z2\n x2, y2, z2\n ...\n xm, ym, z2\n ...\n
        x1, y1, zn\n x2, y2, zn\n ...\n xm, ym, zn\n
        xa, yn, z1\n xb, yb, z1\n ...\n xc, yc, z1\n
        xa, ya, z2\n xb, yb, z2\n ...\n xc, yc, z2\n ...\n
        xa, ya, zn\n xb, yb, zn\n ...\n xc, yc, zn\n
        (a series of same z coordinates will appear periodically)
        
        The following steps outline the process:
        - For each series of same z-coordinate(not including all data points of that coordinate), 
        export the corresponding 2D grid of data points into an okc file.

    1.2 Merge separated z slices
        Merge z-slices with same z-coordinates together.

    1.3 Interpolate completed z slices


2. Then, we will interpolate the data along y axis to get completed datas stored in in many y slices. 
    2.1 Y-slice separation 
        After doing z-slice interpolation, the data is arranged in csv file.
        Separated by empty rows, there are some data points grouped by same y coordinate, with slight deviations from the original value due to floating-point precision errors.

        The following steps outline the process:
        - for every files, separate it into multiple files based on the separation of empty rows.

    2.2 Merge separated y slices
        Merge siles with same y coordinates together.
    
    2.3 Interpolate completed y slices
        The results will be in final_dir = f"{base_dir}/{filename_without_ext}_result",
        Which contains the completely interpolated data, stored in many y slices.

There will also be a dictionary organizing these files, in format {<prefix>_0:Data(path_0), <prefix>_1:Data(path_1), ...}
The prefix is named by user.
'''
import csv
import numpy as np
import os
import pandas as pd
import scipy.interpolate
import shutil
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
    return interpolated_df


# Codes below are functions tools for interpolation of 3D data, because if we interpolate the 3D data directly, the Bletchley will be out of memory and will take forever to finish this task.
def okc_split(file: typing.IO, index: int, output_dir: str, z_index: int, row_count: int = 0, first_line: str = None) -> typing.Tuple[bool, str, int]:
    '''
    Splits the original 3D data file into separate files, each containing data for one z-coordinate slice (not unique, many files are having same z-coordinate).
    The function processes of one section with same z-coordinate per iteration, saves it in a temp file.
    
    Args:
        file: Opened file object of the original 3D data file.
        index: This is appended to the filename, indicating this is which iteration.
        output_dir: Path to the directory where the temporary files (z-slices) will be saved.
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
    output_filepath = os.path.join(output_dir, output_filename)

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


def okc_merge(dir_path: str, z_index: int):
    '''
    Merges files that contain the same z-coordinate in a specified column. 
    Each file is assumed to represent a partial z-slice of data, and multiple files may have the same z-coordinate. 
    This function consolidates these files into a single file per z-coordinate, appending the contents of files with the same z-coordinate into the first file and then deleting the others.

    Args:
        dir_path: Path to the directory containing the partial z-slice files.
        z_index: The index (0-based) of the column that contains the z-coordinate in each file.
    '''

    z_coord_file = os.path.join(dir_path, 'z_coord_map.txt')  # File to store z-coordinate mapping

    # Ensure z_coord_map.txt exists, or create it if it doesn't
    if not os.path.exists(z_coord_file):
        with open(z_coord_file, 'w'):
            pass  # Create an empty file

    # Step 1: First, scan all files and create the map
    files = os.listdir(dir_path)
    index = 0

    for filename in files:
        file_path = os.path.join(dir_path, filename)
        
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

        print(f"\rChecking the z coord of files... {index}/{len(files)-1}", end='', flush=True) # exclude z_coord_file
    
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


def mapper_z_slice(filepath: str, resolution: list, var_ranges: dict, use_comma: bool = False) -> pd.DataFrame:
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
    return interpolated_df


def csv_split(input_dir: str, output_dir: str):
    '''
    This function splits large CSV files in a directory into multiple smaller CSV files based on empty rows, 
    saving each segment as a separate file in the specified directory. Each section between 
    empty rows is written into a new file. Empty rows are not written to the output files.

    Args:
        input_dir: The path to the directory that has CSV files that need to be split.
        output_dir: The directory path where the new files will be saved. 
    '''

    # List all files in the input directory
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    number_of_file_processed = 0
    
    for file_name in files:
        file_path = os.path.join(input_dir, file_name)

        # Open the input CSV file
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            # Skip the first line (header)
            header = next(reader)
            part_index = 0
            part_lines = []

            for row in reader:
                # Check if the line is empty
                if any(row):         
                    part_lines.append(row)# Add the non-empty row to the current part
                else: # An empty row is detected (all elements are empty)
                    # If we have collected any data, write them to a new file
                    if part_lines:
                        # Create a new file in the output directory (no subdirectories)
                        output_file_name = f"{os.path.splitext(file_name)[0]}_{part_index}.csv"
                        output_file_path = os.path.join(output_dir, output_file_name)
                        
                        with open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
                            writer = csv.writer(outfile)
                            writer.writerow(header)  # Write the header
                            writer.writerows(part_lines)  # Write the collected data
                        
                        # Increment the file part index for next split
                        part_index += 1
                        part_lines = []  # Reset the collected data

            # Write the remaining rows if any exist after the last empty row
            if part_lines:
                output_file_name = f"{os.path.splitext(file_name)[0]}_{part_index}.csv"
                output_file_path = os.path.join(output_dir, output_file_name)
                
                with open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(header)
                    writer.writerows(part_lines)

        number_of_file_processed += 1
        print(f"\rSeparating z slices... {number_of_file_processed}/{len(files)}", end='', flush=True)

    print("\nDone!")


def csv_merge(dir_path: str):
    '''
    Merges CSV files that contain the same y-coordinate in a specified column.
    Each file is assumed to represent a partial y-slice of data, and multiple files may have the same y-coordinate.
    This function consolidates these files into a single file per y-coordinate, appending the contents of files with the same y-coordinate into the first file and then deleting the others.

    Args:
        dir_path: Path to the directory containing the partial y-slice files.
    '''
    y_coord_file = os.path.join(dir_path, 'y_coord_map.txt')  # File to store y-coordinate mapping

    # Ensure y_coord_map.txt exists, or create it if it doesn't
    if not os.path.exists(y_coord_file):
        with open(y_coord_file, 'w'):
            pass  # Create an empty file

    # Step 1: First, scan all CSV files and create the map
    files = os.listdir(dir_path)
    index = 0

    for filename in files:
        file_path = os.path.join(dir_path, filename)
        
        # Check if the path is indeed a file and not the y_coord_map.txt itself
        if os.path.isfile(file_path) and filename != 'y_coord_map.txt' and filename.endswith('.csv'):
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                headers = next(reader)  # Read the headers
                y_index = headers.index('y')  # Find the index of the 'y' column
                
                # Get the y-coordinate value from the first data row
                first_row = next(reader)
                y_coord = first_row[y_index]
                
                # Step 2: Update the y_coord_map.txt with the y_coord and filename
                # Read the y_coord_map.txt file to find if the y_coord already exists
                if os.path.getsize(y_coord_file) > 0:
                    with open(y_coord_file, 'r') as directory:
                        lines = directory.readlines()
                else:
                    lines = []

                found = False
                for i, line in enumerate(lines):
                    existing_y, file_list = line.strip().split(':', 1)
                    if existing_y == y_coord:
                        file_list = file_list.strip('[]')  # Clean the brackets
                        file_paths = [p.strip() for p in file_list.split(',')] if file_list else []
                        if file_path not in file_paths:
                            file_paths.append(file_path)  # Add the new file to the list
                        new_line = f"{y_coord}:[{', '.join(file_paths)}]\n"
                        lines[i] = new_line
                        found = True
                        break  # We found the y_coord, no need to continue

                if not found:
                    # If y_coord was not found, add a new entry for it
                    new_line = f"{y_coord}:[{file_path}]\n"
                    lines.append(new_line)
                    # Write the new line into the file

                # Now, write the updated lines back to y_coord_map.txt
                with open(y_coord_file, 'w') as directory:
                    directory.writelines(lines)
                    index += 1

        print(f"\rChecking the y coord of files... {index}/{len(files)-1}", end='', flush=True)  # exclude y_coord_file
    
    print("\nDone!")

    # Step 3: Now, merge files based on y-coordinates
    index = 0
    with open(y_coord_file, 'r') as directory:
        lines = directory.readlines()

    for line in lines:
        y_coord, file_list = line.strip().split(':', 1)
        file_list = file_list.strip('[]').split(', ')
        file_list = [f.strip() for f in file_list]
        
        # If there is more than one file with the same y coordinate, merge them
        if len(file_list) > 1:
            primary_file = file_list[0]
            with open(primary_file, 'r') as pri_file:
                reader = csv.reader(pri_file)
                headers = next(reader)  # Store headers for consistency

            for secondary_file in file_list[1:]:
                with open(secondary_file, 'r') as sec_file, open(primary_file, 'a', newline='') as pri_file:
                    reader = csv.reader(sec_file)
                    next(reader)  # Skip the header row in the secondary file
                    writer = csv.writer(pri_file)
                    for row in reader:
                        writer.writerow(row)

                # Delete the secondary file after merging
                os.remove(secondary_file)
        index += 1
        print(f"\rMerging files... {index}/{len(lines)}", end='', flush=True)

    os.remove(y_coord_file)
    print("\nDone!")


def mapper_y_slice(filepath: str, resolution: list, var_ranges: dict) -> pd.DataFrame:
    '''
    Interpolates the y-slice data from a CSV file into a pandas DataFrame, using regular grid spacing.

    Args:
        filepath: The path to the CSV file containing the y-slice data.
        resolution: A list specifying the grid resolution as [x_res, z_res].
        var_ranges: A dictionary containing the column titles and their ranges, used to identify which columns represent the coordinates.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the interpolated data for the z-slice.
    '''
    
    # Prepare mesh grid based on x and z ranges in var_ranges and the specified resolution
    grid_1, grid_2 = (np.linspace(var_ranges[var][0], var_ranges[var][1], res) for var, res in zip(['x', 'z'], resolution))
    mesh_grid_1, mesh_grid_2 = np.meshgrid(grid_1, grid_2)
    coordinates = np.stack((mesh_grid_1.ravel(), mesh_grid_2.ravel()), axis=-1)

    # Define columns to read, excluding any unwanted columns (e.g., 'time_derivative/conn_based/mesh_time')
    columns_to_read = [col for col in var_ranges.keys() if col not in ['time_derivative/conn_based/mesh_time']]
    indices = [i for i, var in enumerate(var_ranges.keys()) if var in columns_to_read]
    
    # Read the CSV file
    batch = pd.read_csv(filepath, usecols=indices, engine='c', header=0)

    # Replace NaN values with 0 for safety in processing
    batch.fillna(0, inplace=True)

    # Get points (x, z coordinates) for interpolation
    points = batch[['x', 'z']].values
    
    # Initialize an empty array to store the interpolated values
    interpolated_batch = np.full((coordinates.shape[0], len(columns_to_read)), np.nan)

    # Interpolate each column (variable) in the batch based on the provided x, y coordinates
    for i, col in enumerate(columns_to_read):
        values = batch[col].values
        interpolator = scipy.interpolate.LinearNDInterpolator(points, values)
        grid_data = interpolator(coordinates)
        grid_data[np.isnan(grid_data)] = np.nan  # Fill NaNs explicitly if there are out-of-bound grid points
        interpolated_batch[:, i] = grid_data

    # Convert the interpolated results into a DataFrame
    interpolated_df = pd.DataFrame(interpolated_batch, columns=columns_to_read)
    interpolated_df = interpolated_df.reset_index(drop=True)

    return interpolated_df


def mapper_3D(filepath: str, resolution: list, use_comma: bool = False) -> str:
    '''
    This function processes a large 3D data file by splitting it into z slices, interpolating each slice, and saving the results.
    The original slices are saved as .tmp files, and the interpolated data is saved as .csv files.

    several tempory directories with following paths will be created, so don't create directory at their path:
    f"{base_dir}/{filename_without_ext}_z_tmp"
    f"{base_dir}/{filename_without_ext}_y_tmp"
    f"{base_dir}/{filename_without_ext}_z_csv"

    The completed interpolated data will be in f"{base_dir}/{filename_without_ext}_result", store in many y slices

    Args:
        filepath: Path to the original file.
        resolution: Resolution of the interpolation in the form [x_res, y_res, z_res].
        use_comma: Whether the original file uses commas as delimiters. Defaults to False.

    Returns:
        f"{base_dir}/{filename_without_ext}_result" (The path to store the result).
    '''
    # Get the base directory and filename (without the .okc extension)
    base_dir = "/".join(filepath.split('/')[:-1])
    filename_without_ext = filepath.split('/')[-1].replace('.okc', '')

    # Create directories for temporary files and interpolated CSVs
    z_tmp_dir = f"{base_dir}/{filename_without_ext}_z_tmp"
    y_tmp_dir = f"{base_dir}/{filename_without_ext}_y_tmp"
    z_csv_dir = f"{base_dir}/{filename_without_ext}_z_csv"
    final_dir = f"{base_dir}/{filename_without_ext}_result"
    os.makedirs(z_tmp_dir, exist_ok=True)
    os.makedirs(z_csv_dir, exist_ok=True)
    os.makedirs(y_tmp_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

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
            end_of_file, first_line, row_count = okc_split(file, index, z_tmp_dir, z_index, row_count, first_line)
            print(f"\rsplitting file to partial z slices... {row_count}/{grid_num}  ", end='', flush=True)
            if end_of_file:  # Ensure we break if we reached the end
                break
            index += 1 # Update start_line and index for the next slice
        print("\nDone!")

    # Merge files with same z coordinates
    okc_merge(z_tmp_dir, z_index)

    z_slices = os.listdir(z_tmp_dir)
    index = 0
    for slice_file in z_slices:
        df = mapper_z_slice(f"{z_tmp_dir}/{slice_file}", [resolution[0], resolution[1]], var_ranges, use_comma)
        index += 1
        df.to_csv(f"{z_csv_dir}/{filename_without_ext}_{index}.csv", index=False)
        print(f"\rInterpolating z slices... {index}/{len(z_slices)}", end='', flush=True)
    print("\nDone!")

    # Split the original CSVs to many little files, each file containing same y coordinates, although different file may have same coordinates
    csv_split(z_csv_dir, y_tmp_dir)

    # Merge files with same y coordinates
    csv_merge(y_tmp_dir)

    y_slices = os.listdir(y_tmp_dir)
    index = 0
    for slice_file in y_slices:
        df = mapper_y_slice(f"{y_tmp_dir}/{slice_file}", [resolution[0], resolution[2]], var_ranges)
        index += 1
        df.to_csv(f"{final_dir}/{filename_without_ext}_{index}.csv", index=False)
        print(f"\rInterpolating y slices... {index}/{len(y_slices)}", end='', flush=True)
    print("\nDone!")

    shutil.rmtree(z_tmp_dir)
    shutil.rmtree(y_tmp_dir)
    shutil.rmtree(z_csv_dir)

    return final_dir


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


