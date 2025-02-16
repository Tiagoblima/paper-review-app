import os
import pandas as pd
from openpyxl.utils.exceptions import InvalidFileException

def save_df_to_excel(dataframe, sheet_name, save_path):
    """
    Save a pandas DataFrame to an Excel file, either creating a new file or updating an existing sheet.
    
    Args:
        dataframe (pd.DataFrame): The DataFrame to save
        sheet_name (str): Name of the sheet to save/update
        save_path (str): Full path to the Excel file
        
    Raises:
        ValueError: If there are issues reading/writing the Excel file
    """
   
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")

    try:
        if os.path.exists(save_path):
            try:
                # Try to read existing sheet and append new data
                existing_df = pd.read_excel(save_path, sheet_name=sheet_name)
                dataframe = pd.concat([existing_df, dataframe], ignore_index=True)
            except ValueError:
                # Sheet doesn't exist - will create new one
                pass
            except InvalidFileException:
                raise ValueError(f"File {save_path} exists but is not a valid Excel file")
            
            # Update the workbook with new data
            with pd.ExcelWriter(save_path, engine='openpyxl', mode='a', 
                              if_sheet_exists='replace') as writer:
                dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            # Create new workbook
            with pd.ExcelWriter(save_path, engine='openpyxl', mode='w') as writer:
                dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
                
    except PermissionError:
        raise ValueError(f"Unable to write to {save_path}. File may be open or protected.")
    except Exception as e:
        raise ValueError(f"Error saving DataFrame to Excel: {str(e)}")
