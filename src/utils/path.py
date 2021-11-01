import os


def get_project_path(project_name = "planogram"):
    """
    Returns the absolute path of the current project
    
    Parameters:
    project_name (str): Name of the current project
    
    Returns:
    str: Absolute path of the current project
    """
    
    return '/media/premium/common-biscuit/main/planogram_biscuit'
    #return os.path.join(os.getcwd().split(project_name)[0])
