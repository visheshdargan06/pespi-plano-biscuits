import os


def get_project_path(project_name = "pepsi-plano-biscuits"):
    """
    Returns the absolute path of the current project
    
    Parameters:
    project_name (str): Name of the current project
    
    Returns:
    str: Absolute path of the current project
    """
    
    #return '/media/premium/common-biscuit/main/planogram_biscuit'
    return '/tf/pepsi-plano-biscuits/'
    #return os.path.join(os.getcwd().split('/')[0], os.getcwd().split('/')[1])
