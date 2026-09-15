import sys

class LearnSmartAIException(Exception):

    """
    Custom Exception class for this project
    """

    def __init__(                ### constructor of custom exception class
            self,
            error_message:str,   ### the actual error message
            error_detail:None    ### optional info about the error
    ):

        super().__init__(error_message)    ### calls the constructor of the parent exception class

        self.error_message = error_message  ### stores the error message as an attribute 

        if error_detail is not None:        ### checks whether additional info about the error/traceback info was provided
            _,_, traceback = error_detail.exc_info()   ### error detail contains = error_type, error_value, traceback

            if traceback is not None:           ### checks whether valid traceback object exists
                self.file_name = traceback.tb_frame.f_code.co_filename     ### gets the file name where the error ocurred
                self.line_number = traceback.tb_lineno                     ### gets the line number of the error ocurred

            else:
                self.file_name = __file__   ### __file__ contains the file name of the python file which the error was ocurred
                self.line_number = 0        ### uses 0 because the actual error line is not available

        else:
            ### identify the caller that raised this custom 
            ### this executes when no error detail was provided

            frame = sys._getframe(1)                   ### gets the execution frame of the caller

            self.file_name =frame.f_code.co_filename   ### gets the file name associated with the caller
            self.line_number = frame.f_lineno          ### gets the line number 

    def __str__(self) -> str:

        """
        defines how our error message should be displayed
        """
        return(
            f"Error Ocurred in file "
            f"[{self.file_name}] "
            f"at line [{self.line_number}]: "
            f"{self.error_message}"

        )