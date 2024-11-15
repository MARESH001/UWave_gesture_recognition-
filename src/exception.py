import sys

def error_message_detail(error, error_detail: sys):
    """Generate a detailed error message for debugging purposes."""
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = (
        f"Error occurred in the Python script: [{file_name}] "
        f"on line number [{line_number}]. "
        f"Error message: [{str(error)}]"
    )
    return error_message

class CustomException(Exception):
    """Custom exception class for more detailed error reporting."""
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message

class DataIngestionError(CustomException):
    """Exception raised for errors during data ingestion."""
    def __init__(self, error_message="An error occurred during data ingestion", error_detail: sys = None):
        if error_detail:
            super().__init__(error_message, error_detail)
        else:
            super().__init__(error_message)

class DataTransformationError(CustomException):
    """Exception raised for errors during data transformation."""
    def __init__(self, error_message="An error occurred during data transformation", error_detail: sys = None):
        if error_detail:
            super().__init__(error_message, error_detail)
        else:
            super().__init__(error_message)

class ModelTrainingError(CustomException):
    """Exception raised for errors during model training."""
    def __init__(self, error_message="An error occurred during model training", error_detail: sys = None):
        if error_detail:
            super().__init__(error_message, error_detail)
        else:
            super().__init__(error_message)

class ModelEvaluationError(CustomException):
    """Exception raised for errors during model evaluation."""
    def __init__(self, error_message="An error occurred during model evaluation", error_detail: sys = None):
        if error_detail:
            super().__init__(error_message, error_detail)
        else:
            super().__init__(error_message)


