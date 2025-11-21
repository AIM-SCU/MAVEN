# # utils/logger.py
# import os
# from datetime import datetime
# from rich.console import Console

# # Global console instance (for colored terminal output)
# console = Console()
# _current_prompt_id = None
# _current_output_folder = None

# def set_current_prompt(prompt_id: str, output_folder: str):
#     global _current_prompt_id, _current_output_folder
#     _current_prompt_id = prompt_id
#     _current_output_folder = output_folder
#     os.makedirs(_current_output_folder, exist_ok=True)

# def log(msg: str, level: str = "info"):
#     timestamp = datetime.now().strftime("[%H:%M:%S]")
#     full_msg = f"{timestamp} {msg}"

#     # Print to terminal
#     if level == "info":
#         console.print(full_msg)
#     elif level == "warn":
#         console.print(f"[yellow]{full_msg}[/yellow]")
#     elif level == "error":
#         console.print(f"[red]{full_msg}[/red]")
#     else:
#         console.print(full_msg)

#     # Write to log file if set
#     if _current_prompt_id and _current_output_folder:
#         log_file = os.path.join(_current_output_folder, f"{_current_prompt_id}.log")
#         with open(log_file, "a", encoding="utf-8") as f:
#             f.write(full_msg + "\n")







#########
# # utils/logger.py
# import os
# import builtins
# from datetime import datetime
# from rich.console import Console

# console = Console()
# _current_prompt_id = None
# _current_output_folder = None
# _original_print = builtins.print  # Keep original print function

# def set_current_prompt(prompt_id: str, output_folder: str):
#     global _current_prompt_id, _current_output_folder
#     _current_prompt_id = prompt_id
#     _current_output_folder = output_folder
#     os.makedirs(_current_output_folder, exist_ok=True)

#     # Override built-in print only once per prompt
#     builtins.print = global_print_redirect

# def global_print_redirect(*args, sep=' ', end='\n', file=None, flush=False):
#     # Convert all args to string
#     msg = sep.join(str(arg) for arg in args) + end

#     # Get timestamp
#     timestamp = datetime.now().strftime("[%H:%M:%S]")
#     full_msg = f"{timestamp} {msg}".rstrip()

#     # Write to terminal using rich console
#     console.print(full_msg)

#     # Write to log file if prompt is set
#     if _current_prompt_id and _current_output_folder:
#         log_file = os.path.join(_current_output_folder, f"{_current_prompt_id}.log")
#         with open(log_file, "a", encoding="utf-8") as f:
#             f.write(full_msg + "\n")

# def log(msg: str, level: str = "info"):
#     timestamp = datetime.now().strftime("[%H:%M:%S]")
#     full_msg = f"{timestamp} {msg}"

#     if level == "info":
#         console.print(full_msg)
#     elif level == "warn":
#         console.print(f"[yellow]{full_msg}[/yellow]")
#     elif level == "error":
#         console.print(f"[red]{full_msg}[/red]")
#     else:
#         console.print(full_msg)

#     if _current_prompt_id and _current_output_folder:
#         log_file = os.path.join(_current_output_folder, f"{_current_prompt_id}.log")
#         with open(log_file, "a", encoding="utf-8") as f:
#             f.write(full_msg + "\n")





#######
# utils/logger.py
import sys
import os
from datetime import datetime
from rich.console import Console

console = Console()
_current_prompt_id = None
_current_output_folder = None
_original_stdout = sys.stdout
_original_stderr = sys.stderr

class DualOutput:
    def __init__(self, original, log_file_path):
        self.original = original
        self.log_file_path = log_file_path

    def write(self, message):
        self.original.write(message)
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(message)

    def flush(self):
        self.original.flush()

def set_current_prompt(prompt_id: str, output_folder: str):
    global _current_prompt_id, _current_output_folder
    global _original_stdout, _original_stderr

    _current_prompt_id = prompt_id
    _current_output_folder = output_folder
    os.makedirs(_current_output_folder, exist_ok=True)

    log_file_path = os.path.join(_current_output_folder, f"{_current_prompt_id}.log")

    # Redirect sys.stdout and sys.stderr
    sys.stdout = DualOutput(_original_stdout, log_file_path)
    sys.stderr = DualOutput(_original_stderr, log_file_path)

    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp} Logging started for prompt `{prompt_id}`\n")

def restore_console_output():
    global _original_stdout, _original_stderr
    sys.stdout = _original_stdout
    sys.stderr = _original_stderr


