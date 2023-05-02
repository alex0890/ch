from cx_Freeze import setup, Executable
import sys
import os

# Include the dependent files in the build
include_files = ["bot.json", "random_responses.py"]

# Set the base for the executable
base = None
if sys.platform == "win32":
    base = "Win32GUI"

# Define the executable and options
exe = Executable(
    script="main.py",
    base=base,
    icon=None  # You can add an icon file path here, if you want to set an icon for your executable
)

setup(
    name="PythonChatBot",
    version="1.0",
    description="My Python ChatBot Application",
    options={"build_exe": {"include_files": include_files}},
    executables=[exe],
)
