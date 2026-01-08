import subprocess
import os

def run_r_script(filename: str, args: list = None) -> str:
    try:
        # Ensure the file exists
        if not os.path.isfile(filename):
            return f"Error: R script file '{filename}' not found."

        # Build command with arguments
        command = ["Rscript", filename]
        if args:
            command.extend(args)
        
        print(f"Running R script with command: {' '.join(command)}")
        print(f"Working directory: {os.getcwd()}")

        # Run the R script via subprocess
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120  # Increase timeout for long computations
        )

        output = result.stdout.strip()
        error = result.stderr.strip()

        print(f"R script return code: {result.returncode}")
        if output:
            print(f"R script stdout: {output}")
        if error:
            print(f"R script stderr: {error}")

        if result.returncode != 0:
            return f"Script failed with return code {result.returncode}.\nStderr:\n{error}\nStdout:\n{output}"
        return f"Script ran successfully.\nOutput:\n{output}"
    except subprocess.TimeoutExpired:
        return "Error: R script execution timed out."
    except FileNotFoundError as e:
        return f"Error: Command not found. Make sure Rscript is installed and in PATH. Details: {str(e)}"
    except Exception as e:
        return f"Exception occurred: {str(e)}"
