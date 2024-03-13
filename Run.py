import subprocess

for cluster in [0, 2, 3]:
    for run in [1]:
        for target_name, target_signal in [("Maximize", "np.ones"), ("Minimize", "np.zeros")]:

            preference_trask = True
            # Assuming your script is named 'your_script.py' and located at 'C:\\path\\to\\your\\script\\'
            script_path = './PCGEnvironment_GA.py'

            # Command to activate Conda environment and run your script
            command = f'cd Desktop/Affect-Driven-RL && conda activate unity_gym && python {script_path} {cluster} {run} {target_name} {target_signal} {preference_trask}'

            # Open a new Windows Terminal and execute the command
            subprocess.run(['wt', '-p', 'Command Prompt', 'cmd', '/c', command], shell=True)
