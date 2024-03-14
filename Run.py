import subprocess

for cluster in [0, 1, 2, 3]:
    for run in [1, 2, 3]:

        for target_name, target_signal in [("Maximize", "np.ones")]:

            preference_trask = True
            script_path = './PCGEnvironment_GA.py'
            command = f'cd Desktop/Affect-Driven-RL && conda activate unity_gym && python {script_path} {cluster} {run} {target_name} {target_signal} {preference_trask}'
            subprocess.run(['wt', '-p', 'Command Prompt', 'cmd', '/c', command], shell=True)
