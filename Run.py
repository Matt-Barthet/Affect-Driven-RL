import subprocess

for cluster in [0, 1, 2, 3]:
    for run in [4]:
        for target_name, target_signal in [("Minimize", "np.zeros"), ("Maximize", "np.ones"), ("imitate", "imitate")]:
            preference_trask = True
            script_path = './PCGEnvironment_Go-Explore.py'
            command = f'cd Desktop/Affect-Driven-RL && conda activate unity_gym && python {script_path} {cluster} {run} {target_name} {target_signal} {preference_trask} False'
            subprocess.run(['wt', '-p', 'Command Prompt', 'cmd', '/c', command], shell=True)
