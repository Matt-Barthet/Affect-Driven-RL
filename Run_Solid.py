import subprocess

# for cluster in [0, 1, 2, 3]:
for run in range(1, 6):
    for weight in [0, 0.5, 1]:
        # for target_name, target_signal in [("Minimize", "np.zeros"), ("Maximize", "np.ones"), ("imitate", "imitate")]:
        script_path = './Solid_PPOEnvironment.py'
        command = f'cd Work Projects/Affect-Driven-RL && conda activate unity_gym && python {script_path} {run} {weight}'
        subprocess.run(['wt', '-p', 'Command Prompt', 'cmd', '/c', command], shell=True)
