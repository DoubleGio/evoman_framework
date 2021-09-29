

for i in range(1,3):
    enemy = i
    experiment_loc= 'Pygad_enemy' + str(enemy) + '_experiment5'
    exec(open("EA1.py").read())