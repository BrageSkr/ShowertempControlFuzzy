import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Define input variables
temperature = ctrl.Antecedent(np.arange(-20, 20, 0.01), 'temperature')
flow = ctrl.Antecedent(np.arange(-1, 1, 0.01), 'flow')

# Define output variables
cold = ctrl.Consequent(np.arange(-1, 1, 0.01), 'cold')
hot = ctrl.Consequent(np.arange(-1, 1, 0.01), 'hot')

# Define membership functions for input variable 'temperature'
temperature['cold'] = fuzz.trapmf(temperature.universe, [-20, -20, -15, 0])
temperature['good'] = fuzz.trimf(temperature.universe, [-10, 0, 10])
temperature['hot'] = fuzz.trapmf(temperature.universe, [0, 15, 20, 20])

# Define membership functions for input variable 'flow'
flow['soft'] = fuzz.trapmf(flow.universe, [-1, -1, -0.8, 0])
flow['good'] = fuzz.trimf(flow.universe, [-0.4, 0, 0.4])
flow['hard'] = fuzz.trapmf(flow.universe, [0, 0.8, 1, 1])

# Define membership functions for output variable 'cold'
cold['openFast'] = fuzz.trimf(hot.universe, [0.3, 0.6, 1])
cold['openSlow'] = fuzz.trimf(cold.universe, [0, 0.3, 0.6])
cold['closeSlow'] = fuzz.trimf(cold.universe, [-0.6, -0.3, 0])
cold['closeFast'] = fuzz.trimf(cold.universe, [-1, -0.6, -0.3])
cold['steady'] = fuzz.trimf(cold.universe, [-0.2, 0, 0.2])

# Define membership functions for output variable 'hot'
hot['openFast'] = fuzz.trimf(hot.universe, [0.3, 0.6, 1])
hot['openSlow'] = fuzz.trimf(hot.universe, [0, 0.3, 0.6])
hot['closeSlow'] = fuzz.trimf(hot.universe, [-0.6, -0.3, 0])
hot['closeFast'] = fuzz.trimf(hot.universe, [-1, -0.6, -0.3])
hot['steady'] = fuzz.trimf(hot.universe, [-0.2, 0, 0.2])

# Define rules
rule1 = ctrl.Rule(temperature['cold'] & flow['soft'], (cold['openSlow'], hot['openFast']))
rule2 = ctrl.Rule(temperature['cold'] & flow['good'], (cold['closeSlow'], hot['openSlow']))
rule3 = ctrl.Rule(temperature['cold'] & flow['hard'], (cold['closeFast'], hot['closeSlow']))
rule4 = ctrl.Rule(temperature['good'] & flow['soft'], (cold['openSlow'], hot['openSlow']))
rule5 = ctrl.Rule(temperature['good'] & flow['good'], (cold['steady'], hot['steady']))
rule6 = ctrl.Rule(temperature['good'] & flow['hard'], (cold['closeSlow'], hot['closeSlow']))
rule7 = ctrl.Rule(temperature['hot'] & flow['soft'], (cold['openFast'], hot['openSlow']))
rule8 = ctrl.Rule(temperature['hot'] & flow['good'], (cold['openSlow'], hot['closeSlow']))
rule9 = ctrl.Rule(temperature['hot'] & flow['hard'], (cold['closeSlow'], hot['closeFast']))

# Create control system
shower_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
shower_simulation = ctrl.ControlSystemSimulation(shower_ctrl)

# Input values
input_temp = 15
input_flow = 0.5

# Compute result
shower_simulation.input['temperature'] = input_temp
shower_simulation.input['flow'] = input_flow
shower_simulation.compute()

# Print result
print(f"Input temperature: {input_temp} °C")
print(f"Input water flow: {input_flow} units")
print(f"Suggested cold output: {shower_simulation.output['cold']} units")
print(f"Suggested hot output: {shower_simulation.output['hot']} units")

# Visualization
temperature.view(sim=shower_simulation)
flow.view(sim=shower_simulation)
cold.view(sim=shower_simulation)
hot.view(sim=shower_simulation)
plt.show()