import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Define input variables
temperature = ctrl.Antecedent(np.arange(-20, 20, 0.001), 'temperature')
flow = ctrl.Antecedent(np.arange(-1, 1, 0.001), 'flow')

# Define output variables
cold = ctrl.Consequent(np.arange(-1, 1, 0.001), 'cold')
hot = ctrl.Consequent(np.arange(-1, 1, 0.001), 'hot')

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


# Simulation parameters
shower_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
shower_simulation = ctrl.ControlSystemSimulation(shower_ctrl)

# Simulation parameters
num_steps = 50
temp_setpoints = [15, 15, 15, 15, 15, 15, 15, 30, 30, 30, 30, 30, 30, 30]  # Periodic changes in temperature setpoint
flow_setpoints = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1, 1, 1, 1]



time_steps = []
temp_actual = [15]
flow_actual = [0.5]

# Simulate the control system
for i in range(1, num_steps):  # Start from the second time step
    step = i % len(temp_setpoints)  # Get the current setpoints
    temp_setpoint = temp_setpoints[step]
    flow_setpoint = flow_setpoints[step]

    # Calculate the error between setpoint and actual value
    temp_error = temp_setpoint - temp_actual[i - 1]  # Error for temperature
    flow_error = flow_setpoint - flow_actual[i - 1]  # Error for flow

    # Input the errors to the fuzzy control system
    shower_simulation.input['temperature'] = temp_error
    shower_simulation.input['flow'] = flow_error
    shower_simulation.compute()

    # Get the control actions
    temp_adjustment = (shower_simulation.output['cold'] - shower_simulation.output['hot']) * 30
    flow_adjustment = (-shower_simulation.output['hot']  + shower_simulation.output['cold'])

    # Update the actual values based on the control actions
    temp_actual.append(temp_actual[i - 1] + temp_adjustment)
    flow_actual.append(max(min(flow_actual[i - 1] + flow_adjustment, 1.0), 0.0))


    time_steps.append(i)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot( [temp_setpoints[i % len(temp_setpoints)] for i in range(num_steps)], label='Temperature Setpoints')
plt.plot( temp_actual, label='Temperature Adjustments')
plt.title('Temperature actual')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

plt.plot( [flow_setpoints[i % len(flow_setpoints)] for i in range(num_steps)], label='Flow Setpoints')
plt.plot( (flow_actual), label='Flow Adjustments')
plt.title('Flow actual')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()

