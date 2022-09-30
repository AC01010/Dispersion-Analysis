from rocketpy import Environment, Rocket, SolidMotor, Flight
import matplotlib.image as mpimg
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from numpy.random import normal, choice
from datetime import date
year, month, day = str(date.today()).split('-')
import time
import csv
import os 

start = time.time()

df = list(csv.reader(open('weights.csv')))

# Gets the mass, center of mass, and length of the specified section given by the section number.
def spliceBySection(section, columnToSearch):
    minPos = 1000
    maxPos = 0
    masses = []
    coms = []
    indices = []
    finLoc = []
    propLoc = -1
    nozzLoc = -1
    for i in range(1, len(df)):
        pos = float(df[i][7])
        localCoM = float(df[i][6])
        length = float(df[i][5])
        mass = float(df[i][4])
        simulationSection = df[i][columnToSearch]
        globalCoM = pos + localCoM
        if simulationSection != "a" and simulationSection != "" and simulationSection != "x" and int(simulationSection) == section:
            if pos < minPos:
                minPos = pos
            if pos + length > maxPos:
                maxPos = pos + length
            masses.append(mass)
            coms.append(globalCoM)
            indices.append(i)
            if df[i][1] == "Fins":
                finLoc.insert(0, pos)
        elif simulationSection == "a":
            if df[i][1] == "Propellant":
                propLoc = pos + localCoM
                nozzLoc = float(df[i - 1][7]) + float(df[i - 1][5])
    
    # Takes out length that is shared with a section higher up on the rocket
    for i in range(1, len(df)):
        pos = float(df[i][7])
        localCoM = float(df[i][6])
        length = float(df[i][5])
        mass = float(df[i][4])
        simulationSection = df[i][columnToSearch]
        # If the piece is not in this section, see if the piece overlaps the current section
        if simulationSection != "a" and simulationSection != "" and simulationSection != "x" and int(simulationSection) < section:
            # If the piece impedes on the current section, the current section is shortened from the top
            if pos + length > minPos:
                minPos = pos + length
        elif simulationSection == "x":
            if pos > minPos and pos < maxPos:
                maxPos = pos
            elif pos < minPos and pos + length > minPos:
                minPos = pos + length
    com = 0
    mass = sum(masses)
    for i in range(0, len(coms)):
        com += coms[i] * masses[i] / mass
    
    moi = 0
    for i in range(0, len(indices)):
        index = indices[i]
        pos = float(df[index][7])
        localCoM = float(df[index][6])
        mass2 = float(df[index][4])
        moi += float(df[index][8]) - mass2 * (pos + localCoM) * (pos + localCoM) + mass2 * (pos + localCoM - com) * (pos + localCoM - com)
        #moi += float(df[index][8]) - mass2 * (pos + localCoM - com) * (pos + localCoM - com)
    return {"m": mass, "cg": com - minPos, "l": maxPos - minPos, "I_I": moi, "d_finsets": [com - loc for loc in finLoc],
            "d_nozzle": nozzLoc - com, "d_prop": propLoc - com, "d_nose": com - float(df[1][5]) }


# Rocket stages/simulation events
event6 = spliceBySection(1,15)


# Input parameters
x0 = {
'A_t_1': 2, #booster motor
'odia_1': 5,
'dia_1': 2.329,
'length_1': 53.3945,
'A_t_2': 2, #sustainer motor
'odia_2': 5,
'dia_2': 2.329,
'length_2': 53.3945,
'root_1': 20, #booster fins
'span_1': 7.7,
'tip_1': 8,
'sweep_1': 12,
'root_2': 20, #sustainer fins
'span_2': 7.2,
'tip_2': 5,
'sweep_2': 15,
'ignition_delay': 11,
'diachute_1': 4, #booster
'diachute_2': 4, #sustainer
'diachute_3': 3, #nose
'diam': 6,
'number_1': 5,
'number_2': 5,
'motor1_name': "PmotorRasp2.eng",
'motor2_name': "PmotorRasp2.eng",
'wind speed (mph)': 10
}

mpl.rcParams['figure.figsize'] = [8, 5]
mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['figure.titlesize'] = 14

# Time for weather data, adjust in 6 hour increments
DATE = (2022, 9, 30, 12)
DIAM = 10

number_of_simulations = 20

# Toggle True/False to deploy parachutes or land ballistic
parachutes = True

# Rocket data that will vary between each iteration, in the form of (data, uncertainty),
# varies along a normal distribution
analysis_parameters = {
    "massNR": (event6["m"], 0),
    "inertiaINR": (event6["I_I"], 0),
    "inertiaZNR": (0.5*event6["m"]*(x0["diam"]/2*0.0254)**2, 0),
    "noseConeParachute": (x0["diachute_3"], 0),
    "heading": (0, 5),
    "inclination": (90, 1),
    "ensembleMember": list(range(10)),
}

# Helper Function to export data from the flights to a file to be accessed for analysis later
def export_flight_data(data, file):
    # Generate flight data
    result = {
        "apogeeAltitude": data.apogee - Env.elevation,
        "apogeeX": data.apogeeX,
        "apogeeY": data.apogeeY,
        "impactX": data.xImpact,
        "impactY": data.yImpact,
        "impactVelocity": data.impactVelocity,
    }
    # Export data
    file.write(str(result) + '\n')


# Function to create the settings for each simulation
def flight_settings(analysis_parameters, total_number):
    i = 0
    while i < total_number:
        flight_setting = {}
        for parameter_key, parameter_value in analysis_parameters.items():
            # Gets a value for each of the analysis parameter, if the parameter has an uncertainty it varies along a
            # normal distribution
            if type(parameter_value) is tuple:
                flight_setting[parameter_key] = normal(*parameter_value)
            # If the parameter is a discrete value it chooses one of the values randomly
            else:
                flight_setting[parameter_key] = choice(parameter_value)
        i += 1
        yield flight_setting


# Creates and opens files to export the flight data to
nose_output_file = open('nose_analysis_outputs'+'.disp_outputs.txt', 'w')

# Creates an environment object using launch location data
Env = Environment(
    railLength=7.3152,
    latitude=33.9127,
    longitude=-84.9417,
    elevation=1300,
    date=(2022,10,1,12) # change this <- this is the date of the launch
    )
#Env.allInfo()
# Sets the atmospheric model to an Ensemble type so a new forecast can be used for each iteration
Env.setAtmosphericModel(type='Ensemble', file='GEFS')
# Env.setAtmosphericModel(type='CustomAtmosphere') # wind assumed 0

#Env.allInfo()

# Parachute triggers
def drogueTrigger(p, y):
    return True if y[5] < 0 else False


def mainTrigger(p, y):
    return True if y[5] < 0 else False


# Toggles the parachute cd depending on if the simulation is using parachutes, sets cd to zero to make parachutes have
# no effect
cd = 0
if parachutes:
    cd = 0.97

# Counts the number of simulations that have run
i = 0

# Iterates the input number of times, using a new flight setting for each iteration
for setting in flight_settings(analysis_parameters, number_of_simulations):
    # Sets a new forecast for the weather data
    Env.selectEnsembleMember(setting["ensembleMember"])
    i += 1
    
    motor_empty = SolidMotor( #Empty Motor
        thrustSource="./Motors/Empty.eng",
        burnOut=0.02,
        grainNumber=1,
        grainSeparation=0.0015875,
        grainDensity=0.0001,
        grainOuterRadius=(x0["odia_1"]*0.0254)/2,
        grainInitialInnerRadius=(x0["dia_1"]*0.0254)/2,
        grainInitialHeight=(x0["length_1"]/x0["number_1"]*0.0254),
        nozzleRadius=0.113284/2,
        throatRadius=np.sqrt(x0["A_t_1"]/np.pi)*0.0254,
        interpolationMethod='linear'
    )

    Event6 = Rocket( #Nose Recovery
        motor=motor_empty,
        radius=x0["diam"]/2*0.0254,
        mass=setting['massNR'],
        inertiaI=setting['inertiaINR'],
        inertiaZ=setting['inertiaZNR'],
        distanceRocketNozzle=event6["d_nozzle"],
        distanceRocketPropellant=event6["d_prop"],
        powerOffDrag="./CDs/s_power_off.csv",
        powerOnDrag="./CDs/s_power_on.csv"
    )

    Event6.setRailButtons([0.1, -0.2])

    Event6Main = Event6.addParachute('Main',
                                    CdS=cd*np.pi*(0.3048*setting['noseConeParachute']/2)**2,
                                    trigger=mainTrigger,
                                    samplingRate=105,
                                    lag=1.5,
                                    noise=(0, 8.3, 0.5))
    initial_v = 4000 # important !! ✈️ <- change this
    initial_bearing = 128 # <- and this
    drop_elevation = 30000 # <- and this

    x_velocity = initial_v*np.sin(np.radians(initial_bearing))
    y_velocity = initial_v*np.cos(np.radians(initial_bearing))

    initial = [0,0,0,Env.elevation+drop_elevation,x_velocity,y_velocity,0,0,0,0,0,0,0,0.0]
    #initial = None

    noseRecovery = Flight(rocket=Event6, environment=Env, initialSolution=initial, timeOvershoot = True, maxTime=4000)
    noseRecovery.postProcess()

    df_nose = pd.DataFrame([[noseRecovery.apogee, noseRecovery.apogeeX, noseRecovery.apogeeY, noseRecovery.xImpact, noseRecovery.yImpact, noseRecovery.impactVelocity]], columns = ['apogeeAltitude', 'apogeeX', 'apogeeY', 'impactX', 'impactY', 'impactVelocity'])

    if os.path.exists("df_nose.csv"):
        if os.stat("df_nose.csv").st_size <= 4:
            df_nose.to_csv("df_nose.csv", mode = 'a', index = False, header = True)
        else:
            df_nose.to_csv("df_nose.csv", mode = 'a', index = False, header = False)
    else:
        df_nose.to_csv("df_nose.csv", mode = 'a', index = False, header = True)


    export_flight_data(noseRecovery, nose_output_file)

    print("Simulations Completed: " + str(i))

# Closes export files
nose_output_file.close()

# Extracts the data from each export file
def extractData(filename):
    results = {
        "apogeeAltitude": [],
        "apogeeX": [],
        "apogeeY": [],
        "impactX": [],
        "impactY": [],
        "impactVelocity": [],
    }
    output_file = open(str(filename)+'.disp_outputs.txt', 'r+')
    for line in output_file:
        if line[0] != '{': continue
        flight_result = eval(line)
        for parameter_key, parameter_value in flight_result.items():
            results[parameter_key].append(parameter_value)
    output_file.close()
    return results


nose_results = extractData('nose_analysis_outputs')

# Print number of flights simulated
print('Total number of simulations: ', i)

print(f'Nose Impact X Position -         Mean Value: {np.mean(nose_results["impactX"]):0.3f} m')
print(f'Nose Impact X Position - Standard Deviation: {np.std(nose_results["impactX"]):0.3f} m')

print(f'Nose Impact Y Position -         Mean Value: {np.mean(nose_results["impactY"]):0.3f} m')
print(f'Nose Impact Y Position - Standard Deviation: {np.std(nose_results["impactY"]):0.3f} m')

end = time.time()
print("Elapsed time: ", end - start)

# Import background map
img = mpimg.imread("staticmap.png")

# Gets the data to be graphed
ApogeeX = np.array(nose_results['apogeeX'])
ApogeeY = np.array(nose_results['apogeeY'])
noseImpactX = np.array(nose_results['impactX'])
noseImpactY = np.array(nose_results['impactY'])
df_nose = pd.DataFrame.from_dict(nose_results) 

df_nose.to_csv("df_nose.csv", index = False)

# Function to calculate eigen values
def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]


# Create plot figure
plt.figure(num=None, figsize=(8, 8), dpi=150, facecolor='w', edgecolor='k')
ax = plt.subplot(111)

# Calculate error ellipses for impact
impactCov = np.cov(np.negative(noseImpactX), np.negative(noseImpactY))
impactVals, impactVecs = eigsorted(impactCov)
impactTheta = np.degrees(np.arctan2(*impactVecs[:,0][::-1]))
impactW, impactH = 2 * np.sqrt(impactVals)
plt.scatter(np.negative(noseImpactX), np.negative(noseImpactY), s=5, marker='v', color='blue', label='Nose Cone Landing Point')
plt.legend()

ax.set_title('1$\sigma$, 2$\sigma$ and 3$\sigma$ Dispersion Ellipses: Landing Points')
ax.set_ylabel('North (m)')
ax.set_xlabel('East (m)')
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='both', which='minor', labelsize=8)

# Add background image to plot, translate map by changing dx and dy
dx = 0
dy = 0
plt.imshow(img, zorder=0, extent=[-35000-dx, 35000-dx, -35000-dy, 35000-dy])
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlim(-35000, 35000)
plt.ylim(-35000, 35000)

# Draw error ellipses for nose cone impact
impact_ellipses = []
for j in [1, 2, 3]:
    impactEll = Ellipse(xy=(np.mean(np.negative(noseImpactX)), np.mean(np.negative(noseImpactY))),
                        width=impactW*j, height=impactH*j,
                        angle=impactTheta, color='black')
    impactEll.set_facecolor((0, 0, 1, 0.2))
    impact_ellipses.append(impactEll)
    ax.add_artist(impactEll)


filename = 'Dispersion Analysis Map'
plt.savefig(str(filename) + '.pdf', bbox_inches='tight', pad_inches=0)

plt.show()

print("Elapsed time: ", end - start)