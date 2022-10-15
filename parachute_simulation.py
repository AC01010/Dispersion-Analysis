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
DATE = (2022, 11, 20, 12)
DIAM = 10

number_of_simulations = 10

# Toggle True/False to deploy parachutes or land ballistic
parachutes = True

# Rocket data that will vary between each iteration, in the form of (data, uncertainty),
# varies along a normal distribution

analysis_parameters = {
    "thrustScaleB": (1, 0.05),
    "thrustScaleS": (1, 0.05),
    "ignitionDelay": (x0["ignition_delay"], 3),
    "throatRadiusB": (np.sqrt(x0["A_t_1"]/np.pi)*0.0254, 0),
    "throatRadiusS": (np.sqrt(x0["A_t_2"]/np.pi)*0.0254, 0),
    "massNR": (event6["m"], 0),
    "inertiaINR": (event6["I_I"], 0),
    "inertiaZNR": (0.5*event6["m"]*(x0["diam"]/2*0.0254)**2, 0),
    "finSpan1": (x0["span_1"], 0), #inches
    "finRoot1": (x0["root_1"], 0), #inches
    "finTip1": (x0["tip_1"], 0), #inches
    "finSpan2": (x0["span_2"], 0), #inches
    "finRoot2": (x0["root_2"], 0), #inches
    "finTip2": (x0["tip_2"], 0), #inches
    "boosterParachute": (x0["diachute_1"], 0),
    "sustainerParachute": (x0["diachute_2"], 0),
    "noseConeParachute": (x0["diachute_3"], 0),
    "heading": (0, 5),
    "inclination": (90, 1),
    "ensembleMember": list(range(10)),
    "chuteFailure": list(range(100)),
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
obj1_output_file = open('obj1_analysis_outputs'+'.disp_outputs.txt', 'w')
obj2_output_file = open('obj2_analysis_outputs'+'.disp_outputs.txt', 'w')

# Creates an environment object using launch location data
Env = Environment(
    railLength=7.3152,
    latitude=33.9127,
    longitude=-84.9417,
    elevation=1300,
    date=(2022,10,20,12) # change this <- this is the date of the launch
    )
#Env.allInfo()
# Sets the atmospheric model to an Ensemble type so a new forecast can be used for each iteration
Env.setAtmosphericModel(type='Ensemble', file='GEFS')
# Env.setAtmosphericModel(type='CustomAtmosphere') # wind assumed 0

#Env.allInfo()

# Parachute triggers
def drogueTrigger(p, y):
    return True if y[2] < 31100 else False


def mainTrigger(p, y):
    return True if y[5] < 0 else False


# Toggles the parachute cd depending on if the simulation is using parachutes, sets cd to zero to make parachutes have
# no effect
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

    # Event6Main = Event6.addParachute('Main',
    #                                 CdS=cd*np.pi*(0.3048*setting['noseConeParachute']/2)**2,
    #                                 trigger=mainTrigger,
    #                                 samplingRate=105,
    #                                 lag=1.5,
    #                                 noise=(0, 8.3, 0.5))
    initial_v = 60 # important !! ✈️ <- change this
    initial_bearing = 128 # <- and this
    drop_elevation = 30000 # <- and this

    obj1_mass = 10 # <- and this
    obj2_mass = 10 # <- and this
    c_d = 0.97 # <- and this

    x_velocity = initial_v*np.sin(np.radians(initial_bearing))
    y_velocity = initial_v*np.cos(np.radians(initial_bearing))

    initial = [0,0,0,Env.elevation+drop_elevation,x_velocity,y_velocity,0,0,0,0,0,0,0,0.0]

    initial_drop = Flight(rocket=Event6, environment=Env, initialSolution=initial, timeOvershoot = True, maxTime=6)
    # print(initial_drop.solution)
    # breakpoint()
    initial_drop.postProcess()  

    after_drop = initial_drop.solution[-1] #[x for x in initial_drop.solution if x[6] < 0][-1]
    after_drop = [0.0, *after_drop[1:]] #ඞ


    Object1 = Rocket( 
        motor=motor_empty,
        radius=x0["diam"]/2*0.0254,
        mass=obj1_mass,
        inertiaI=setting['inertiaINR'],
        inertiaZ=setting['inertiaZNR'],
        distanceRocketNozzle=event6["d_nozzle"],
        distanceRocketPropellant=event6["d_prop"],
        powerOffDrag="./CDs/s_power_off.csv",
        powerOnDrag="./CDs/s_power_on.csv"
    )

    Object2 = Rocket( 
        motor=motor_empty,
        radius=x0["diam"]/2*0.0254,
        mass=obj2_mass,
        inertiaI=setting['inertiaINR'],
        inertiaZ=setting['inertiaZNR'],
        distanceRocketNozzle=event6["d_nozzle"],
        distanceRocketPropellant=event6["d_prop"],
        powerOffDrag="./CDs/s_power_off.csv",
        powerOnDrag="./CDs/s_power_on.csv"
    )

    Object1.setRailButtons([0.1, -0.2])
    Object2.setRailButtons([0.1, -0.2])

    Main1 = Object1.addParachute('Main1',
        CdS=cd*np.pi*(0.3048*setting['noseConeParachute']/2)**2,            
        trigger=drogueTrigger,
        samplingRate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5))

    Main2 = Object2.addParachute('Main2',
        CdS=cd*np.pi*(0.3048*setting['noseConeParachute']/2)**2,            
        trigger=drogueTrigger,
        samplingRate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5))

    obj1_flight = Flight(rocket=Object1, environment=Env, initialSolution=after_drop, timeOvershoot = True, maxTime=4000)
    obj1_flight.postProcess()

    obj2_flight = Flight(rocket=Object2, environment=Env, initialSolution=after_drop, timeOvershoot = True, maxTime=4000)
    obj2_flight.postProcess()

    df_settings = pd.DataFrame([setting])
    df_obj1 = pd.DataFrame([[obj1_flight.apogee, obj1_flight.apogeeX, obj1_flight.apogeeY, obj1_flight.xImpact, obj1_flight.yImpact, obj1_flight.impactVelocity]], columns = ['apogeeAltitude', 'apogeeX', 'apogeeY', 'impactX', 'impactY', 'impactVelocity'])
    df_obj2 = pd.DataFrame([[obj2_flight.apogee, obj2_flight.apogeeX, obj2_flight.apogeeY, obj2_flight.xImpact, obj2_flight.yImpact, obj2_flight.impactVelocity]], columns = ['apogeeAltitude', 'apogeeX', 'apogeeY', 'impactX', 'impactY', 'impactVelocity'])
    df_nose = pd.DataFrame([[initial_drop.apogee, initial_drop.apogeeX, initial_drop.apogeeY, initial_drop.xImpact, initial_drop.yImpact, initial_drop.impactVelocity]], columns = ['apogeeAltitude', 'apogeeX', 'apogeeY', 'impactX', 'impactY', 'impactVelocity'])

    if os.path.exists("df_nose.csv"):
        if os.stat("df_nose.csv").st_size <= 4:
            df_obj1.to_csv("df_obj1", mode='a', header=True, index=False)
            df_obj2.to_csv("df_obj2", mode='a', header=True, index=False)
            df_nose.to_csv("df_nose.csv", mode = 'a', index = False, header = True)
        else:
            df_obj1.to_csv("df_obj1", mode='a', header=False, index=False)
            df_obj2.to_csv("df_obj2", mode='a', header=False, index=False)
            df_nose.to_csv("df_nose.csv", mode = 'a', index = False, header = False)
    else:
        df_obj1.to_csv("df_obj1", mode='a', header=True, index=False)
        df_obj2.to_csv("df_obj2", mode='a', header=True, index=False)
        df_nose.to_csv("df_nose.csv", mode = 'a', index = False, header = True)

    export_flight_data(obj1_flight, obj1_output_file)
    export_flight_data(obj2_flight, obj2_output_file)
    export_flight_data(initial_drop, nose_output_file)

    print("Simulations Completed: " + str(i))

# Closes export files
obj1_output_file.close()
obj2_output_file.close()
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
obj1_results = extractData('obj1_analysis_outputs')
obj2_results = extractData('obj2_analysis_outputs')


# Print number of flights simulated
print('Total number of simulations: ', i)

print(f'Nose Impact X Position -         Mean Value: {np.mean(nose_results["impactX"]):0.3f} m')
print(f'Nose Impact X Position - Standard Deviation: {np.std(nose_results["impactX"]):0.3f} m')

print(f'Nose Impact Y Position -         Mean Value: {np.mean(nose_results["impactY"]):0.3f} m')
print(f'Nose Impact Y Position - Standard Deviation: {np.std(nose_results["impactY"]):0.3f} m')

print(f'Object 1 Impact X Position -         Mean Value: {np.mean(obj1_results["impactX"]):0.3f} m')
print(f'Object 1 Impact X Position - Standard Deviation: {np.std(obj1_results["impactX"]):0.3f} m')

print(f'Object 1 Impact Y Position -         Mean Value: {np.mean(obj1_results["impactY"]):0.3f} m')
print(f'Object 1 Impact Y Position - Standard Deviation: {np.std(obj1_results["impactY"]):0.3f} m')

print(f'Object 2 Impact X Position -         Mean Value: {np.mean(obj2_results["impactX"]):0.3f} m')
print(f'Object 2 Impact X Position - Standard Deviation: {np.std(obj2_results["impactX"]):0.3f} m')

print(f'Object 2 Impact Y Position -         Mean Value: {np.mean(obj2_results["impactY"]):0.3f} m')
print(f'Object 2 Impact Y Position - Standard Deviation: {np.std(obj2_results["impactY"]):0.3f} m')

end = time.time()
print("Elapsed time: ", end - start)

# Import background map
img = mpimg.imread("staticmap.png")

# Gets the data to be graphed
ApogeeX = np.array(nose_results['apogeeX'])
ApogeeY = np.array(nose_results['apogeeY'])
noseImpactX = np.array(nose_results['impactX'])
noseImpactY = np.array(nose_results['impactY'])
obj1ImpactX = np.array(obj1_results['impactX'])
obj1ImpactY = np.array(obj1_results['impactY'])
obj2ImpactX = np.array(obj2_results['impactX'])
obj2ImpactY = np.array(obj2_results['impactY'])
df_nose = pd.DataFrame.from_dict(nose_results) 

df_nose.to_csv("df_nose.csv", index = False)

# Function to calculate eigen values
def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]


# Create plot figure
def plot(x,y, name):
    plt.figure(num=None, figsize=(8, 8), dpi=150, facecolor='w', edgecolor='k')
    ax = plt.subplot(111)

    # Calculate error ellipses for impact
    impactCov = np.cov(np.negative(x), np.negative(y))
    impactVals, impactVecs = eigsorted(impactCov)
    impactTheta = np.degrees(np.arctan2(*impactVecs[:,0][::-1]))
    impactW, impactH = 2 * np.sqrt(impactVals)
    plt.scatter(np.negative(x), np.negative(y), s=5, marker='v', color='red', label=f'{name}Landing Point')
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
        impactEll = Ellipse(xy=(np.mean(np.negative(x)), np.mean(np.negative(y))),
                            width=impactW*j, height=impactH*j,
                            angle=impactTheta, color='black')
        impactEll.set_facecolor((0, 0, 1, 0.2))
        impact_ellipses.append(impactEll)
        ax.add_artist(impactEll)


    filename = f'{name}Dispersion Analysis Map'
    plt.savefig(str(filename) + '.pdf', bbox_inches='tight', pad_inches=0)

    plt.show()

plot(obj1ImpactX,obj1ImpactY, 'Object 1 ')
plot(obj2ImpactX,obj2ImpactY, 'Object 2 ')

print("Elapsed time: ", end - start)