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
event1 = spliceBySection(1,10)
event2 = spliceBySection(1,11)
event3 = spliceBySection(1,12)
event4 = spliceBySection(1,13)
event5 = spliceBySection(1,14)
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







motor_file = open("./Motors/%s" %x0["motor1_name"], 'r')
thrust1 = str.splitlines(motor_file.read())
thrust1 = thrust1[4:len(thrust1)] 
burnOut_1 = float(str.split(thrust1[-1])[0])
motor_file.close()
time1 = np.array([str.split(x)[0] for x in thrust1])
time1 = pd.to_numeric(time1, errors='coerce') 
time1 = time1[~np.isnan(time1)]
thrust1 = np.array([str.split(x)[1] for x in thrust1])
thrust1 = pd.to_numeric(thrust1, errors='coerce') 
thrust1 = thrust1[~np.isnan(thrust1)]
motor_file.close()




motor_file = open("./Motors/%s" %x0["motor2_name"], 'r')
thrust2 = str.splitlines(motor_file.read())
thrust2 = thrust2[4:len(thrust2)] 
burnOut_2 = float(str.split(thrust2[-1])[0])
time2 = np.array([str.split(x)[0] for x in thrust2])
time2 = pd.to_numeric(time2, errors='coerce') 
time2 = time2[~np.isnan(time2)]
thrust2 = np.array([str.split(x)[1] for x in thrust2])
thrust2 = pd.to_numeric(thrust2, errors='coerce') 
thrust2 = thrust2[~np.isnan(thrust2)]

motor_file.close()


mpl.rcParams['figure.figsize'] = [8, 5]
mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['figure.titlesize'] = 14

# Time for weather data, adjust in 6 hour increments
DATE = (2022, 7, 22, 12)
DIAM = 10

number_of_simulations = 200

# Toggle True/False to deploy parachutes or land ballistic
parachutes = True

# Rocket data that will vary between each iteration, in the form of (data, uncertainty),
# varies along a normal distribution
analysis_parameters = {
    "thrustScaleB": (1, 0.05),
    "thrustScaleS": (1, 0.05),
    "ignitionDelay": (x0["ignition_delay"], 3),
    "boosterMotorBurnout": (burnOut_1, 0),
    "sustainerMotorBurnout": (burnOut_2, 0),
    "throatRadiusB": (np.sqrt(x0["A_t_1"]/np.pi)*0.0254, 0),
    "throatRadiusS": (np.sqrt(x0["A_t_2"]/np.pi)*0.0254, 0),
    "massB": (float(event1["m"]), 4),
    "massSC": (event2["m"],2),
    "massS": (event3["m"], 2),
    "massBR": (event4["m"], 1),
    "massSR": (event5["m"], 0.5),
    "massNR": (event6["m"], 0),
    "inertiaIB": (event1["I_I"], 10),
    "inertiaZB": (0.5*event1["m"]*(x0["diam"]/2*0.0254)**2, 0.01),
    "inertiaISC": (event2["I_I"], 1),
    "inertiaZSC": (0.5*event2["m"]*(x0["diam"]/2*0.0254)**2, 0.005),
    "inertiaIS": (event3["I_I"], 1),
    "inertiaZS": (0.5*event3["m"]*(x0["diam"]/2*0.0254)**2, 0.005),
    "inertiaIBR": (event4["I_I"], 0),
    "inertiaZBR": (0.5*event4["m"]*(x0["diam"]/2*0.0254)**2, 0),
    "inertiaISR": (event5["I_I"], 0),
    "inertiaZSR": (0.5*event5["m"]*(x0["diam"]/2*0.0254)**2, 0),
    "inertiaINR": (event6["I_I"], 0),
    "inertiaZNR": (0.5*event6["m"]*(x0["diam"]/2*0.0254)**2, 0),
    "distanceToCMBoosterNC": (event1["d_nose"]+0.0254, 0.0127),
    "distanceToCMBoosterFins1": (event1["d_finsets"][0], 0.0127),
    "distanceToCMBoosterFins2": (event1["d_finsets"][1], 0.0127),
    "distanceToCMSustainerCoastNC": (event2["d_nose"]+0.0254, 0.0127),
    "distanceToCMSustainerCoastFins2": (event2["d_finsets"][0], 0.0127),
    "distanceToCMSustainerNC": (event3["d_nose"]+0.0254, 0.0127),
    "distanceToCMSustainerFins2": (event3["d_finsets"][0], 0.1),
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
booster_output_file = open('booster_analysis_outputs'+'.disp_outputs.txt', 'w')
sustainer_output_file = open('sustainer_analysis_outputs'+'.disp_outputs.txt', 'w')

# Creates an environment object using launch location data
Env = Environment(
    railLength=7.3152,
    latitude=32.990254,
    longitude=-106.974998,
    elevation=1400,
    date=DATE,
    )
# Sets the atmospheric model to an Ensemble type so a new forecast can be used for each iteration
Env.setAtmosphericModel(type='Ensemble', file='GEFS')
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

df_settings = pd.DataFrame()

# Iterates the input number of times, using a new flight setting for each iteration
for setting in flight_settings(analysis_parameters, number_of_simulations):
    # Sets a new forecast for the weather data
    Env.selectEnsembleMember(setting["ensembleMember"])
    #Env.windVelocityX = Env.windVelocityX + 3
    #Env.windVelocityY = Env.windVelocityY + 3
    i += 1

    # if setting["chuteFailure"]<=4:
    #     cd = 0
    # else:
    #     cd  = 0.97

    # Creates motor object

    booster = np.column_stack((time1, thrust1*setting["thrustScaleB"])).tolist()
    motor1 = SolidMotor(
        thrustSource=booster,
        burnOut=setting['boosterMotorBurnout'],
        grainNumber=x0["number_1"],
        grainSeparation=0.0015875,
        grainDensity=1567.4113,
        grainOuterRadius=(x0["odia_1"]*0.0254)/2,
        grainInitialInnerRadius=(x0["dia_1"]*0.0254)/2,
        grainInitialHeight=(x0["length_1"]/x0["number_1"]*0.0254),
        nozzleRadius=0.113284/2,
        throatRadius=setting['throatRadiusB'],
        interpolationMethod='linear'
    )
 
    sustainer = np.column_stack((time2, thrust2*setting["thrustScaleS"])).tolist()
    motor2 = SolidMotor(
        thrustSource=sustainer,
        burnOut=setting['sustainerMotorBurnout'],
        grainNumber=x0["number_2"],
        grainSeparation=0.0015875,
        grainDensity=1567.4113,
        grainOuterRadius=(x0["odia_2"]*0.0254)/2,
        grainInitialInnerRadius=(x0["dia_2"]*0.0254)/2,
        grainInitialHeight=(x0["length_2"]/x0["number_2"]*0.0254),
        nozzleRadius=0.113284/2,
        throatRadius=setting['throatRadiusS'],
        interpolationMethod='linear'
    )
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

    Event1 = Rocket( # Booster
        motor=motor1,
        radius=x0["diam"]/2*0.0254,
        mass=setting['massB'],
        inertiaI=setting['inertiaIB'],
        inertiaZ=setting['inertiaZB'],
        distanceRocketNozzle=float(event1["d_nozzle"]),
        distanceRocketPropellant=float(event1["d_prop"]),
        powerOffDrag="./CDs/b_power_off.csv",
        powerOnDrag="./CDs/b_power_on.csv"
    )
    Event2 = Rocket( #Sustaienr Coast
        motor=motor_empty,
        radius=x0["diam"]/2*0.0254,
        mass=setting['massSC'],
        inertiaI=setting['inertiaISC'],
        inertiaZ=setting['inertiaZSC'],
        distanceRocketNozzle=event2["d_nozzle"],
        distanceRocketPropellant=event2["d_prop"],
        powerOffDrag="./CDs/s_power_off.csv",
        powerOnDrag="./CDs/s_power_on.csv"
    )
    Event3 = Rocket( #Sustainer
        motor=motor2,
        radius=x0["diam"]/2*0.0254,
        mass=setting['massS'],
        inertiaI=setting['inertiaIS'],
        inertiaZ=setting['inertiaZS'],
        distanceRocketNozzle=event3["d_nozzle"],
        distanceRocketPropellant=event3["d_prop"],
        powerOffDrag="./CDs/s_power_off.csv",
        powerOnDrag="./CDs/s_power_on.csv"
    )
    Event4 = Rocket( #Booster Recovery
        motor=motor_empty,
        radius=x0["diam"]/2*0.02544,
        mass=setting['massBR'],
        inertiaI=setting['inertiaIBR'],
        inertiaZ=setting['inertiaZBR'],
        distanceRocketNozzle=event4["d_nozzle"],
        distanceRocketPropellant=event4["d_prop"],
        powerOffDrag="./CDs/s_power_off.csv",
        powerOnDrag="./CDs/s_power_on.csv"
    )
    Event5 = Rocket( #Sustainer Recovery
        motor=motor_empty,
        radius=x0["diam"]/2*0.0254,
        mass=setting['massSR'],
        inertiaI=setting['inertiaISR'],
        inertiaZ=setting['inertiaZSR'],
        distanceRocketNozzle=event5["d_nozzle"],
        distanceRocketPropellant=event5["d_prop"],
        powerOffDrag="./CDs/s_power_off.csv",
        powerOnDrag="./CDs/s_power_on.csv"
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


    Event1.setRailButtons([0.1, -0.2])
    Event2.setRailButtons([0.1, -0.2])
    Event3.setRailButtons([0.1, -0.2])
    Event4.setRailButtons([0.1, -0.2])
    Event5.setRailButtons([0.1, -0.2])
    Event6.setRailButtons([0.1, -0.2])

    # Adds aerodynamic surfaces
    # Booster

    #booster
    NoseCone = Event1.addNose(length=x0["diam"]*6*0.0254, kind="Von Karman", distanceToCM=setting["distanceToCMBoosterNC"])
    FinSet = Event1.addFins(4, span=setting['finSpan1']*0.0254, rootChord=setting['finRoot1']*0.0254, tipChord=setting['finTip1']*0.0254, distanceToCM=setting["distanceToCMBoosterFins1"])
    FinSet = Event1.addFins(4, span=setting['finSpan2']*0.0254, rootChord=setting['finRoot2']*0.0254, tipChord=setting['finTip2']*0.0254, distanceToCM=setting["distanceToCMBoosterFins2"])

    #Sustainer_Coast
    NoseCone = Event2.addNose(length=x0["diam"]*6*0.0254, kind="Von Karman", distanceToCM=setting["distanceToCMSustainerCoastNC"])
    FinSet = Event2.addFins(4, span=x0["span_2"]*0.0254, rootChord=x0["root_2"]*0.0254, tipChord=x0["tip_2"]*0.0254, distanceToCM=setting["distanceToCMSustainerCoastFins2"])

    #Sustianer
    NoseCone = Event3.addNose(length=x0["diam"]*6*0.0254, kind="Von Karman", distanceToCM=setting["distanceToCMSustainerNC"])
    FinSet = Event3.addFins(4, span=x0["span_2"]*0.0254, rootChord=x0["root_2"]*0.0254, tipChord=x0["tip_2"]*0.0254, distanceToCM=setting["distanceToCMSustainerFins2"])


    if setting["chuteFailure"]<=4 and False:
        cd = 0.01
    else:
        cd = 0.97
    if setting["chuteFailure"]>=0:
        # Booster
        Event4Main = Event4.addParachute('Main',
                                        CdS=cd*np.pi*(0.3048*setting['boosterParachute']/2)**2,
                                        trigger=mainTrigger,
                                        samplingRate=105,
                                        lag=1.5,
                                        noise=(0, 8.3, 0.5))

        # Sustainer
        Event5Main = Event5.addParachute('Main',
                                        CdS=cd*np.pi*(0.3048*setting['sustainerParachute']/2)**2,
                                        trigger=mainTrigger,
                                        samplingRate=105,
                                        lag=1.5,
                                        noise=(0, 8.3, 0.5))

        # Nose cone
        Event6Main = Event6.addParachute('Main',
                                        CdS=cd*np.pi*(0.3048*setting['noseConeParachute']/2)**2,
                                        trigger=mainTrigger,
                                        samplingRate=105,
                                        lag=1.5,
                                        noise=(0, 8.3, 0.5))




    #Booster simulation
    booster = Flight(rocket=Event1, environment=Env, inclination=setting['inclination'], heading=setting['heading'], timeOvershoot = True, maxTime=burnOut_1 + 2)
    booster.postProcess()

    #Find starting conditions for booster recovery
    initial = [x for x in booster.solution if x[6] > 0][-1]
    initial = [0, *initial[1:]]
    boosterRecovery = Flight(rocket=Event4, environment=Env, initialSolution=initial, timeOvershoot = True)
    boosterRecovery.postProcess()

    #Find starting conditions for sustainer_coast
    initial = booster.solution[-1]
    initial = [0, *initial[1:]]
    sustainer_coast = Flight(rocket=Event2, environment=Env, initialSolution=initial, timeOvershoot = True, maxTime=setting["ignitionDelay"])
    sustainer_coast.postProcess()

    #Find starting conditions for sustainer
    initial = sustainer_coast.solution[-1]
    initial = [0, *initial[1:]]


    sustainer = Flight(rocket=Event3, environment=Env, initialSolution=initial, timeOvershoot = True, terminateOnApogee=True, maxTime=4000)
    sustainer.postProcess()

    #Find starting conditions for nose and sustainer recovery
    initial = [x for x in sustainer.solution if x[6] > 0][-1]
    initial = [0, *initial[1:]]
    sustainerRecovery = Flight(rocket=Event5, environment=Env, initialSolution=initial, timeOvershoot = True, maxTime=4000)
    sustainerRecovery.postProcess()
    noseRecovery = Flight(rocket=Event6, environment=Env, initialSolution=initial, timeOvershoot = True, maxTime=4000)
    noseRecovery.postProcess()

    df_settings = pd.DataFrame([setting])
    df_nose = pd.DataFrame([[noseRecovery.apogee, noseRecovery.apogeeX, noseRecovery.apogeeY, noseRecovery.xImpact, noseRecovery.yImpact, noseRecovery.impactVelocity]], columns = ['apogeeAltitude', 'apogeeX', 'apogeeY', 'impactX', 'impactY', 'impactVelocity'])
    df_sustainer = pd.DataFrame([[sustainerRecovery.apogee, sustainerRecovery.apogeeX, sustainerRecovery.apogeeY, sustainerRecovery.xImpact, sustainerRecovery.yImpact, sustainerRecovery.impactVelocity, sustainerRecovery.horizontalSpeed(0)]], columns = ['apogeeAltitude', 'apogeeX', 'apogeeY', 'impactX', 'impactY', 'impactVelocity', 'deploymentWind'])
    df_booster = pd.DataFrame([[boosterRecovery.apogee, boosterRecovery.apogeeX, boosterRecovery.apogeeY, boosterRecovery.xImpact, boosterRecovery.yImpact, boosterRecovery.impactVelocity, boosterRecovery.horizontalSpeed(0)]], columns = ['apogeeAltitude', 'apogeeX', 'apogeeY', 'impactX', 'impactY', 'impactVelocity', 'deploymentWind'])

    np.max(sustainer.horizontalSpeed[:])  

    if os.path.exists("df_nose.csv"):
        if os.stat("df_nose.csv").st_size <= 4:
            df_booster.to_csv("df_booster.csv", mode = 'a', index = False, header = True)
            df_sustainer.to_csv("df_sustainer.csv", mode = 'a', index = False, header = True)
            df_nose.to_csv("df_nose.csv", mode = 'a', index = False, header = True)
            df_settings.to_csv("df_settings.csv", mode = 'a', index = False, header = True)
        else:
            df_booster.to_csv("df_booster.csv", mode = 'a', index = False, header = False)
            df_sustainer.to_csv("df_sustainer.csv", mode = 'a', index = False, header = False)
            df_nose.to_csv("df_nose.csv", mode = 'a', index = False, header = False)
            df_settings.to_csv("df_settings.csv", mode = 'a', index = False, header = False)
    else:
        df_booster.to_csv("df_booster.csv", mode = 'a', index = False, header = True)
        df_sustainer.to_csv("df_sustainer.csv", mode = 'a', index = False, header = True)
        df_nose.to_csv("df_nose.csv", mode = 'a', index = False, header = True)
        df_settings.to_csv("df_settings.csv", mode = 'a', index = False, header = True)


    export_flight_data(noseRecovery, nose_output_file)
    export_flight_data(sustainerRecovery, sustainer_output_file)
    export_flight_data(boosterRecovery, booster_output_file)

    print("Simulations Completed: " + str(i))

# Closes export files
nose_output_file.close()
booster_output_file.close()
sustainer_output_file.close()


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
sustainer_results = extractData('sustainer_analysis_outputs')
booster_results = extractData('booster_analysis_outputs')

# Print number of flights simulated
print('Total number of simulations: ', i)
# Print data and standard deviations
print(f'Apogee Altitude -         Mean Value: {np.mean(nose_results["apogeeAltitude"]):0.3f} m')
print(f'Apogee Altitude - Standard Deviation: {np.std(nose_results["apogeeAltitude"]):0.3f} m')

print(f'Nose Impact X Position -         Mean Value: {np.mean(nose_results["impactX"]):0.3f} m')
print(f'Nose Impact X Position - Standard Deviation: {np.std(nose_results["impactX"]):0.3f} m')

print(f'Nose Impact Y Position -         Mean Value: {np.mean(nose_results["impactY"]):0.3f} m')
print(f'Nose Impact Y Position - Standard Deviation: {np.std(nose_results["impactY"]):0.3f} m')

print(f'Sustainer Impact X Position -         Mean Value: {np.mean(sustainer_results["impactX"]):0.3f} m')
print(f'Sustainer Impact X Position - Standard Deviation: {np.std(sustainer_results["impactX"]):0.3f} m')

print(f'Sustainer Impact Y Position -         Mean Value: {np.mean(sustainer_results["impactY"]):0.3f} m')
print(f'Sustainer Impact Y Position - Standard Deviation: {np.std(sustainer_results["impactY"]):0.3f} m')

print(f'Booster Impact X Position -         Mean Value: {np.mean(booster_results["impactX"]):0.3f} m')
print(f'Booster Impact X Position - Standard Deviation: {np.std(booster_results["impactX"]):0.3f} m')

print(f'Booster Impact Y Position -         Mean Value: {np.mean(booster_results["impactY"]):0.3f} m')
print(f'Booster Impact Y Position - Standard Deviation: {np.std(booster_results["impactY"]):0.3f} m')

end = time.time()
print("Elapsed time: ", end - start)

# Plot apogee data
plt.figure()
plt.hist(nose_results["apogeeAltitude"], bins=int(i**0.5))
plt.title('Apogee Altitude')
plt.xlabel('Altitude (m)')
plt.ylabel('Number of Occurences')
plt.show()

# Import background map
img = mpimg.imread("SampleMap.jpg")

# Gets the data to be graphed
ApogeeX = np.array(nose_results['apogeeX'])
ApogeeY = np.array(nose_results['apogeeY'])
noseImpactX = np.array(nose_results['impactX'])
noseImpactY = np.array(nose_results['impactY'])
sustainerImpactX = np.array(sustainer_results['impactX'])
sustainerImpactY = np.array(sustainer_results['impactY'])
boosterImpactX = np.array(booster_results['impactX'])
boosterImpactY = np.array(booster_results['impactY'])


# df_booster = pd.DataFrame.from_dict(booster_results) 
# df_sustainer = pd.DataFrame.from_dict(sustainer_results) 
# df_nose = pd.DataFrame.from_dict(nose_results) 

# df_booster.to_csv("df_booster.csv", index = False)
# df_sustainer.to_csv("df_sustainer.csv", index = False)
# df_nose.to_csv("df_nose.csv", index = False)
# df_settings.to_csv("df_settings.csv", index = False)

breakpoint()

# Function to calculate eigen values
def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]


# Create plot figure
plt.figure(num=None, figsize=(8, 8), dpi=150, facecolor='w', edgecolor='k')
ax = plt.subplot(111)

# Calculate error ellipses for impact
impactCov = np.cov(noseImpactX, noseImpactY)
impactVals, impactVecs = eigsorted(impactCov)
impactTheta = np.degrees(np.arctan2(*impactVecs[:,0][::-1]))
impactW, impactH = 2 * np.sqrt(impactVals)

# Calculate error ellipses for apogee
# apogeeCov = np.cov(apogeeX, apogeeY)
# apogeeVals, apogeeVecs = eigsorted(apogeeCov)
# apogeeTheta = np.degrees(np.arctan2(*apogeeVecs[:,0][::-1]))
# apogeeW, apogeeH = 2 * np.sqrt(apogeeVals)

# Draw error ellipses for apogee
# for j in [1, 2, 3]:
#    apogeeEll = Ellipse(xy=(np.mean(apogeeX), np.mean(apogeeY)),
#                  width=apogeeW*j, height=apogeeH*j,
#                  angle=apogeeTheta, color='black')
#    apogeeEll.set_facecolor((0, 1, 0, 0.2))
#    ax.add_artist(apogeeEll)

# Draw launch point
plt.scatter(0, 0, s=30, marker='*', color='black', label='Launch Point')
# Draw apogee points
plt.scatter(ApogeeX, ApogeeY, s=5, marker='^', color='orange', label='Simulated Apogee')
# Draw impact points
plt.scatter(noseImpactX, noseImpactY, s=5, marker='v', color='blue', label='Nose Cone Landing Point')
plt.scatter(sustainerImpactX, sustainerImpactY, s=5, marker='D', color='green', label='Sustainer Landing Point')
plt.scatter(boosterImpactX, boosterImpactY, s=5, marker='^', color='red', label='Booster Landing Point')
plt.legend()

# Add title and labels to plot
ax.set_title('1$\sigma$, 2$\sigma$ and 3$\sigma$ Dispersion Ellipses: Landing Points')
#ax.set_title('10%, 90% and 99% Confidence Dispersion Ellipses: Landing Points')
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
    impactEll = Ellipse(xy=(np.mean(noseImpactX), np.mean(noseImpactY)),
                        width=impactW*j, height=impactH*j,
                        angle=impactTheta, color='black')
    impactEll.set_facecolor((0, 0, 1, 0.2))
    impact_ellipses.append(impactEll)
    ax.add_artist(impactEll)


# Save plot and show result
filename = 'Dispersion Analysis Map'
plt.savefig(str(filename) + '.pdf', bbox_inches='tight', pad_inches=0)

plt.show()

print("Elapsed time: ", end - start)