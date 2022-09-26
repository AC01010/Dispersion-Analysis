
import pandas as pd
import matplotlib.image as mpimg
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np




df_settings = pd.read_csv("df_settings.csv")
df_nose = pd.read_csv("df_nose.csv")
df_sustainer = pd.read_csv("df_sustainer.csv")
df_booster = pd.read_csv("df_booster.csv")



img = plt.imread("launchSite.png")





def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]









fig=plt.figure(figsize=(9,6), dpi = 200)
x1 = df_nose["impactX"][df_settings["chuteFailure"]>4]
y1 = df_nose["impactY"][df_settings["chuteFailure"]>4]

x2 = df_sustainer["impactX"][df_settings["chuteFailure"]>4]
y2 = df_sustainer["impactY"][df_settings["chuteFailure"]>4]

x3 = df_booster["impactX"][df_settings["chuteFailure"]>4]
y3 = df_booster["impactY"][df_settings["chuteFailure"]>4]



x_ballistic = df_nose["impactX"][df_settings["chuteFailure"]<=4]
y_ballistic = df_nose["impactY"][df_settings["chuteFailure"]<=4]

breakpoint()

ax1=fig.add_subplot(221)
ax1.set_title('Nose Impact')
ax1.set_xlabel("East (m)", fontsize=10)
ax1.set_ylabel("North (m)", fontsize=10)
ax1.xaxis.set_major_locator(plt.MaxNLocator(2))
ax1.yaxis.set_major_locator(plt.MaxNLocator(2))
ax1.imshow(img, extent=[-50000, 50000, -50000, 50000])
ax1.scatter(x1, y1, alpha = 0.5, s =1, c = 'tab:orange')
# ax1.set_xlim([-50000, 50000])
# ax1.set_ylim([-50000, 50000])
# ax1.scatter(x_ballistic, y_ballistic, alpha = 0.4, color = 'r')

impactCov = np.cov(x1, y1)
impactVals, impactVecs = eigsorted(impactCov)
impactTheta = np.degrees(np.arctan2(*impactVecs[:,0][::-1]))
impactW, impactH = 2 * np.sqrt(impactVals)

impact_ellipses = []
for j in [1, 2, 3]:
    impactEll = Ellipse(xy=(np.mean(x1), np.mean(y1)),
                        width=impactW*j, height=impactH*j,
                        angle=impactTheta, color='black', fill = False)
    impactEll.set_facecolor((0, 0, 1, 0.2))
    impact_ellipses.append(impactEll)
    ax1.add_artist(impactEll)
ax1.scatter([0], [0], s = 5, c = 'r', marker ='x')



ax2=fig.add_subplot(222)
ax2.title.set_text('Sustainer Impact')
ax2.set_xlabel("East (m)", fontsize=10)
ax2.set_ylabel("North (m)", fontsize=10)
ax2.xaxis.set_major_locator(plt.MaxNLocator(2))
ax2.yaxis.set_major_locator(plt.MaxNLocator(2))
ax2.scatter(x2, y2, alpha = 0.5, s =1, c = 'tab:orange')
ax2.imshow(img, extent=[-50000, 50000, -50000, 50000])

impactCov = np.cov(x2, y2)
impactVals, impactVecs = eigsorted(impactCov)
impactTheta = np.degrees(np.arctan2(*impactVecs[:,0][::-1]))
impactW, impactH = 2 * np.sqrt(impactVals)

impact_ellipses = []
for j in [1, 2, 3]:
    impactEll = Ellipse(xy=(np.mean(x2), np.mean(y2)),
                        width=impactW*j, height=impactH*j,
                        angle=impactTheta, color='black', fill = False)
    impactEll.set_facecolor((0, 0, 1, 0.2))
    impact_ellipses.append(impactEll)
    ax2.add_artist(impactEll)
ax2.scatter([0], [0], s = 5, c = 'r', marker ='x')

ax3=fig.add_subplot(223)
ax3.set_xlabel("East (m)", fontsize=10)
ax3.set_ylabel("North (m)", fontsize=10)
ax3.title.set_text('Booster Impact')
ax3.xaxis.set_major_locator(plt.MaxNLocator(2))
ax3.yaxis.set_major_locator(plt.MaxNLocator(2))
ax3.scatter(x3, y3, alpha = 0.5, s =1, c = 'tab:orange')
ax3.imshow(img, extent=[-50000, 50000, -50000, 50000])

impactCov = np.cov(x3, y3)
impactVals, impactVecs = eigsorted(impactCov)
impactTheta = np.degrees(np.arctan2(*impactVecs[:,0][::-1]))
impactW, impactH = 2 * np.sqrt(impactVals)

impact_ellipses = []
for j in [1, 2, 3]:
    impactEll = Ellipse(xy=(np.mean(x3), np.mean(y3)),
                        width=impactW*j, height=impactH*j,
                        angle=impactTheta, color='black', fill = False)
    impactEll.set_facecolor((0, 0, 1, 0.2))
    impact_ellipses.append(impactEll)
    ax3.add_artist(impactEll)
ax3.scatter([0], [0],  s = 5, c = 'r', marker ='x')




ax4=fig.add_subplot(224)
ax4.set_xlabel("East (m)", fontsize=10)
ax4.set_ylabel("North (m)", fontsize=10)
ax4.title.set_text('Ballistic Nose Impact')
ax4.xaxis.set_major_locator(plt.MaxNLocator(2))
ax4.yaxis.set_major_locator(plt.MaxNLocator(2))
ax4.scatter(x_ballistic, y_ballistic, alpha = 0.5, s =1, c = 'tab:orange')
ax4.imshow(img, extent=[-50000, 50000, -50000, 50000])

impactCov = np.cov(x_ballistic, y_ballistic)
impactVals, impactVecs = eigsorted(impactCov)
impactTheta = np.degrees(np.arctan2(*impactVecs[:,0][::-1]))
impactW, impactH = 2 * np.sqrt(impactVals)

impact_ellipses = []
for j in [1, 2, 3]:
    impactEll = Ellipse(xy=(np.mean(x_ballistic), np.mean(y_ballistic)),
                        width=impactW*j, height=impactH*j,
                        angle=impactTheta, color='black', fill = False)
    impactEll.set_facecolor((0, 0, 1, 0.2))
    impact_ellipses.append(impactEll)
    ax4.add_artist(impactEll)
ax4.scatter([0], [0],  s = 5, c = 'r', marker ='x')


plt.tight_layout()
plt.show()




fig=plt.figure(figsize=(20,5), dpi = 200)
ax=fig.add_subplot(111)

ax.hist(df_nose["apogeeAltitude"], bins=100)
ax.set_xlabel("Apogee (m)", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)


plt.show()