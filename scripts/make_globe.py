"""Rotating celestial globe of DES pointings -> graphs/26_des_celestial_globe.gif"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.animation import FuncAnimation, PillowWriter

OUT = str(Path(__file__).resolve().parents[1] / 'figures')

# ---- style (dark) ----
BG = '#101014'
GRAT = '#33333d'
INK = '#c3c2b7'
CMAP = LinearSegmentedColormap.from_list('deep', ['#1c5cab', '#3987e5', '#86b6ef', '#cde2fb'])
GAL = '#e66767'
SN = '#eda100'

# ---- data: aggregate exposures into 0.5-deg pointing bins ----
df = pd.read_csv(Path(__file__).resolve().parents[1] / 'data' / 'des-exposures.csv.gz')
df = df[pd.to_datetime(df['datetime'], errors='coerce').dt.year >= 2012]
pts = (df.assign(ra_b=(df['ra'] * 2).round() / 2, dec_b=(df['dec'] * 2).round() / 2)
         .groupby(['ra_b', 'dec_b']).size().reset_index(name='cnt'))
print(len(pts), 'pointing bins')

def unit(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])

P = unit(pts['ra_b'].values, pts['dec_b'].values)          # 3 x N
C = pts['cnt'].values
sn = pts[pts['cnt'] > 800]                                  # supernova fields: revisited relentlessly
PSN = unit(sn['ra_b'].values, sn['dec_b'].values)

# galactic plane b=0 -> equatorial (J2000 rotation matrix, transposed for gal->eq)
R = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
              [ 0.4941094279, -0.4448296300,  0.7469822445],
              [-0.8676661490, -0.1980763734,  0.4559837762]])
l = np.radians(np.linspace(0, 360, 361))
GP = R.T @ np.vstack([np.cos(l), np.sin(l), np.zeros_like(l)])

# graticule: RA meridians every 30 deg, Dec parallels every 30 deg
merids, parls = [], []
t = np.linspace(-90, 90, 121)
for ra0 in range(0, 360, 30):
    merids.append(unit(np.full_like(t, ra0), t))
t2 = np.linspace(0, 360, 241)
for dec0 in range(-60, 61, 30):
    parls.append((dec0, unit(t2, np.full_like(t2, dec0))))

ra_labels = [(0, '0h'), (90, '6h'), (180, '12h'), (270, '18h')]
norm = LogNorm(vmin=1, vmax=C.max())

fig = plt.figure(figsize=(6.4, 6.4), facecolor=BG)
ax = fig.add_subplot(projection='3d', computed_zorder=False)

N_FRAMES = 132

def draw(frame):
    ax.clear()
    ax.set_facecolor(BG)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-0.62, 0.62); ax.set_ylim(-0.62, 0.62); ax.set_zlim(-0.62, 0.62)

    azim = frame * 360 / N_FRAMES
    elev = -28 + 7 * np.sin(2 * np.pi * frame / N_FRAMES)
    ax.view_init(elev=elev, azim=azim)
    e, a = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])

    def cull(V, margin=0.03):                      # NaN out the far hemisphere
        vis = (V.T @ view) > margin
        W = V.copy().astype(float)
        W[:, ~vis] = np.nan
        return W

    # graticule
    for M in merids:
        m = cull(M)
        ax.plot(m[0], m[1], m[2], color=GRAT, lw=0.6, zorder=1)
    for dec0, Pl in parls:
        p = cull(Pl)
        lw = 1.1 if dec0 == 0 else 0.6
        ax.plot(p[0], p[1], p[2], color=GRAT, lw=lw, zorder=1)

    # galactic plane
    g = cull(GP)
    ax.plot(g[0], g[1], g[2], color=GAL, lw=1.4, ls=(0, (4, 3)), zorder=2)

    # pointings
    vis = (P.T @ view) > 0.03
    order = np.argsort(C[vis])                     # draw dense fields last
    x, y, z = P[0, vis][order], P[1, vis][order], P[2, vis][order]
    ax.scatter(x, y, z, c=C[vis][order], cmap=CMAP, norm=norm,
               s=2.2, alpha=0.85, lw=0, zorder=3)

    # supernova fields
    vsn = (PSN.T @ view) > 0.03
    ax.scatter(PSN[0, vsn], PSN[1, vsn], PSN[2, vsn], color=SN, s=34,
               marker='*', lw=0, zorder=4)

    # RA hour labels on the equator, dec labels on the 0h meridian
    for ra0, lab in ra_labels:
        v = unit(ra0, 2)
        if v @ view > 0.15:
            ax.text(*(v * 1.10), lab, color=INK, fontsize=8, ha='center', zorder=5)
    for dec0 in (-60, -30, 30, 60):
        v = unit(0, dec0)
        if v @ view > 0.15:
            ax.text(*(v * 1.12), f'{dec0:+d}°', color=INK, fontsize=7, ha='center', zorder=5)

    fig.suptitle('The Dark Energy Survey on the celestial sphere',
                 color='#ffffff', fontsize=12, y=0.96, fontweight='bold')
    ax.text2D(0.5, 0.045, 'dot = pointing (brighter = more exposures, log)   '
              '★ supernova fields   ╌ galactic plane',
              transform=ax.transAxes, color=INK, fontsize=7.5, ha='center')

anim = FuncAnimation(fig, draw, frames=N_FRAMES, interval=50)
anim.save(f'{OUT}/26_des_celestial_globe.gif', writer=PillowWriter(fps=20),
          savefig_kwargs={'facecolor': BG}, dpi=105)
print('gif saved')
