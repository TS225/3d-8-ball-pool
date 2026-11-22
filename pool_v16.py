"""
8-Ball Pool  -  Panda3D + Bullet
--------------------------------
Controls
  Left-Right (or A/D)          aim      (hold SHIFT for fine aiming)
  Hold left mouse + move       aim
  Mouse wheel                  set shot power (before shooting)
  Middle mouse button (wheel click)  shoot with the selected power
  Hold SPACE                   alternative: charge power, release to shoot
  Right mouse + move           orbit the camera
  Up / Down arrows             camera zoom in / out (fine steps)
  C                            cycle camera: Overview / Player view / Top-down
  G                            graphics: High (shadows, per-pixel light) / Standard
  M                            toggle 1-player / 2-player
  R                            new rack          ESC  quit
Scoring
  +10 per ball, combo (+10 per extra ball in one shot), long pot (+15),
  break pot (+20), streak (+5 per consecutive scoring turn), 8-ball finish (+30)
  Scratch: -10 and your pots in that shot are void.
  Sinking the 8-ball before the table is clear loses the game.
"""
import sys
import os
import json
import gzip
import time
import math
import random
import struct
import tempfile
import wave

from panda3d.core import loadPrcFileData
loadPrcFileData("", "framebuffer-multisample 1\nmultisamples 4\n"
                    "window-title 8-Ball Pool\nwin-size 1280 800\nsync-video 1")

import panda3d.bullet as bp
from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import DirectWaitBar, DirectFrame, DirectEntry, DirectButton
from panda3d.core import (
    Vec3, Vec4, Point3, TextNode, WindowProperties,
    AmbientLight, DirectionalLight, PointLight, LineSegs, ClockObject,
    KeyboardButton, GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    Geom, GeomTriangles, GeomNode, Material, PNMImage, Texture,
    TransparencyAttrib, AntialiasAttrib, CallbackObject, Filename, TextureStage,
    Fog, ColorBlendAttrib, CardMaker, TexGenAttrib, Quat,
)

HW, HL = 10.0, 20.0          # half width / half length of the playing field
R = 0.5                      # ball radius
BALL_MASS = 0.16
MAX_SPEED = 64.35         # 45 -> 49.5 (+10 %) -> 64.35 (+30 % more)
CUE_START = Point3(0, -10, R)
RACK_APEX = (0, 10)


# ============================================================================
# Procedural geometry
# ============================================================================
def _norm(v):
    l = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) or 1.0
    return (v[0] / l, v[1] / l, v[2] / l)


class Mesh:
    """Tiny mesh builder: vertex + normal + color + uv."""
    def __init__(self):
        self.v = []
        self.t = []

    def vert(self, p, n, c=(1, 1, 1, 1), uv=(0, 0)):
        self.v.append((p, n, c, uv))
        return len(self.v) - 1

    def tri(self, a, b, c):
        self.t.append((a, b, c))

    def node(self, name="mesh"):
        vdata = GeomVertexData(name, GeomVertexFormat.getV3n3c4t2(), Geom.UHStatic)
        vdata.setNumRows(len(self.v))
        vw = GeomVertexWriter(vdata, "vertex")
        nw = GeomVertexWriter(vdata, "normal")
        cw = GeomVertexWriter(vdata, "color")
        tw = GeomVertexWriter(vdata, "texcoord")
        for p, n, c, uv in self.v:
            vw.addData3(p[0], p[1], p[2])
            nw.addData3(n[0], n[1], n[2])
            cw.addData4(c[0], c[1], c[2], c[3])
            tw.addData2(uv[0], uv[1])
        prim = GeomTriangles(Geom.UHStatic)
        for a, b, c in self.t:
            prim.addVertices(a, b, c)
        geom = Geom(vdata)
        geom.addPrimitive(prim)
        gn = GeomNode(name)
        gn.addGeom(geom)
        return gn


def sphere_node(radius=1.0, rings=28, segs=44, color=(1, 1, 1, 1), colorfn=None):
    m = Mesh()
    for i in range(rings + 1):
        phi = math.pi * i / rings
        cz = math.cos(phi)
        for j in range(segs + 1):
            th = 2 * math.pi * j / segs
            n = (math.sin(phi) * math.cos(th), math.sin(phi) * math.sin(th), cz)
            col = colorfn(cz) if colorfn else color
            m.vert((n[0] * radius, n[1] * radius, n[2] * radius), n, col,
                   (j / segs, 1 - i / rings))
    for i in range(rings):
        for j in range(segs):
            a = i * (segs + 1) + j
            b = a + segs + 1
            m.tri(a, b, a + 1)
            m.tri(a + 1, b, b + 1)
    return m.node("sphere")


def box_node(color=(1, 1, 1, 1)):
    """Box with half-extents 1 (scale it)."""
    m = Mesh()
    faces = [
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
        ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
        ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
    ]
    for n, u, v in faces:
        ids = []
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            p = tuple(n[k] + su * u[k] + sv * v[k] for k in range(3))
            ids.append(m.vert(p, n, color, ((su + 1) / 2, (sv + 1) / 2)))
        m.tri(ids[0], ids[1], ids[2])
        m.tri(ids[0], ids[2], ids[3])
    return m.node("box")


def extrusion_node(profile, length, color):
    """Extrude a convex CCW (a, z) profile along local Y (centered)."""
    m = Mesh()
    h = length / 2.0
    n_pts = len(profile)
    for i in range(n_pts):
        p0, p1 = profile[i], profile[(i + 1) % n_pts]
        da, db = p1[0] - p0[0], p1[1] - p0[1]
        n = _norm((db, 0, -da))
        a = m.vert((p0[0], -h, p0[1]), n, color)
        b = m.vert((p1[0], -h, p1[1]), n, color)
        c = m.vert((p1[0], h, p1[1]), n, color)
        d = m.vert((p0[0], h, p0[1]), n, color)
        m.tri(a, c, b)
        m.tri(a, d, c)
    ids = [m.vert((p[0], h, p[1]), (0, 1, 0), color) for p in profile]
    for k in range(1, n_pts - 1):
        m.tri(ids[0], ids[k + 1], ids[k])
    ids = [m.vert((p[0], -h, p[1]), (0, -1, 0), color) for p in profile]
    for k in range(1, n_pts - 1):
        m.tri(ids[0], ids[k], ids[k + 1])
    return m.node("extrusion")


def lathe_node(profile, segs=24):
    """Surface of revolution around local Y. profile = [(y, radius, color)], y increasing."""
    m = Mesh()
    for y, r, col in profile:
        for j in range(segs + 1):
            th = 2 * math.pi * j / segs
            m.vert((r * math.cos(th), y, r * math.sin(th)),
                   (math.cos(th), 0, math.sin(th)), col, (j / segs, 0))
    for i in range(len(profile) - 1):
        for j in range(segs):
            a = i * (segs + 1) + j
            b = a + segs + 1
            m.tri(a, b, a + 1)
            m.tri(b, b + 1, a + 1)
    return m.node("lathe")


def torus_node(major, minor, color, segs=44, sides=14):
    m = Mesh()
    for i in range(segs + 1):
        a = 2 * math.pi * i / segs
        for j in range(sides + 1):
            b = 2 * math.pi * j / sides
            n = (math.cos(b) * math.cos(a), math.cos(b) * math.sin(a), math.sin(b))
            p = (major * math.cos(a) + minor * n[0],
                 major * math.sin(a) + minor * n[1], minor * n[2])
            m.vert(p, n, color)
    for i in range(segs):
        for j in range(sides):
            a0 = i * (sides + 1) + j
            a1 = a0 + sides + 1
            m.tri(a0, a1, a0 + 1)
            m.tri(a1, a1 + 1, a0 + 1)
    return m.node("torus")


def disc_node(radius, color, segs=36):
    m = Mesh()
    c = m.vert((0, 0, 0), (0, 0, 1), color, (0.5, 0.5))
    ring = []
    for i in range(segs):
        th = 2 * math.pi * i / segs
        ring.append(m.vert((radius * math.cos(th), radius * math.sin(th), 0),
                           (0, 0, 1), color, (0.5, 0.5)))
    for i in range(segs):
        m.tri(c, ring[i], ring[(i + 1) % segs])
    return m.node("disc")


def felt_node(color):
    """Subdivided felt with baked lamp-pool shading (brighter under the lamps)."""
    nx, ny = 8, 16
    m = Mesh()
    for j in range(ny + 1):
        y = -HL + 2 * HL * j / ny
        for i in range(nx + 1):
            x = -HW + 2 * HW * i / nx
            lit = 0.70 + 0.40 * max(math.exp(-((x / 14.0) ** 2 + ((y - 10) / 13.0) ** 2)),
                                     math.exp(-((x / 14.0) ** 2 + ((y + 10) / 13.0) ** 2)))
            col = (color[0] * lit, color[1] * lit, color[2] * lit, 1)
            m.vert((x, y, 0), (0, 0, 1), col, (x / 6.0, y / 6.0))
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b = a + 1
            c = b + nx + 1
            d = a + nx + 1
            m.tri(a, b, c)
            m.tri(a, c, d)
    return m.node("felt")


def solid_node(faces, color, uv_scale=0.1, inward=False, center=None):
    """Convex solid from planar faces (each a list of 3D points, any winding).
    inward=True makes normals point toward `center` (used for room shells)."""
    m = Mesh()
    allp = [p for f in faces for p in f]
    cen = center if center is not None else tuple(
        sum(p[k] for p in allp) / len(allp) for k in range(3))
    for f in faces:
        a, b, c = f[0], f[1], f[2]
        e1 = tuple(b[k] - a[k] for k in range(3))
        e2 = tuple(c[k] - a[k] for k in range(3))
        n = _norm((e1[1] * e2[2] - e1[2] * e2[1],
                   e1[2] * e2[0] - e1[0] * e2[2],
                   e1[0] * e2[1] - e1[1] * e2[0]))
        fc = tuple(sum(p[k] for p in f) / len(f) for k in range(3))
        d = sum(n[k] * (fc[k] - cen[k]) for k in range(3))
        if (d < 0) != inward:
            f = list(reversed(f))
            n = (-n[0], -n[1], -n[2])
        ax = max(range(3), key=lambda k: abs(n[k]))
        ids = []
        for p in f:
            if ax == 0:
                uv = (p[1] * uv_scale, p[2] * uv_scale)
            elif ax == 1:
                uv = (p[0] * uv_scale, p[2] * uv_scale)
            else:
                uv = (p[0] * uv_scale, p[1] * uv_scale)
            ids.append(m.vert(p, n, color, uv))
        for i in range(1, len(ids) - 1):
            m.tri(ids[0], ids[i], ids[i + 1])
    return m.node("solid")


def cushion_points(h, taper):
    """Cushion in local space: a = outward from the playing edge, y along the rail.
    The nose (a=0) is full length; the back (a=1) is shorter, so the end faces are
    angled (chamfered) jaws; here taper < 0 makes the back longer, so the mouth
    narrows as it goes deeper into the pocket."""
    hb = h - taper
    return [
        (0, -h, 0), (0, h, 0), (0, -h, 0.62), (0, h, 0.62),
        (1, -hb, 0), (1, hb, 0), (1, -hb, 0.95), (1, hb, 0.95),
    ]


def cushion_node(h, taper, color):
    nb0, nb1, nt0, nt1, bb0, bb1, bt0, bt1 = cushion_points(h, taper)
    faces = [
        (nb0, nb1, nt1, nt0), (nt0, nt1, bt1, bt0), (bb0, bb1, bt1, bt0),
        (nb0, nb1, bb1, bb0), (nb0, nt0, bt0, bb0), (nb1, nt1, bt1, bb1),
    ]
    return solid_node(faces, color)


# ============================================================================
# Procedural textures
# ============================================================================
def make_felt_texture():
    size = 256
    img = PNMImage(size, size)
    for x in range(size):
        for y in range(size):
            fibre = 0.5 + 0.5 * math.sin(y * 0.9 + 3.0 * math.sin(x * 0.11) + random.random() * 0.8)
            v = 0.86 + 0.08 * random.random() + 0.06 * fibre
            if random.random() < 0.01:
                v *= 1.12                      # stray lighter fibres
            img.setXel(x, y, v, v, v)
    tex = Texture("felt")
    tex.load(img)
    tex.setMinfilter(Texture.FTLinearMipmapLinear)
    tex.setAnisotropicDegree(4)
    return tex


def make_wood_texture():
    size = 256
    img = PNMImage(size, size)
    for x in range(size):
        for y in range(size):
            g = math.sin(y * 0.09 + 5.0 * math.sin(x * 0.02) + 2.0 * math.sin(x * 0.05 + y * 0.01))
            g += 0.5 * math.sin(y * 0.4 + 3.0 * math.sin(x * 0.03))
            v = 0.80 + 0.12 * g / 1.5 + 0.04 * random.random()
            v = max(0.45, min(1.0, v))
            img.setXel(x, y, v, v, v)
    tex = Texture("wood")
    tex.load(img)
    tex.setMinfilter(Texture.FTLinearMipmapLinear)
    return tex


def shiny(np, spec=0.9, shininess=70):
    mat = Material()
    mat.setSpecular(Vec4(spec, spec, spec, 1))
    mat.setShininess(shininess)
    np.setMaterial(mat, 1)


# ============================================================================
# Ball
# ============================================================================
# ---------------------------------------------------------------------------
# Environment reflections (sphere map), glow sprites
# ---------------------------------------------------------------------------
ENV_NODES = []
ENV_STATE = {"on": True}
_env_cache = {}


def make_env_texture(scale):
    size = 96
    img = PNMImage(size, size)
    blobs = ((-0.35, -0.45, 0.05, 1.0), (0.42, -0.30, 0.03, 0.7), (-0.10, 0.55, 0.08, 0.35))
    for x in range(size):
        for y in range(size):
            u = (x / (size - 1)) * 2 - 1
            v = (y / (size - 1)) * 2 - 1
            r = math.hypot(u, v)
            if r > 1.0:
                val = 0.0
            else:
                val = 0.03 + 0.30 * r ** 4
                for cx, cy, s, k in blobs:
                    val += k * math.exp(-((u - cx) ** 2 + (v - cy) ** 2) / s)
            val = min(1.0, val * scale)
            img.setXel(x, y, val, val, val * 0.95)
    tex = Texture("env")
    tex.load(img)
    tex.setWrapU(Texture.WMClamp)
    tex.setWrapV(Texture.WMClamp)
    return tex


def apply_env(np, strength=0.5):
    """Fake environment reflection (window/lamp glints) via an eye-space sphere map."""
    key = round(strength, 2)
    if key not in _env_cache:
        _env_cache[key] = make_env_texture(key)
    tex = _env_cache[key]
    ts = TextureStage("env")
    ts.setMode(TextureStage.MAdd)
    ENV_NODES.append((np, ts, tex))
    if ENV_STATE["on"]:
        np.setTexGen(ts, TexGenAttrib.MEyeSphereMap)
        np.setTexture(ts, tex)


def set_env_enabled(flag):
    ENV_STATE["on"] = flag
    ENV_NODES[:] = [e for e in ENV_NODES if not e[0].isEmpty()]
    for np, ts, tex in ENV_NODES:
        if flag:
            np.setTexGen(ts, TexGenAttrib.MEyeSphereMap)
            np.setTexture(ts, tex)
        else:
            np.clearTexGen(ts)
            np.clearTexture(ts)


def make_glow_texture():
    size = 64
    img = PNMImage(size, size, 4)
    for x in range(size):
        for y in range(size):
            r = math.hypot(x / (size - 1) * 2 - 1, y / (size - 1) * 2 - 1)
            a = max(0.0, 1.0 - r) ** 2.2
            img.setXelA(x, y, 1.0, 1.0, 1.0, a)
    tex = Texture("glow")
    tex.load(img)
    tex.setWrapU(Texture.WMClamp)
    tex.setWrapV(Texture.WMClamp)
    return tex


def make_glow(parent, tex, pos, size, color):
    """Additive billboard halo, a cheap stand-in for bloom around light sources."""
    cm = CardMaker("glow")
    cm.setFrame(-1, 1, -1, 1)
    np = parent.attachNewNode(cm.generate())
    np.setTexture(tex)
    np.setPos(*pos)
    np.setScale(size)
    np.setColor(*color)
    np.setBillboardPointEye()
    np.setLightOff()
    np.setDepthWrite(False)
    np.setAttrib(ColorBlendAttrib.make(ColorBlendAttrib.MAdd,
                                       ColorBlendAttrib.OIncomingAlpha, ColorBlendAttrib.OOne))
    np.setBin("transparent", 20)
    return np


BALL_COLORS = {
    1: (1.0, 0.82, 0.05), 2: (0.05, 0.2, 0.85), 3: (0.9, 0.1, 0.08),
    4: (0.4, 0.1, 0.6), 5: (1.0, 0.45, 0.05), 6: (0.05, 0.5, 0.15),
    7: (0.5, 0.08, 0.1), 8: (0.03, 0.03, 0.03),
}


class Ball:
    _sticker = None
    _dot = None

    def __init__(self, game, number, position):
        self.game = game
        self.number = number
        self.alive = True
        self.in_world = True

        self.node = bp.BulletRigidBodyNode(f"ball_{number}")
        self.node.setMass(BALL_MASS)
        self.node.setFriction(0.05)
        self.node.setRestitution(0.9)
        self.node.setLinearDamping(0.5)      # rolling resistance
        self.node.setAngularDamping(0.0)     # spin is set from velocity (pure rolling)
        self.node.setLinearSleepThreshold(0.4)
        self.node.setLinearFactor(Vec3(1, 1, 0))          # stay on the table
        self.node.setCcdMotionThreshold(1e-7)             # no tunnelling
        self.node.setCcdSweptSphereRadius(R * 0.8)
        self.node.addShape(bp.BulletSphereShape(R))

        self.np = game.render.attachNewNode(self.node)
        self.np.setPos(position)
        self.np.setHpr(random.uniform(0, 360), random.uniform(-90, 90), random.uniform(0, 360))
        game.world.attachRigidBody(self.node)

        # ---- visuals
        if number == 0:
            geom = sphere_node(R, color=(0.97, 0.97, 0.94, 1))
        else:
            base = BALL_COLORS[(number - 1) % 8 + 1] if number != 8 else BALL_COLORS[8]
            base = (base[0], base[1], base[2], 1)
            white = (0.96, 0.96, 0.92, 1)
            if number > 8:   # stripe: coloured band around the equator
                geom = sphere_node(R, colorfn=lambda cz: base if abs(cz) < 0.5 else white)
            else:
                geom = sphere_node(R, color=base)
        model = self.np.attachNewNode(geom)
        shiny(model, 0.95, 90)
        apply_env(model, 0.6)

        if Ball._sticker is None:
            Ball._sticker = sphere_node(1.0, 8, 18, (0.97, 0.97, 0.94, 1))
            Ball._dot = sphere_node(1.0, 8, 12, (0.85, 0.05, 0.05, 1))
        for sgn, h in ((1, 90), (-1, -90)):
            if number == 0:
                d = self.np.attachNewNode(Ball._dot)
                d.setScale(0.09)
                d.setPos(sgn * 0.49, 0, 0)
                continue
            s = self.np.attachNewNode(Ball._sticker)
            s.setScale(0.03, 0.21, 0.21)
            s.setPos(sgn * 0.478, 0, 0)
            tn = TextNode(f"num{number}")
            tn.setText(str(number))
            tn.setAlign(TextNode.ACenter)
            tn.setTextColor(0.02, 0.02, 0.02, 1)
            t = self.np.attachNewNode(tn)
            t.setScale(0.26 if number < 10 else 0.2)
            t.setPos(sgn * 0.5, 0, -0.07 if number < 10 else -0.055)
            t.setH(h)
            t.setLightOff()
            t.setTwoSided(True)
            t.setDepthOffset(1)

        # soft blob shadow
        self.shadow = game.render.attachNewNode(disc_node(R * 1.05, (0, 0, 0, 0.38)))
        self.shadow.setTransparency(TransparencyAttrib.MAlpha)
        self.shadow.setLightOff()
        self.shadow.setDepthWrite(False)

    def pos(self):
        return self.np.getPos()

    def speed(self):
        return self.node.getLinearVelocity().length()

    def stop(self):
        self.node.setLinearVelocity(Vec3(0, 0, 0))
        self.node.setAngularVelocity(Vec3(0, 0, 0))

    def leave_world(self):
        if self.in_world:
            self.game.world.removeRigidBody(self.node)
            self.in_world = False
        self.stop()
        self.alive = False

    def respawn(self, pos):
        self.np.setScale(1)
        self.np.setPos(pos)
        self.np.show()
        if not self.in_world:
            self.game.world.attachRigidBody(self.node)
            self.in_world = True
        self.stop()
        self.node.setActive(True)
        self.alive = True

    def destroy(self):
        if self.in_world:
            self.game.world.removeRigidBody(self.node)
            self.in_world = False
        self.np.removeNode()
        self.shadow.removeNode()


# ============================================================================
# Table
# ============================================================================
class Table:
    def __init__(self, game):
        self.game = game
        self.render = game.render
        self.wood_tex = make_wood_texture()
        self.felt_tex = make_felt_texture()

        wood = (0.36, 0.17, 0.06, 1)
        dark_wood = (0.20, 0.09, 0.03, 1)
        green = (0.06, 0.48, 0.17, 1)
        cushion_green = (0.04, 0.34, 0.11, 1)

        # ---------- room / legs / apron / dark base
        leg_profile = [(0, 1.5, dark_wood), (0.5, 1.7, dark_wood), (0.9, 1.2, dark_wood),
                       (1.8, 1.1, dark_wood), (3.6, 1.3, dark_wood), (5.2, 1.75, dark_wood),
                       (6.2, 1.8, dark_wood), (7.2, 1.5, dark_wood), (8.4, 1.15, dark_wood),
                       (9.8, 1.15, dark_wood), (10.6, 1.55, dark_wood), (11.0, 1.55, dark_wood)]
        for sx in (1, -1):
            for sy in (1, -1):                      # four turned legs at the outer corners
                leg = self.render.attachNewNode(lathe_node(leg_profile, 24))
                leg.setPos(sx * 13.2, sy * 23.2, -12.0)
                leg.setP(90)
                leg.setTwoSided(True)
                leg.setTexture(self.wood_tex)
                shiny(leg, 0.4, 30)
        self.vis_box(-14.3, 14.3, -24.3, 24.3, -1.7, -0.1, dark_wood, self.wood_tex)
        self.vis_box(-14.0, 14.0, -24.0, 24.0, -0.4, -0.02, (0.01, 0.01, 0.01, 1))

        # ---------- physical floor (top at z = 0)
        self.phys_box((0, 0, -0.5), (14.2, 24.2, 0.5), friction=0.6, restitution=0.3, name="floor")

        # ---------- felt
        felt = self.render.attachNewNode(felt_node(green))
        felt.setTexture(self.felt_tex)
        rnd = random.Random(7)
        for _ in range(18):                      # chalk dust / wear marks
            r = rnd.uniform(0.5, 1.6)
            sm = self.render.attachNewNode(disc_node(r, (0.60, 0.65, 0.75, 0.07), 16))
            sm.setPos(rnd.uniform(-8, 8), rnd.uniform(-16, 6), 0.011)
            sm.setTransparency(TransparencyAttrib.MAlpha)
            sm.setLightOff()
            sm.setDepthWrite(False)

        # ---------- field markings
        for pos in ((0, -10), RACK_APEX):
            spot = self.render.attachNewNode(disc_node(0.16, (0.9, 0.9, 0.9, 1), 16))
            spot.setPos(pos[0], pos[1], 0.012)
            spot.setLightOff()
        ls = LineSegs()
        ls.setThickness(1.5)
        ls.setColor(0.8, 0.85, 0.8, 0.45)
        ls.moveTo(-HW, -10, 0.01)
        ls.drawTo(HW, -10, 0.01)
        line = self.render.attachNewNode(ls.create())
        line.setTransparency(TransparencyAttrib.MAlpha)
        line.setLightOff()

        # ---------- cushions: trapezoid wedges with angled (chamfered) pocket jaws
        for sx in (1, -1):
            for y0, y1 in [(1.3, 18.3), (-18.3, -1.3)]:
                self.cushion((y1 - y0) / 2, -1.0, (sx * HW, (y0 + y1) / 2),
                             0 if sx == 1 else 180, cushion_green)
        for sy in (1, -1):
            self.cushion(8.3, -1.0, (0, sy * HL), 90 if sy == 1 else -90, cushion_green)

        # ---------- wooden rails with pocket openings
        top = 1.1
        mirrored = [
            (11, 14, 1.5, 19.5),      # long rail
            (12.5, 14, 19.5, 24),     # outer corner
            (9.5, 12.5, 22.5, 24),    # corner back
        ]
        for sx in (1, -1):
            for sy in (1, -1):
                for x0, x1, y0, y1 in mirrored:
                    xs = sorted((sx * x0, sx * x1))
                    ys = sorted((sy * y0, sy * y1))
                    self.rail(xs[0], xs[1], ys[0], ys[1], top, wood)
        for sy in (1, -1):
            ys = sorted((sy * 21, sy * 24))
            self.rail(-9.5, 9.5, ys[0], ys[1], top, wood)           # end rails
        for sx in (1, -1):
            xs = sorted((sx * 13.5, sx * 14))
            self.rail(xs[0], xs[1], -1.5, 1.5, top, wood)           # side-pocket backs

        # ---------- diamonds (pearl inlays)
        pearl = (0.95, 0.93, 0.84, 1)
        for sx in (1, -1):
            for y in (-15, -10, -5, 5, 10, 15):
                self.inlay(sx * 12.5, y, top + 0.01, pearl)
        for sy in (1, -1):
            for x in (-5, 0, 5):
                self.inlay(x, sy * 22.5, top + 0.01, pearl)

        # ---------- pockets
        # Openings sit at table-bed level (no raised rings): a dark leather
        # outline disc with a black hole disc on top.
        specs = [(sx * 11.0, sy * 21.0, 1.5) for sx in (1, -1) for sy in (1, -1)]
        specs += [(sx * 12.0, 0.0, 1.4) for sx in (1, -1)]
        self.pockets = []
        for x, y, r in specs:
            rim = self.render.attachNewNode(disc_node(r + 0.2, (0.20, 0.10, 0.05, 1)))
            rim.setPos(x, y, 0.012)
            rim.setLightOff()
            hole = self.render.attachNewNode(disc_node(r, (0, 0, 0, 1)))
            hole.setPos(x, y, 0.016)
            hole.setLightOff()
            self.pockets.append(Vec3(x, y, 0))

        # 45-degree chamfer fills in the corners of each rail opening, so the
        # opening reads as a rounded octagon with slightly flared walls.
        for sx in (1, -1):
            for sy in (1, -1):
                for cx, cy, ux, vy in [(12.5, 19.5, -1, 1), (9.5, 22.5, 1, -1),
                                       (12.5, 22.5, -1, -1)]:
                    self.chamfer_fill(sx * cx, sy * cy, sx * ux, sy * vy, top, wood)
            for cx, cy, ux, vy in [(13.5, 1.5, -1, -1), (13.5, -1.5, -1, 1),
                                   (11.0, 1.5, 1, -1), (11.0, -1.5, 1, 1)]:
                self.chamfer_fill(sx * cx, cy, sx * ux, vy, top, wood)

    # ----- helpers
    def vis_box(self, x0, x1, y0, y1, z0, z1, color, tex=None):
        np = self.render.attachNewNode(box_node(color))
        np.setPos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
        np.setScale((x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2)
        if tex:
            np.setTexture(tex)
        return np

    def phys_box(self, center, half, friction=0.4, restitution=0.5, name="rail"):
        node = bp.BulletRigidBodyNode(name)
        node.setMass(0)
        node.setFriction(friction)
        node.setRestitution(restitution)
        node.addShape(bp.BulletBoxShape(Vec3(*half)))
        np = self.render.attachNewNode(node)
        np.setPos(*center)
        self.game.world.attachRigidBody(node)
        return np

    def rail(self, x0, x1, y0, y1, top, color):
        z0 = -0.2
        v = self.vis_box(x0, x1, y0, y1, z0, top, color, self.wood_tex)
        shiny(v, 0.35, 30)
        self.phys_box(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + top) / 2),
                      ((x1 - x0) / 2, (y1 - y0) / 2, (top - z0) / 2), 0.1, 0.6)

    def cushion(self, h, taper, pos, heading, color):
        node = bp.BulletRigidBodyNode("cushion")
        node.setMass(0)
        node.setFriction(0.05)
        node.setRestitution(0.85)
        shape = bp.BulletConvexHullShape()
        shape.setMargin(0.01)
        for p in cushion_points(h, taper):
            shape.addPoint(Point3(*p))
        node.addShape(shape)
        np = self.render.attachNewNode(node)
        np.setPos(pos[0], pos[1], 0)
        np.setH(heading)
        self.game.world.attachRigidBody(node)
        np.attachNewNode(cushion_node(h, taper, color))

    def inlay(self, x, y, z, color):
        d = self.render.attachNewNode(disc_node(0.22, color, 14))
        d.setPos(x, y, z)
        shiny(d, 1.0, 60)

    def chamfer_fill(self, cx, cy, ux, vy, top, color, L=0.75, shift=-0.3):
        """Triangular 45-degree corner piece; its top edge overhangs inward so the
        pocket walls funnel toward the hole."""
        z0, z1 = -0.2, top
        A0, B0, C0 = (cx, cy, z0), (cx + ux * L, cy, z0), (cx, cy + vy * L, z0)
        A1 = (cx, cy, z1)
        B1 = (cx + ux * (L - shift), cy, z1)
        C1 = (cx, cy + vy * (L - shift), z1)
        faces = [(A0, B0, C0), (A1, B1, C1), (A0, B0, B1, A1),
                 (A0, C0, C1, A1), (B0, C0, C1, B1)]
        np = self.render.attachNewNode(solid_node(faces, color))
        np.setTexture(self.wood_tex)
        shiny(np, 0.35, 30)

    def nearest_pocket(self, p):
        return min(self.pockets, key=lambda q: (q.x - p.x) ** 2 + (q.y - p.y) ** 2)


# ============================================================================
# Cue stick
# ============================================================================
class CueStick:
    def __init__(self, game):
        dark = (0.10, 0.05, 0.03, 1)
        gold = (0.85, 0.7, 0.3, 1)
        maple = (0.86, 0.68, 0.40, 1)
        ferrule = (0.92, 0.92, 0.95, 1)
        tip = (0.15, 0.35, 0.7, 1)
        profile = [
            (-14.0, 0.32, dark), (-10.0, 0.27, dark), (-10.0, 0.26, gold),
            (-9.6, 0.26, gold), (-9.6, 0.23, maple), (-0.6, 0.12, maple),
            (-0.6, 0.115, ferrule), (-0.15, 0.11, ferrule), (-0.15, 0.10, tip),
            (0.0, 0.10, tip),
        ]
        self.np = game.render.attachNewNode(lathe_node(profile, 20))
        self.np.setTwoSided(True)
        shiny(self.np, 0.8, 60)
        self.np.hide()

    def place(self, tip_pos, heading):
        self.np.setPos(tip_pos)
        self.np.setHpr(heading, -5, 0)
        self.np.show()


# ============================================================================
# Game
# ============================================================================
# ============================================================================
# Pub interior
# ============================================================================
def make_brick_texture():
    size, bw, bh = 128, 32, 16
    img = PNMImage(size, size)
    shades = {}
    for y in range(size):
        row = y // bh
        off = bw // 2 if row % 2 else 0
        for x in range(size):
            xx = (x + off) % size
            if (y % bh) < 2 or (xx % bw) < 2:
                v = 0.60 + 0.06 * random.random()
                img.setXel(x, y, v, v * 0.97, v * 0.92)
            else:
                key = (row, xx // bw)
                if key not in shades:
                    shades[key] = 0.65 + 0.35 * random.random()
                s = shades[key] * (0.92 + 0.08 * random.random())
                img.setXel(x, y, 0.62 * s + 0.10, 0.27 * s + 0.05, 0.18 * s + 0.03)
    tex = Texture("brick")
    tex.load(img)
    tex.setMinfilter(Texture.FTLinearMipmapLinear)
    return tex


def make_plank_texture():
    size, bwid, blen = 128, 16, 64
    img = PNMImage(size, size)
    base = [0.65 + 0.35 * random.random() for _ in range(size // bwid)]
    offs = [random.randrange(blen) for _ in range(size // bwid)]
    for x in range(size):
        b = x // bwid
        for y in range(size):
            if x % bwid < 1 or (y + offs[b]) % blen < 1:
                img.setXel(x, y, 0.08, 0.05, 0.03)
                continue
            grain = 0.9 + 0.1 * math.sin(y * 0.25 + b * 3.1 + 2.0 * math.sin(x * 0.4)) \
                + 0.05 * random.random()
            s = base[b] * grain
            img.setXel(x, y, 0.50 * s, 0.30 * s, 0.15 * s)
    tex = Texture("planks")
    tex.load(img)
    tex.setMinfilter(Texture.FTLinearMipmapLinear)
    return tex


def make_paper_texture():
    size = 128
    img = PNMImage(size, size)
    for x in range(size):
        for y in range(size):
            sx = x % 32
            n = 0.95 + 0.05 * random.random()
            if sx < 2:
                c = (0.62, 0.50, 0.22)                       # gold pinstripe
            else:
                c = (0.13, 0.27, 0.18)                       # deep green
                if abs(sx - 16) + abs((y % 32) - 16) < 6:    # little damask diamond
                    c = (0.19, 0.36, 0.25)
            img.setXel(x, y, c[0] * n, c[1] * n, c[2] * n)
    tex = Texture("paper")
    tex.load(img)
    tex.setMinfilter(Texture.FTLinearMipmapLinear)
    return tex


def make_plaster_texture():
    size = 64
    img = PNMImage(size, size)
    for x in range(size):
        for y in range(size):
            v = 0.88 + 0.12 * random.random()
            img.setXel(x, y, v, v, v)
    tex = Texture("plaster")
    tex.load(img)
    return tex


def arc_band_node(radius, z0, z1, a0, a1, color, segs=14):
    """Curved band (part of a cylinder wall) - used for round chair backs."""
    m = Mesh()
    for i in range(segs + 1):
        a = math.radians(a0 + (a1 - a0) * i / segs)
        n = (math.cos(a), math.sin(a), 0)
        m.vert((radius * n[0], radius * n[1], z0), n, color, (i / segs, 0))
        m.vert((radius * n[0], radius * n[1], z1), n, color, (i / segs, 1))
    for i in range(segs):
        a, b, c, d = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        m.tri(a, c, b)
        m.tri(b, c, d)
    return m.node("arcband")


class Pub:
    """Room, bar, fireplace, furniture and decoration around the pool table."""
    FZ, CZ = -12.0, 27.0
    X0, X1, Y0, Y1 = -55.0, 55.0, -62.0, 62.0

    def __init__(self, game, wood_tex):
        self.game = game
        self.render = game.render
        self.wood = wood_tex
        self.root = self.render.attachNewNode("pub")
        self.ceiling = self.root.attachNewNode("ceiling")
        self.cache = {}
        self.time = 0.0
        self.flames = []
        self.fire_light = None
        self.t_brick = make_brick_texture()
        self.t_plank = make_plank_texture()
        self.t_paper = make_paper_texture()
        self.t_plaster = make_plaster_texture()
        self.t_glow = make_glow_texture()
        self.fire_glows = []
        self.motes = []

        self.build_room()
        self.build_fireplace()
        self.build_bar()
        self.build_seating()
        self.build_decor()
        self.build_props()
        self.build_ceiling()
        self.build_glows()
        self.build_lights()

    # ------------------------------------------------------------- helpers
    def cached(self, key, fn):
        if key not in self.cache:
            self.cache[key] = fn()
        return self.cache[key]

    def box(self, x0, x1, y0, y1, z0, z1, color, tex=None, parent=None,
            light=True, spec=None, tile=None):
        node = self.cached(("box", color), lambda: box_node(color))
        np = (parent or self.root).attachNewNode(node)
        np.setPos((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
        np.setScale(max(abs(x1 - x0), 1e-3) / 2, max(abs(y1 - y0), 1e-3) / 2,
                    max(abs(z1 - z0), 1e-3) / 2)
        if tex:
            np.setTexture(tex)
        if tile:
            np.setTexScale(TextureStage.getDefault(), tile[0], tile[1])
        if not light:
            np.setLightOff()
        if spec:
            shiny(np, *spec)
            if spec[0] >= 0.85:
                apply_env(np, 0.5)
        return np

    def vlathe(self, profile, x, y, z, parent=None, segs=20, spec=None, light=True):
        key = ("lathe", tuple(profile), segs)
        np = (parent or self.root).attachNewNode(self.cached(key, lambda: lathe_node(profile, segs)))
        np.setPos(x, y, z)
        np.setP(90)                      # profile "y" becomes height
        np.setTwoSided(True)
        if spec:
            shiny(np, *spec)
            if spec[0] >= 0.85:
                apply_env(np, 0.5)
        if not light:
            np.setLightOff()
        return np

    def cap(self, x, y, z, r, color, parent=None, spec=None, light=True):
        node = self.cached(("disc", r, color), lambda: disc_node(r, color, 24))
        np = (parent or self.root).attachNewNode(node)
        np.setPos(x, y, z)
        if spec:
            shiny(np, *spec)
            if spec[0] >= 0.85:
                apply_env(np, 0.5)
        if not light:
            np.setLightOff()
        return np

    def text3d(self, text, pos, heading, scale, color, light=False, parent=None):
        tn = TextNode("pubtext")
        tn.setText(text)
        tn.setAlign(TextNode.ACenter)
        tn.setTextColor(*color)
        np = (parent or self.root).attachNewNode(tn)
        np.setPos(*pos)
        np.setH(heading)
        np.setScale(scale)
        np.setTwoSided(True)
        if not light:
            np.setLightOff()
        return np

    def wall_rect(self, wall, a0, a1, z0, z1, d0, d1, color, **kw):
        if wall == "L":
            return self.box(self.X0 + d0, self.X0 + d1, a0, a1, z0, z1, color, **kw)
        if wall == "R":
            return self.box(self.X1 - d1, self.X1 - d0, a0, a1, z0, z1, color, **kw)
        if wall == "F":
            return self.box(a0, a1, self.Y0 + d0, self.Y0 + d1, z0, z1, color, **kw)
        return self.box(a0, a1, self.Y1 - d1, self.Y1 - d0, z0, z1, color, **kw)

    def wall_text(self, wall, a, z, d, text, scale, color, light=False):
        if wall == "L":
            return self.text3d(text, (self.X0 + d, a, z), 90, scale, color, light)
        if wall == "R":
            return self.text3d(text, (self.X1 - d, a, z), -90, scale, color, light)
        if wall == "F":
            return self.text3d(text, (a, self.Y0 + d, z), 180, scale, color, light)
        return self.text3d(text, (a, self.Y1 - d, z), 0, scale, color, light)

    def quad(self, pts, tex, color, uv, parent=None):
        cen = ((self.X0 + self.X1) / 2, (self.Y0 + self.Y1) / 2, (self.FZ + self.CZ) / 2)
        np = (parent or self.root).attachNewNode(
            solid_node([pts], color, uv, inward=True, center=cen))
        if tex:
            np.setTexture(tex)
        return np

    # ---------------------------------------------------------------- room
    def build_room(self):
        X0, X1, Y0, Y1, FZ, CZ = self.X0, self.X1, self.Y0, self.Y1, self.FZ, self.CZ
        floor = self.quad([(X0, Y0, FZ), (X1, Y0, FZ), (X1, Y1, FZ), (X0, Y1, FZ)],
                          self.t_plank, (1, 1, 1, 1), 0.1)
        shiny(floor, 0.3, 25)
        wain = 2.0
        walls = {
            "L": lambda a, b: [(X0, Y0, a), (X0, Y1, a), (X0, Y1, b), (X0, Y0, b)],
            "R": lambda a, b: [(X1, Y0, a), (X1, Y1, a), (X1, Y1, b), (X1, Y0, b)],
            "F": lambda a, b: [(X0, Y0, a), (X1, Y0, a), (X1, Y0, b), (X0, Y0, b)],
            "B": lambda a, b: [(X0, Y1, a), (X1, Y1, a), (X1, Y1, b), (X0, Y1, b)],
        }
        for k, f in walls.items():
            if k == "B":
                self.quad(f(FZ, CZ), self.t_brick, (1, 1, 1, 1), 0.07)
            else:
                self.quad(f(FZ, wain), self.wood, (0.30, 0.14, 0.06, 1), 0.06)
                self.quad(f(wain, CZ), self.t_paper, (1, 1, 1, 1), 0.06)
        trim = (0.22, 0.10, 0.04, 1)
        for wall, (a0, a1) in {"L": (Y0, Y1), "R": (Y0, Y1), "F": (X0, X1), "B": (X0, X1)}.items():
            self.wall_rect(wall, a0, a1, wain - 0.3, wain + 0.5, 0, 0.35, trim, tex=self.wood, spec=(0.3, 20))
            self.wall_rect(wall, a0, a1, FZ, FZ + 1.3, 0, 0.35, trim, tex=self.wood, spec=(0.3, 20))
        # rug under the pool table
        rug = self.box(-22, 22, -32, 32, FZ + 0.01, FZ + 0.07, (0.40, 0.07, 0.09, 1), self.t_plaster,
                       tile=(6, 9))
        for x0, x1, y0, y1 in [(-22, 22, -32, -31), (-22, 22, 31, 32), (-22, -21, -32, 32), (21, 22, -32, 32)]:
            self.box(x0, x1, y0, y1, FZ + 0.07, FZ + 0.10, (0.75, 0.6, 0.25, 1))
        # blue cue chalk on the table rail
        self.box(12.6, 13.4, -9.4, -8.6, 1.1, 1.7, (0.15, 0.35, 0.85, 1))

    # ----------------------------------------------------------- fireplace
    def build_fireplace(self):
        Y1, FZ, CZ = self.Y1, self.FZ, self.CZ
        stone = (0.5, 0.48, 0.44, 1)
        self.box(-14, 14, Y1 - 8, Y1, FZ, CZ - 1.5, (1, 1, 1, 1), self.t_brick, tile=(4, 6))
        self.box(-8.6, 8.6, Y1 - 12.5, Y1 - 8, FZ, FZ + 0.6, stone, spec=(0.3, 20))          # hearth
        for x0, x1, z0, z1 in [(-7.5, -5.5, FZ, FZ + 11), (5.5, 7.5, FZ, FZ + 11),
                               (-7.5, 7.5, FZ + 9.5, FZ + 11)]:
            self.box(x0, x1, Y1 - 8.5, Y1 - 7.9, z0, z1, stone)                              # surround
        self.box(-5.5, 5.5, Y1 - 8.2, Y1 - 4.5, FZ + 0.3, FZ + 9.5, (0.02, 0.02, 0.02, 1), light=False)
        self.box(-14.5, 14.5, Y1 - 9.8, Y1 - 7.6, FZ + 12.5, FZ + 13.5, (0.25, 0.11, 0.04, 1),
                 self.wood, spec=(0.4, 30))                                                   # mantel
        bark, ember = (0.22, 0.12, 0.06, 1), (1.0, 0.3, 0.05, 1)
        for i, (y, z) in enumerate([(Y1 - 6.8, FZ + 1.2), (Y1 - 6.0, FZ + 1.2), (Y1 - 6.4, FZ + 2.3)]):
            lg = self.cached(("lathe", "log"), lambda: lathe_node(
                [(0, 0.45, ember), (0.3, 0.9, bark), (7.7, 0.9, bark), (8, 0.45, ember)], 12))
            np = self.root.attachNewNode(lg)
            np.setPos(-4 + 0.5 * i, y, z)
            np.setH(-90)
        self.box(-4.8, 4.8, Y1 - 7.6, Y1 - 5.0, FZ + 0.3, FZ + 0.7, (0.9, 0.15, 0.02, 1), light=False)
        flame = [(0.0, 0.9, 0.45, 1.0), (0.6, 0.6, 0.5, 0.9), (1.0, 0.9, 0.8, 0.5), (0.9, 0.2, 0.8, 0.5)]
        for i, (xo, yo, zo, sc) in enumerate([(-2.5, 6.4, 3.0, 0.9), (0.2, 6.6, 3.0, 1.3),
                                              (2.6, 6.3, 3.0, 0.9), (-1.0, 6.9, 3.2, 0.7)]):
            f = self.vlathe([(0, 0.9, (1, 0.45, 0.05, 1)), (1.6, 0.6, (1, 0.65, 0.1, 1)),
                             (3.2, 0.0, (1, 0.92, 0.45, 1))], xo, Y1 - yo, FZ + zo, light=False)
            f.setScale(sc, sc, sc)
            self.flames.append((f, sc, i))
        # mantel decorations
        self.vlathe([(0, 0.5, (0.8, 0.65, 0.3, 1)), (1.2, 0.2, (0.8, 0.65, 0.3, 1)),
                     (3.0, 0.3, (0.8, 0.65, 0.3, 1))], -10, Y1 - 8.7, FZ + 13.5, spec=(1, 90))
        self.vlathe([(0, 0.5, (0.8, 0.65, 0.3, 1)), (1.2, 0.2, (0.8, 0.65, 0.3, 1)),
                     (3.0, 0.3, (0.8, 0.65, 0.3, 1))], 10, Y1 - 8.7, FZ + 13.5, spec=(1, 90))
        self.vlathe([(3.0, 0.12, (1, 0.7, 0.2, 1)), (3.5, 0.0, (1, 0.9, 0.5, 1))], -10, Y1 - 8.7,
                    FZ + 13.5, light=False)
        self.vlathe([(3.0, 0.12, (1, 0.7, 0.2, 1)), (3.5, 0.0, (1, 0.9, 0.5, 1))], 10, Y1 - 8.7,
                    FZ + 13.5, light=False)
        self.box(-3, 3, Y1 - 9.3, Y1 - 8.2, FZ + 13.5, FZ + 17.5, (0.22, 0.1, 0.04, 1), self.wood)   # clock case
        self.cap(0, Y1 - 9.4, FZ + 15.5, 0.01, (1, 1, 1, 1))
        face = self.root.attachNewNode(self.cached(("sph", "clock"), lambda: sphere_node(1.0, 10, 20, (0.95, 0.93, 0.85, 1))))
        face.setPos(0, Y1 - 9.35, FZ + 15.5)
        face.setScale(2.0, 0.08, 2.0)
        self.box(-0.1, 0.1, Y1 - 9.5, Y1 - 9.4, FZ + 15.5, FZ + 17.0, (0.05, 0.05, 0.05, 1))
        self.box(-0.1, 1.3, Y1 - 9.5, Y1 - 9.4, FZ + 15.4, FZ + 15.6, (0.05, 0.05, 0.05, 1))
        self.wall_text("B", 0, CZ - 8.0, 8.3, "THE 8-BALL INN", 1.9, (1.0, 0.78, 0.25, 1))

    # ----------------------------------------------------------------- bar
    def build_bar(self):
        X0, FZ = self.X0, self.FZ
        dark = (0.20, 0.09, 0.03, 1)
        brass = (0.82, 0.66, 0.28, 1)
        chrome = (0.78, 0.80, 0.84, 1)
        # counter body, panels and top
        self.box(-49, -43, -30, 30, FZ, 3.4, dark, self.wood, spec=(0.3, 25))
        for k in range(-5, 6):
            y = k * 5.4
            self.box(-43.05, -42.8, y - 2.0, y + 2.0, FZ + 2.0, 2.4, (0.30, 0.14, 0.05, 1), self.wood)
        self.box(-49.8, -42.0, -30.8, 30.8, 3.4, 4.2, (0.16, 0.07, 0.03, 1), self.wood, spec=(0.9, 80))
        self.box(-43.0, -41.8, -30.8, 30.8, 3.0, 3.5, brass, spec=(1, 90))
        # brass foot rail
        rail = self.root.attachNewNode(self.cached(("lathe", "footrail"),
                                                   lambda: lathe_node([(0, 0.3, brass), (60, 0.3, brass)], 12)))
        rail.setPos(-41.4, -30, FZ + 2.6)
        shiny(rail, 1, 100)
        for y in (-30, -15, 0, 15, 30):
            self.vlathe([(0, 0.25, brass), (2.6, 0.25, brass)], -41.4, y, FZ, spec=(1, 90))
        # back bar: cabinet, mirror, shelves, bottles
        self.wall_rect("L", -28.5, 28.5, FZ, -4.0, 0, 5.5, dark, tex=self.wood, spec=(0.3, 25))
        self.wall_rect("L", -28.5, 28.5, -4.3, -3.8, 0, 6.0, (0.16, 0.07, 0.03, 1), tex=self.wood)
        self.wall_rect("L", -27, 27, 5, 20, 0.0, 0.25, (0.62, 0.72, 0.78, 1), spec=(1, 120))
        for a0, a1, z0, z1 in [(-27.8, 27.8, 20, 21), (-27.8, 27.8, 4.2, 5), (-27.8, -27, 4.2, 21), (27, 27.8, 4.2, 21)]:
            self.wall_rect("L", a0, a1, z0, z1, 0, 0.6, brass, spec=(1, 90))
        palette = [(0.45, 0.22, 0.05, 1), (0.05, 0.30, 0.10, 1), (0.72, 0.82, 0.86, 1),
                   (0.08, 0.05, 0.04, 1), (0.12, 0.2, 0.55, 1), (0.55, 0.1, 0.08, 1)]
        for shelf_z in (6.0, 10.5, 15.0):
            self.wall_rect("L", -27.5, 27.5, shelf_z - 0.4, shelf_z, 0, 3.4, (0.3, 0.14, 0.05, 1),
                           tex=self.wood, spec=(0.4, 30))
            self.wall_rect("L", -27, 27, shelf_z - 0.55, shelf_z - 0.4, 3.0, 3.3, (1.0, 0.85, 0.55, 1), light=False)
            y = -26.0
            while y < 26.0:
                col = random.choice(palette)
                tall = random.choice([4.2, 4.4, 3.6])
                prof = [(0, 0.55, col), (tall * 0.55, 0.55, col), (tall * 0.68, 0.42, col),
                        (tall * 0.82, 0.2, col), (tall, 0.18, col)]
                b = self.vlathe(prof, X0 + 1.6 + random.uniform(-0.3, 0.3), y, shelf_z, segs=12, spec=(0.9, 80))
                self.cap(X0 + 1.6, y, shelf_z + tall, 0.2, (0.8, 0.65, 0.25, 1))
                y += 2.1 + random.uniform(0, 0.4)
        # beer pumps on the counter
        for y, hc in [(-9, (0.7, 0.1, 0.1, 1)), (0, (0.1, 0.4, 0.2, 1)), (9, (0.9, 0.7, 0.1, 1))]:
            self.vlathe([(0, 0.9, dark), (0.5, 0.9, dark)], -46.5, y, 4.2)
            self.vlathe([(0.5, 0.45, chrome), (4.2, 0.4, chrome)], -46.5, y, 4.2, spec=(1, 100))
            self.box(-47.0, -46.0, y - 0.5, y + 0.5, 8.6, 10.6, hc, spec=(0.7, 60))
            self.box(-46.3, -45.3, y - 0.12, y + 0.12, 7.6, 8.4, (0.1, 0.1, 0.1, 1))
        for x, y in [(-44.5, -15), (-44.0, 4.5), (-45.0, 16), (-44.2, -3)]:
            self.pint(x, y, 4.2)
        # clock above the mirror
        clk = self.root.attachNewNode(self.cached(("sph", "bclock"), lambda: sphere_node(1.0, 10, 20, (0.95, 0.93, 0.85, 1))))
        clk.setPos(X0 + 0.5, 0, 23.5)
        clk.setScale(0.1, 2.2, 2.2)
        # stools
        for y in (-24, -16, -8, 0, 8, 16, 24):
            self.stool(-37.5, y)

    def pint(self, x, y, z, s=1.0):
        glass = (0.85, 0.92, 0.95, 0.28)
        self.cap(x, y, z + 0.02, 1.45 * s, (0.82, 0.70, 0.45, 1))      # beer mat
        z += 0.05
        g = self.vlathe([(0, 0.8 * s, glass), (2.4 * s, 1.0 * s, glass)], x, y, z, segs=14, spec=(1, 100))
        g.setTransparency(TransparencyAttrib.MAlpha)
        self.vlathe([(0.1 * s, 0.72 * s, (0.85, 0.52, 0.08, 1)), (1.9 * s, 0.9 * s, (0.85, 0.52, 0.08, 1))],
                    x, y, z, segs=14)
        self.cap(x, y, z + 2.05 * s, 0.9 * s, (0.96, 0.94, 0.86, 1))

    def stool(self, x, y):
        FZ = self.FZ
        chrome = (0.78, 0.80, 0.84, 1)
        leather = (0.45, 0.06, 0.08, 1)
        self.vlathe([(0, 3.0, chrome), (0.35, 3.0, chrome)], x, y, FZ, spec=(1, 90))
        self.cap(x, y, FZ + 0.35, 3.0, chrome, spec=(1, 90))
        self.vlathe([(0.35, 0.35, chrome), (9.6, 0.35, chrome)], x, y, FZ, spec=(1, 100))
        self.vlathe([(9.2, 3.0, (0.12, 0.05, 0.05, 1)), (10.9, 3.25, leather)], x, y, FZ, spec=(0.5, 40))
        self.cap(x, y, FZ + 10.9, 3.25, leather, spec=(0.6, 50))
        ring = self.root.attachNewNode(self.cached("foot", lambda: torus_node(2.3, 0.18, chrome, 24, 8)))
        ring.setPos(x, y, FZ + 3.8)
        shiny(ring, 1, 90)

    # ------------------------------------------------------------- seating
    def pub_chair(self, x, y, heading):
        """Round pub chair: padded round seat, four turned legs, curved wooden back."""
        wood = (0.30, 0.13, 0.05, 1)
        leather = (0.42, 0.05, 0.07, 1)
        root = self.root.attachNewNode("pubchair")
        root.setPos(x, y, self.FZ)
        root.setH(heading)                       # chair faces local +Y
        self.vlathe([(6.3, 2.75, wood), (6.9, 2.75, wood)], 0, 0, 0, parent=root, segs=24, spec=(0.3, 20))
        self.vlathe([(6.9, 2.55, leather), (7.5, 2.45, leather)], 0, 0, 0, parent=root, segs=24, spec=(0.5, 40))
        self.cap(0, 0, 7.5, 2.45, leather, parent=root, spec=(0.5, 40))
        for k in range(4):                       # legs
            ang = math.radians(45 + 90 * k)
            self.vlathe([(0, 0.2, wood), (3.0, 0.24, wood), (3.3, 0.3, wood), (6.3, 0.34, wood)],
                        2.0 * math.cos(ang), 2.0 * math.sin(ang), 0, parent=root, segs=10, spec=(0.3, 20))
        ring = root.attachNewNode(self.cached("chairring", lambda: torus_node(2.0, 0.12, wood, 20, 6)))
        ring.setPos(0, 0, 2.4)
        for ang in (215, 325):                   # back posts
            a_ = math.radians(ang)
            self.vlathe([(6.9, 0.26, wood), (11.8, 0.26, wood)], 2.5 * math.cos(a_), 2.5 * math.sin(a_), 0,
                        parent=root, segs=8, spec=(0.3, 20))
        for z0, z1 in ((8.3, 9.9), (10.3, 11.8)):  # curved back bands
            band = root.attachNewNode(self.cached(("arc", z0), lambda z0=z0, z1=z1: arc_band_node(
                2.55, z0, z1, 205, 335, wood)))
            band.setTwoSided(True)
            shiny(band, 0.3, 20)

    def build_ring_chairs(self):
        """Round pub chairs standing around the pool table, with generous room for cue play."""
        pts = []
        for y in (-27, -13.5, 0, 13.5, 27):
            pts += [(30, y), (-30, y)]
        for x in (-12, 0, 12):
            pts += [(x, 40), (x, -40)]
        for x, y in pts:
            self.pub_chair(x, y, math.degrees(math.atan2(x, -y)))

    def build_seating(self):
        FZ, Y0 = self.FZ, self.Y0
        # leather banquette along the front wall with small tables
        green = (0.10, 0.28, 0.18, 1)
        self.box(-38, -4, Y0, Y0 + 6.5, FZ, FZ + 6.0, (0.2, 0.09, 0.04, 1), self.wood)
        self.box(-38, -4, Y0 + 0.3, Y0 + 6.2, FZ + 6.0, FZ + 7.0, green, spec=(0.6, 50))
        self.box(-38, -4, Y0, Y0 + 2.0, FZ + 7.0, FZ + 17.0, green, spec=(0.6, 50))
        for xi in range(-36, -4, 3):
            for zi in (FZ + 10.0, FZ + 13.0, FZ + 15.5):
                b = self.root.attachNewNode(self.cached("btn", lambda: sphere_node(0.22, 6, 10, (0.8, 0.65, 0.3, 1))))
                b.setPos(xi, Y0 + 2.1, zi)
        self.build_ring_chairs()

    # --------------------------------------------------------------- decor
    def picture(self, wall, a, z, w, h, kind):
        self.wall_rect(wall, a - w / 2, a + w / 2, z - h / 2, z + h / 2, 0.0, 0.5,
                       (0.55, 0.40, 0.12, 1), spec=(0.8, 40))
        i = 0.5
        x0, x1, z0, z1 = a - w / 2 + i, a + w / 2 - i, z - h / 2 + i, z + h / 2 - i
        if kind == 0:     # landscape
            mid = z0 + (z1 - z0) * 0.38
            self.wall_rect(wall, x0, x1, mid, z1, 0.5, 0.6, (0.45, 0.65, 0.9, 1))
            self.wall_rect(wall, x0, x1, z0, mid, 0.5, 0.6, (0.2, 0.45, 0.2, 1))
            self.wall_rect(wall, x1 - 1.6, x1 - 0.6, z1 - 1.6, z1 - 0.6, 0.6, 0.65, (1, 0.85, 0.4, 1), light=False)
        elif kind == 1:   # portrait
            self.wall_rect(wall, x0, x1, z0, z1, 0.5, 0.6, (0.12, 0.08, 0.06, 1))
            cx = a
            self.wall_rect(wall, cx - 1.2, cx + 1.2, z - 0.4, z + 2.2, 0.6, 0.7, (0.8, 0.62, 0.5, 1))
            self.wall_rect(wall, cx - 2.4, cx + 2.4, z0, z - 0.5, 0.6, 0.7, (0.35, 0.05, 0.08, 1))
        else:             # abstract stripes
            n = 4
            cols = [(0.7, 0.1, 0.1, 1), (0.9, 0.85, 0.7, 1), (0.1, 0.2, 0.5, 1), (0.9, 0.7, 0.1, 1)]
            for k in range(n):
                self.wall_rect(wall, x0 + (x1 - x0) * k / n, x0 + (x1 - x0) * (k + 1) / n, z0, z1,
                               0.5, 0.6, cols[k])

    def sconce(self, wall, a, z):
        brass = (0.82, 0.66, 0.28, 1)
        self.wall_rect(wall, a - 0.8, a + 0.8, z - 1.2, z + 1.2, 0, 0.4, brass, spec=(1, 80))
        self.wall_rect(wall, a - 0.5, a + 0.5, z - 0.4, z + 1.4, 0.4, 1.4, (1.0, 0.88, 0.55, 1), light=False)
        self.glow(self.wall_point(wall, a, z + 0.5, 1.8), 5.5, (1.0, 0.8, 0.45, 0.5))

    def dartboard(self, cy, cz):
        X1 = self.X1
        x = X1 - 0.3
        m = Mesh()

        def pt(r, a):
            al = math.radians(a)
            return (x, cy - r * math.sin(al), cz + r * math.cos(al))

        def ring(r0, r1, a0, a1, color):
            steps = max(2, int(abs(a1 - a0) / 6))
            for k in range(steps):
                aa = a0 + (a1 - a0) * k / steps
                ab = a0 + (a1 - a0) * (k + 1) / steps
                ids = [m.vert(p, (-1, 0, 0), color) for p in (pt(r0, aa), pt(r1, aa), pt(r1, ab), pt(r0, ab))]
                m.tri(ids[0], ids[1], ids[2])
                m.tri(ids[0], ids[2], ids[3])
        black, cream = (0.06, 0.06, 0.06, 1), (0.88, 0.82, 0.62, 1)
        red, green = (0.75, 0.08, 0.08, 1), (0.08, 0.45, 0.2, 1)
        for i in range(20):
            a0, a1 = i * 18 - 9, i * 18 + 9
            dark = (i % 2 == 0)
            ring(1.4, 2.6, a0, a1, black if dark else cream)
            ring(3.0, 4.3, a0, a1, black if dark else cream)
            ring(2.6, 3.0, a0, a1, red if dark else green)
            ring(4.3, 4.7, a0, a1, red if dark else green)
            ring(4.7, 6.2, a0, a1, black)
        ring(0.6, 1.4, 0, 360, green)
        ring(0.0, 0.6, 0, 360, red)
        self.wall_rect("R", cy - 7.2, cy + 7.2, cz - 7.2, cz + 7.2, 0.0, 0.2, (0.25, 0.10, 0.05, 1), tex=self.wood)
        board = self.root.attachNewNode(m.node("dartboard"))
        board.setTwoSided(True)
        ls = LineSegs()
        ls.setThickness(1.2)
        ls.setColor(0.75, 0.75, 0.78, 1)
        for i in range(20):
            a = i * 18 + 9
            p0, p1 = pt(1.4, a), pt(4.7, a)
            ls.moveTo(p0[0] - 0.02, p0[1], p0[2])
            ls.drawTo(p1[0] - 0.02, p1[1], p1[2])
        for rr in (0.6, 1.4, 2.6, 3.0, 4.3, 4.7):
            for k in range(49):
                p = pt(rr, k * 7.5)
                (ls.moveTo if k == 0 else ls.drawTo)(p[0] - 0.02, p[1], p[2])
        wires = self.root.attachNewNode(ls.create())
        wires.setLightOff()
        order = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]
        for i, n in enumerate(order):
            p = pt(5.45, i * 18)
            self.text3d(str(n), (p[0] - 0.03, p[1], p[2] - 0.28), -90, 0.62, (0.95, 0.92, 0.8, 1))
        # darts stuck in the board
        red_f, steel = (0.8, 0.1, 0.1, 1), (0.8, 0.8, 0.85, 1)
        for dy, dz in [(0.4, 0.6), (-1.6, 3.1), (1.9, -2.4)]:
            d = self.root.attachNewNode(self.cached(("lathe", "dart"), lambda: lathe_node(
                [(0, 0.14, red_f), (1.2, 0.14, red_f), (1.2, 0.06, steel), (3.0, 0.03, steel)], 8)))
            d.setPos(x - 3.0, cy + dy, cz + dz)
            d.setH(-90)

    def build_decor(self):
        X0, X1, Y0, Y1, FZ, CZ = self.X0, self.X1, self.Y0, self.Y1, self.FZ, self.CZ
        # right wall: darts, cue rack, sign, pictures, sconces
        self.dartboard(8, 5)
        self.wall_rect("R", -33.5, -26.5, -4, 10, 0.0, 0.8, (0.25, 0.11, 0.04, 1), tex=self.wood, spec=(0.3, 25))
        cue_profile = [(-14.0, 0.32, (0.10, 0.05, 0.03, 1)), (-10.0, 0.27, (0.10, 0.05, 0.03, 1)),
                       (-9.6, 0.23, (0.86, 0.68, 0.40, 1)), (-0.6, 0.12, (0.86, 0.68, 0.40, 1)),
                       (-0.6, 0.115, (0.92, 0.92, 0.95, 1)), (0.0, 0.10, (0.15, 0.35, 0.7, 1))]
        for k, a in enumerate((-32.4, -30.8, -29.2, -27.6)):
            c = self.root.attachNewNode(self.cached(("lathe", "wallcue"), lambda: lathe_node(cue_profile, 12)))
            c.setPos(X1 - 1.4, a, FZ + 20.0 - 4 + 0.0)
            c.setP(90)
            c.setScale(1.0, 1.15, 1.0)
        self.wall_rect("R", -34.5, -25.5, -4.6, -3.8, 0.0, 1.6, (0.25, 0.11, 0.04, 1), tex=self.wood)
        self.wall_rect("R", -34.5, -25.5, 8.4, 9.2, 0.0, 1.6, (0.25, 0.11, 0.04, 1), tex=self.wood)
        for a0, a1 in ((-34.5, -33.6), (-26.4, -25.5)):
            self.wall_rect("R", a0, a1, -4.6, 9.2, 0.0, 1.6, (0.25, 0.11, 0.04, 1), tex=self.wood)
        self.wall_rect("R", 24, 38, 14.5, 20.5, 0.0, 0.3, (0.06, 0.03, 0.05, 1))
        for a0, a1, z0, z1 in [(24, 38, 14.3, 14.7), (24, 38, 20.3, 20.7), (23.8, 24.2, 14.3, 20.7), (37.8, 38.2, 14.3, 20.7)]:
            self.wall_rect("R", a0, a1, z0, z1, 0.3, 0.5, (1.0, 0.25, 0.6, 1), light=False)
        self.wall_text("R", 31, 16.6, 0.45, "ALE & POOL", 2.2, (1.0, 0.35, 0.7, 1))
        self.picture("R", 46, 8, 7, 9, 0)
        self.picture("R", -10, 16, 8, 6, 2)
        self.picture("R", -46, 10, 7, 9, 1)
        for a in (-45, -10, 30, 50):
            self.sconce("R", a, 9)
        # left wall extras
        self.picture("L", -45, 12, 6, 8, 1)
        self.picture("L", 45, 12, 7, 6, 0)
        self.sconce("L", 38, 9)
        self.sconce("L", -38, 9)
        # back wall: chalkboard menu, pictures, sconces
        self.wall_rect("B", 17, 31, -3, 11, 0.0, 0.5, (0.3, 0.14, 0.05, 1), tex=self.wood, spec=(0.3, 25))
        self.wall_rect("B", 17.6, 30.4, -2.4, 10.4, 0.5, 0.6, (0.04, 0.07, 0.06, 1))
        lines = [("TODAY'S MENU", 9.0, 1.1), ("Fish & Chips ... 12", 6.0, 0.75),
                 ("Steak Pie ....... 14", 4.0, 0.75), ("Pint of Ale .... 5", 2.0, 0.75),
                 ("Pool: 1 coin", -0.2, 0.75)]
        for t, zz, sc in lines:
            self.wall_text("B", 24, zz, 0.65, t, sc, (0.92, 0.92, 0.85, 1))
        self.picture("B", -22, 6, 9, 7, 0)
        self.picture("B", -34, 10, 6, 8, 1)
        self.picture("B", -43, 5, 6, 6, 2)
        for a in (-18, 18, 40, -48):
            self.sconce("B", a, 14)
        # front wall: window with curtains, door, pictures
        a, w = 15.0, 12.0
        frame = (0.85, 0.80, 0.70, 1)
        self.wall_rect("F", a - w / 2, a + w / 2, 2.0, 18, 0.0, 0.3, (0.03, 0.05, 0.12, 1), light=False)
        for k in range(8):
            self.wall_rect("F", a - 5 + k * 1.3, a - 5 + k * 1.3 + 0.18, 3 + (k * 7) % 13, 3.18 + (k * 7) % 13,
                           0.3, 0.35, (1, 1, 0.9, 1), light=False)
        self.wall_rect("F", a - w / 2 - 0.5, a + w / 2 + 0.5, 1.5, 2.2, 0.0, 1.2, frame, spec=(0.4, 30))
        for a0, a1, z0, z1 in [(a - w / 2 - 0.5, a - w / 2, 2, 18.5), (a + w / 2, a + w / 2 + 0.5, 2, 18.5),
                               (a - w / 2, a + w / 2, 18, 18.6), (a - 0.2, a + 0.2, 2, 18), (a - w / 2, a + w / 2, 9.8, 10.2)]:
            self.wall_rect("F", a0, a1, z0, z1, 0.0, 0.6, frame, spec=(0.4, 30))
        for side in (-1, 1):
            for k in range(7):
                x0 = a + side * (w / 2 + 0.7) + side * k * 0.75
                shade = (0.50, 0.06, 0.08, 1) if k % 2 == 0 else (0.36, 0.03, 0.06, 1)
                self.wall_rect("F", min(x0, x0 + side * 0.75), max(x0, x0 + side * 0.75), -1.0, 19.5,
                               0.2, 1.4 + 0.3 * (k % 2), shade)
        rod = self.root.attachNewNode(self.cached(("lathe", "rod"), lambda: lathe_node(
            [(0, 0.2, (0.82, 0.66, 0.28, 1)), (26, 0.2, (0.82, 0.66, 0.28, 1))], 8)))
        rod.setPos(a - 13, self.Y0 + 1.0, 20)
        rod.setH(-90)
        shiny(rod, 1, 90)
        # door
        da = 41.0
        self.wall_rect("F", da - 5.2, da + 5.2, FZ, 11.0, 0.0, 0.8, (0.20, 0.09, 0.03, 1), tex=self.wood)
        self.wall_rect("F", da - 4.5, da + 4.5, FZ, 10.4, 0.8, 1.3, (0.30, 0.13, 0.05, 1), tex=self.wood, spec=(0.4, 30))
        for z0, z1 in [(-10.5, -4), (-3, 3), (4, 9.6)]:
            self.wall_rect("F", da - 3.2, da + 3.2, z0, z1, 1.3, 1.5, (0.22, 0.09, 0.03, 1), tex=self.wood)
        self.wall_rect("F", da - 2.6, da + 2.6, 4.6, 9.0, 1.5, 1.6, (1.0, 0.85, 0.5, 1), light=False)
        knob = self.root.attachNewNode(self.cached("knob", lambda: sphere_node(0.45, 8, 12, (0.82, 0.66, 0.28, 1))))
        knob.setPos(da + 3.6, self.Y0 + 1.8, -2)
        shiny(knob, 1, 90)
        self.picture("F", -22, 11, 11, 7, 0)
        self.picture("F", -46, 8, 6, 8, 1)

    # ----------------------------------------------------- extra details
    def glow(self, pos, size, color, parent=None):
        return make_glow(parent or self.root, self.t_glow, pos, size, color)

    def wall_point(self, wall, a, z, d):
        if wall == "L":
            return (self.X0 + d, a, z)
        if wall == "R":
            return (self.X1 - d, a, z)
        if wall == "F":
            return (a, self.Y0 + d, z)
        return (a, self.Y1 - d, z)

    def plant(self, x, y, s=1.0):
        FZ = self.FZ
        terra = (0.62, 0.30, 0.15, 1)
        self.vlathe([(0, 1.5 * s, terra), (0.3, 1.5 * s, (0.5, 0.25, 0.12, 1)),
                     (3.0 * s, 2.2 * s, terra), (3.3 * s, 2.4 * s, terra)], x, y, FZ, spec=(0.3, 20))
        self.cap(x, y, FZ + 3.1 * s, 2.25 * s, (0.12, 0.07, 0.04, 1))
        leaf = self.cached(("sph", "leaf"), lambda: sphere_node(1.0, 6, 10, (1, 1, 1, 1)))
        rnd = random.Random(int(x * 7 + y * 13))
        for _ in range(16):
            piv = self.root.attachNewNode("leaf")
            piv.setPos(x, y, FZ + 3.1 * s)
            piv.setHpr(rnd.uniform(0, 360), rnd.uniform(25, 78), 0)
            l = piv.attachNewNode(leaf)
            l.setY(2.7 * s)
            l.setScale(0.75 * s, 2.7 * s, 0.1 * s)
            g = rnd.uniform(0.7, 1.25)
            l.setColor(0.10 * g, 0.38 * g, 0.12 * g, 1)
            shiny(l, 0.4, 30)

    def pendant(self, x, y, zb=9.0):
        c = self.ceiling
        dark = (0.08, 0.22, 0.13, 1)
        self.vlathe([(0, 2.8, dark), (0.25, 2.8, dark), (2.2, 0.6, dark)], x, y, zb,
                    parent=c, spec=(0.8, 80))
        self.vlathe([(0.05, 2.6, (1.0, 0.92, 0.7, 1)), (2.0, 0.5, (1.0, 0.92, 0.7, 1))], x, y, zb,
                    parent=c, light=False)
        bulb = c.attachNewNode(self.cached(("sph", "bulb"), lambda: sphere_node(0.6, 8, 12, (1, 0.95, 0.7, 1))))
        bulb.setPos(x, y, zb + 0.5)
        bulb.setLightOff()
        self.vlathe([(zb + 2.2 - zb, 0.1, (0.05, 0.05, 0.05, 1)), (self.CZ - zb, 0.1, (0.05, 0.05, 0.05, 1))],
                    x, y, zb, parent=c, segs=6)
        self.glow((x, y, zb - 1.0), 6.5, (1.0, 0.82, 0.45, 0.5), c)

    def build_props(self):
        FZ = self.FZ
        # ---- jukebox (right wall, back corner)
        wood = (0.35, 0.12, 0.06, 1)
        jx0, jx1, jy0, jy1 = 47.0, 54.6, 43.0, 51.0
        jc = (jy0 + jy1) / 2
        self.box(jx0, jx1, jy0, jy1, FZ, FZ + 13, wood, self.wood, spec=(0.7, 60))
        arch = self.root.attachNewNode(self.cached(("lathe", "jukearch"), lambda: lathe_node(
            [(0, 4.0, wood), (jx1 - jx0, 4.0, wood)], 24)))
        arch.setPos(jx0, jc, FZ + 13)
        arch.setH(-90)
        shiny(arch, 0.7, 60)
        cols = [(1, 0.2, 0.2, 1), (1, 0.7, 0.1, 1), (0.2, 0.85, 0.3, 1), (0.2, 0.5, 1, 1), (0.85, 0.25, 0.95, 1)]
        for k in range(8):
            dy = -3.5 + k
            top = FZ + 13 + math.sqrt(max(0.0, 16 - (dy + 0.5) ** 2)) - 0.4
            self.box(jx0 - 0.25, jx0, jc + dy, jc + dy + 0.85, FZ + 5.0, top, cols[k % 5], light=False)
        self.box(jx0 - 0.3, jx0, jc - 3.2, jc + 3.2, FZ + 5.2, FZ + 9.6, (1.0, 0.75, 0.3, 1), light=False)
        self.box(jx0 - 0.5, jx0 - 0.1, jc - 3.6, jc + 3.6, FZ + 0.8, FZ + 4.6, (0.04, 0.04, 0.04, 1), spec=(0.5, 40))
        self.box(jx0 - 0.4, jx0 - 0.1, jc - 3.6, jc + 3.6, FZ + 9.7, FZ + 10.0, (0.8, 0.8, 0.85, 1), spec=(1, 100))
        self.glow((jx0 - 1.2, jc, FZ + 11), 10.0, (1.0, 0.6, 0.3, 0.45))
        # ---- fruit machine (left wall, front corner)
        self.box(-54.6, -49.0, -56.0, -50.0, FZ, FZ + 16, (0.05, 0.10, 0.35, 1), spec=(0.7, 60))
        self.box(-49.1, -48.9, -55.4, -50.6, FZ + 8.0, FZ + 13.0, (0.05, 0.05, 0.06, 1))
        for k in range(3):
            self.box(-48.95, -48.85, -55.0 + k * 1.6, -53.7 + k * 1.6, FZ + 9.0, FZ + 12.0,
                     (1.0, 0.95, 0.8, 1), light=False)
            self.box(-48.9, -48.8, -54.7 + k * 1.6, -54.0 + k * 1.6, FZ + 10.0, FZ + 11.0, cols[k], light=False)
        self.box(-49.0, -48.8, -55.4, -50.6, FZ + 13.6, FZ + 15.6, (0.85, 0.1, 0.15, 1), light=False)
        self.text3d("JACKPOT", (-48.7, -53.0, FZ + 14.0), 90, 0.9, (1, 0.95, 0.4, 1))
        for k, cc in enumerate([(1, 0.2, 0.2, 1), (0.2, 0.9, 0.3, 1), (1, 0.8, 0.1, 1)]):
            self.box(-49.0, -48.6, -54.6 + k * 1.8, -53.6 + k * 1.8, FZ + 6.0, FZ + 7.0, cc, light=False)
        self.glow((-48.0, -53.0, FZ + 11), 7.0, (0.8, 0.5, 1.0, 0.35))
        # ---- plants
        for px, py, sc in [(-50, -58, 1.2), (51, -58, 1.2), (-50, 38, 1.0), (50, 36, 1.1)]:
            self.plant(px, py, sc)
        self.build_pinball()

    def build_pinball(self):
        """Pinball machine in the back-left corner, backbox against the left wall."""
        FZ, X0 = self.FZ, self.X0
        cy, half = 50.0, 5.0
        xb, xf = X0 + 0.5, X0 + 19.5
        zb, zf, zbot = FZ + 13.0, FZ + 11.0, FZ + 8.5
        y0, y1 = cy - half - 0.3, cy + half + 0.3
        chrome = (0.78, 0.80, 0.84, 1)
        red = (0.85, 0.1, 0.1, 1)
        # ---- cabinet (sloping top) and legs
        top = [(xb, y0, zb), (xb, y1, zb), (xf, y1, zf), (xf, y0, zf)]
        bot = [(xb, y0, zbot), (xb, y1, zbot), (xf, y1, zbot), (xf, y0, zbot)]
        faces = [top, bot, [top[0], top[1], bot[1], bot[0]], [top[2], top[3], bot[3], bot[2]],
                 [top[0], top[3], bot[3], bot[0]], [top[1], top[2], bot[2], bot[1]]]
        cab = self.root.attachNewNode(solid_node(faces, (0.06, 0.08, 0.32, 1), 0.1))
        shiny(cab, 0.8, 70)
        dark = (0.1, 0.1, 0.1, 1)
        for lx in (xb + 0.7, xf - 0.7):
            for ly in (y0 - 0.1, y1 + 0.1):          # legs sit on the outside corners
                leg = self.root.attachNewNode(lathe_node(
                    [(0, 0.6, dark), (0.35, 0.6, dark), (0.35, 0.45, chrome),
                     (zbot - FZ, 0.45, chrome)], 16))
                leg.setPos(lx, ly, FZ)
                leg.setP(90)
                leg.setTwoSided(True)
                shiny(leg, 0.9, 90)
                self.box(lx - 0.6, lx + 0.6, ly - 0.6, ly + 0.6, zbot - 0.3, zbot + 0.1, dark)
        # ---- playfield (tilted to match the cabinet top)
        field = self.root.attachNewNode("pinfield")
        field.setPos((xb + xf) / 2, cy, (zb + zf) / 2 + 0.05)
        field.setR(math.degrees(math.atan2(zb - zf, xf - xb)))
        L, W = 9.2, 4.9
        self.box(-L, L, -W, W, -0.05, 0.05, (0.05, 0.05, 0.30, 1), parent=field, spec=(0.6, 60))
        for sgn in (-1, 1):
            self.box(-L - 0.2, L + 0.2, sgn * W, sgn * (W + 0.3), 0, 0.9, (0.78, 0.8, 0.84, 1),
                     parent=field, spec=(1.0, 90))
        self.box(-L - 0.3, -L, -W, W, 0, 0.9, (0.1, 0.1, 0.12, 1), parent=field)
        rnd = random.Random(11)
        insert_cols = [(1, 0.25, 0.25, 1), (1, 0.85, 0.2, 1), (0.3, 0.9, 0.4, 1), (0.3, 0.6, 1, 1), (0.9, 0.4, 1, 1)]
        for _ in range(22):
            self.cap(rnd.uniform(-L + 1, L - 2), rnd.uniform(-W + 0.8, W - 0.8), 0.07, 0.26,
                     rnd.choice(insert_cols), parent=field, light=False)
        for bx, by in [(-3.5, 0.0), (-5.2, -2.3), (-5.2, 2.3)]:      # pop bumpers
            self.vlathe([(0, 0.9, red), (0.55, 0.9, red), (0.95, 0.6, (0.95, 0.95, 0.95, 1)),
                         (1.05, 0.6, (1.0, 0.9, 0.3, 1))], bx, by, 0.05, parent=field, spec=(0.8, 70))
            self.glow_local(field, (bx, by, 1.2), 2.2, (1.0, 0.6, 0.3, 0.5))
        for sgn in (-1, 1):                                           # flippers
            pv = field.attachNewNode("flipper")
            pv.setPos(6.3, sgn * 1.9, 0.05)
            pv.setH(-sgn * 28)
            self.box(0, 2.0, -0.22, 0.22, 0, 0.45, (0.95, 0.95, 0.95, 1), parent=pv, spec=(0.6, 50))
            self.vlathe([(0, 0.3, red), (0.5, 0.3, red)], 0, 0, 0.05, parent=pv)
            self.vlathe([(0, 0.35, chrome), (0.8, 0.35, chrome)], 4.4 * 0, sgn * 4.4, 0.0, parent=field,
                        spec=(1, 90))
        ball = field.attachNewNode(self.cached(("sph", "pinball"), lambda: sphere_node(
            0.4, 10, 16, (0.85, 0.87, 0.90, 1))))
        ball.setPos(4.8, 0.4, 0.45)
        shiny(ball, 1, 100)
        apply_env(ball, 0.7)
        glass = self.box(-L, L, -W, W, 1.15, 1.2, (0.7, 0.85, 1.0, 0.12), parent=field, spec=(1, 120))
        glass.setTransparency(TransparencyAttrib.MAlpha)
        glass.setTwoSided(True)
        # ---- front: coin door, buttons, plunger
        self.box(xf, xf + 0.12, cy - 1.3, cy + 1.3, zbot + 0.4, zbot + 3.3, (0.3, 0.3, 0.35, 1), spec=(0.8, 70))
        for k in (-0.6, 0.6):
            self.box(xf + 0.1, xf + 0.2, cy + k - 0.25, cy + k + 0.25, zbot + 1.6, zbot + 2.6, (1, 0.2, 0.2, 1), light=False)
        for sgn in (-1, 1):
            btn = self.root.attachNewNode(self.cached(("sph", "pbtn"), lambda: sphere_node(
                0.35, 8, 12, (0.9, 0.15, 0.15, 1))))
            btn.setPos(xf - 3.5, cy + sgn * (half + 0.45), zf - 0.35)
            shiny(btn, 0.8, 60)
        plunger = self.root.attachNewNode(self.cached(("lathe", "plunger"), lambda: lathe_node(
            [(0, 0.2, chrome), (2.2, 0.2, chrome), (2.2, 0.4, red), (2.8, 0.4, red)], 10)))
        plunger.setPos(xf, cy + 4.5, zf - 0.1)
        plunger.setH(-90)
        shiny(plunger, 0.9, 80)
        # ---- backbox with lit translite
        bx0, bx1 = X0 + 0.05, X0 + 3.2
        self.box(bx0, bx1, y0 - 0.1, y1 + 0.1, zb, zb + 12.0, (0.04, 0.04, 0.12, 1), spec=(0.6, 50))
        self.box(bx1 - 0.05, bx1 + 0.1, y0 + 0.3, y1 - 0.3, zb + 3.6, zb + 11.4, (0.05, 0.03, 0.28, 1), light=False)
        self.box(bx1 - 0.05, bx1 + 0.12, y0 + 0.8, y1 - 0.8, zb + 0.7, zb + 3.0, (0.02, 0.0, 0.0, 1), light=False)
        self.box(bx0, bx1 + 0.3, y0 - 0.2, y1 + 0.2, zb + 11.4, zb + 12.3, (0.35, 0.05, 0.55, 1), spec=(0.8, 70))
        for pc, pr, py, pz in [((1.0, 0.55, 0.15, 1), 1.7, cy - 2.4, zb + 6.0), ((0.3, 0.9, 1.0, 1), 1.0, cy + 2.8, zb + 8.2),
                               ((0.8, 0.3, 1.0, 1), 0.7, cy + 1.6, zb + 5.2)]:
            pl = self.root.attachNewNode(self.cached(("sph", pc), lambda c=pc: sphere_node(1.0, 10, 16, c)))
            pl.setPos(bx1 + 0.1, py, pz)
            pl.setScale(0.05, pr, pr)
            pl.setLightOff()
        for _ in range(14):
            st = self.root.attachNewNode(self.cached(("sph", "star"), lambda: sphere_node(
                1.0, 6, 8, (1, 1, 0.9, 1))))
            st.setPos(bx1 + 0.1, rnd.uniform(y0 + 0.8, y1 - 0.8), rnd.uniform(zb + 4.0, zb + 11.0))
            st.setScale(0.04, 0.12, 0.12)
            st.setLightOff()
        self.text3d("COSMIC 8-BALL", (bx1 + 0.2, cy, zb + 9.6), 90, 0.85, (1.0, 0.85, 0.2, 1))
        self.text3d("00 042 690", (bx1 + 0.2, cy, zb + 1.25), 90, 0.75, (1.0, 0.15, 0.1, 1))
        self.glow((bx1 + 1.5, cy, zb + 6.5), 15.0, (0.5, 0.4, 1.0, 0.40))
        self.glow((xb + 9.5, cy, zb + 1.5), 9.0, (1.0, 0.7, 0.9, 0.30))

    def glow_local(self, parent, pos, size, color):
        return make_glow(parent, self.t_glow, pos, size, color)

    def build_glows(self):
        c = self.ceiling
        # hanging lamp halos + light shaft down onto the table
        for y in (-14, 0, 14):
            self.glow((0, y, 20.6), 8.5, (1.0, 0.85, 0.5, 0.55), c)
        m = Mesh()
        top, bot = 21.05, 1.3
        tc = [(-5.0, -20.5), (5.0, -20.5), (5.0, 20.5), (-5.0, 20.5)]
        bc = [(-13.0, -27.0), (13.0, -27.0), (13.0, 27.0), (-13.0, 27.0)]
        for i in range(4):
            j = (i + 1) % 4
            ids = [m.vert((tc[i][0], tc[i][1], top), (0, 0, 1), (1, 0.92, 0.7, 0.13)),
                   m.vert((tc[j][0], tc[j][1], top), (0, 0, 1), (1, 0.92, 0.7, 0.13)),
                   m.vert((bc[j][0], bc[j][1], bot), (0, 0, 1), (1, 0.92, 0.7, 0.0)),
                   m.vert((bc[i][0], bc[i][1], bot), (0, 0, 1), (1, 0.92, 0.7, 0.0))]
            m.tri(ids[0], ids[1], ids[2])
            m.tri(ids[0], ids[2], ids[3])
        shaft = c.attachNewNode(m.node("shaft"))
        shaft.setTwoSided(True)
        shaft.setLightOff()
        shaft.setDepthWrite(False)
        shaft.setAttrib(ColorBlendAttrib.make(ColorBlendAttrib.MAdd,
                                              ColorBlendAttrib.OIncomingAlpha, ColorBlendAttrib.OOne))
        shaft.setBin("transparent", 10)
        # drifting dust motes inside the shaft
        for _ in range(36):
            base = (random.uniform(-8, 8), random.uniform(-19, 19), random.uniform(2, 19))
            np = self.glow(base, 0.22, (1.0, 0.95, 0.8, 0.55), c)
            self.motes.append((np, base, random.uniform(0, 6.28), random.uniform(0.2, 0.6)))
        # fire, signs, moon, pendants, candles, back-bar LEDs
        g = self.glow((0, self.Y1 - 7.0, self.FZ + 4.0), 11.0, (1.0, 0.45, 0.1, 0.7))
        self.fire_glows.append((g, 11.0))
        self.glow(self.wall_point("R", 31, 17.5, 0.9), 12.0, (1.0, 0.2, 0.55, 0.40))
        self.glow((0, self.Y1 - 8.6, self.CZ - 8.0), 14.0, (1.0, 0.75, 0.25, 0.30))
        moon = self.root.attachNewNode(self.cached(("sph", "moon"), lambda: sphere_node(
            1.0, 10, 16, (0.95, 0.95, 1.0, 1))))
        moon.setPos(18.0, self.Y0 + 0.4, 15.0)
        moon.setLightOff()
        self.glow((18.0, self.Y0 + 1.0, 15.0), 7.0, (0.55, 0.65, 1.0, 0.5))
        for y in (-20, 0, 20):
            self.glow((self.X0 + 3.5, y, 14.4), 5.0, (1.0, 0.8, 0.5, 0.35))

    # ---------------------------------------------------------- ceiling
    def build_ceiling(self):
        X0, X1, Y0, Y1, CZ = self.X0, self.X1, self.Y0, self.Y1, self.CZ
        c = self.ceiling
        self.quad([(X0, Y0, CZ), (X0, Y1, CZ), (X1, Y1, CZ), (X1, Y0, CZ)],
                  self.t_plaster, (0.92, 0.86, 0.74, 1), 0.05, parent=c)
        beam = (0.16, 0.07, 0.03, 1)
        for y in (-48, -30, -14, 14, 30, 48):
            self.box(X0, X1, y - 1.5, y + 1.5, CZ - 3, CZ, beam, self.wood, c, spec=(0.3, 25))
        for wall, (a0, a1) in {"L": (Y0, Y1), "R": (Y0, Y1), "F": (X0, X1), "B": (X0, X1)}.items():
            self.wall_rect(wall, a0, a1, CZ - 1.4, CZ, 0, 1.2, beam, tex=self.wood, parent=c)
        # billiard lamp: hood, glowing underside, cords
        hood = (0.05, 0.28, 0.14, 1)
        brass = (0.82, 0.66, 0.28, 1)
        self.box(-5.5, 5.5, -21, 21, 21.2, 22.7, hood, spec=(0.8, 80), parent=c)
        self.box(-4.0, 4.0, -19, 19, 22.7, 23.5, hood, spec=(0.8, 80), parent=c)
        for x0, x1, y0, y1 in [(-5.6, 5.6, -21.1, -20.8), (-5.6, 5.6, 20.8, 21.1),
                               (-5.6, -5.3, -21.1, 21.1), (5.3, 5.6, -21.1, 21.1)]:
            self.box(x0, x1, y0, y1, 21.1, 21.4, brass, spec=(1, 90), parent=c)
        self.box(-5.2, 5.2, -20.6, 20.6, 21.1, 21.22, (1.0, 0.94, 0.72, 1), light=False, parent=c)
        for x in (-3, 3):
            for y in (-15, 15):
                self.vlathe([(0, 0.12, (0.05, 0.05, 0.05, 1)), (CZ - 23.5, 0.12, (0.05, 0.05, 0.05, 1))],
                            x, y, 23.5, parent=c, segs=6)

    # ------------------------------------------------------------ lights
    def build_lights(self):
        fire = PointLight("fire")
        fire.setColor(Vec4(1.0, 0.45, 0.15, 1))
        fire.setAttenuation(Vec3(1, 0.02, 0.003))
        fnp = self.render.attachNewNode(fire)
        fnp.setPos(0, self.Y1 - 6.5, self.FZ + 4)
        self.render.setLight(fnp)
        self.fire_light = fire
        bar = PointLight("barlight")
        bar.setColor(Vec4(0.9, 0.65, 0.35, 1))
        bar.setAttenuation(Vec3(1, 0.02, 0.004))
        bnp = self.render.attachNewNode(bar)
        bnp.setPos(-44, 0, 8)
        self.render.setLight(bnp)

    def update(self, dt, cam_z, show_ceiling=True):
        self.time += dt
        t = self.time
        f = 0.85 + 0.15 * math.sin(t * 13.0) + 0.08 * random.random() - 0.04
        self.fire_light.setColor(Vec4(1.0 * f, 0.45 * f, 0.15 * f, 1))
        for node, sc, i in self.flames:
            hs = sc * (0.85 + 0.3 * math.sin(t * 9.0 + i * 1.7) + 0.1 * random.random())
            ws = sc * (0.9 + 0.1 * math.sin(t * 7.0 + i))
            node.setScale(ws, hs, ws)
        for g, size in self.fire_glows:
            g.setScale(size * (0.9 + 0.12 * math.sin(t * 11.0) + 0.05 * random.random()))
        for np, base, ph, sp in self.motes:
            np.setPos(base[0] + math.sin(t * sp + ph) * 0.9, base[1] + math.cos(t * sp * 0.7 + ph) * 0.9,
                      base[2] + math.sin(t * sp * 0.5 + ph * 2) * 0.6)
        if cam_z > self.CZ - 3.5 or not show_ceiling:
            self.ceiling.hide()
        else:
            self.ceiling.show()


# ============================================================================
# Simple synthesized sounds
# ============================================================================
RATE = 22050


def _noise():
    return random.random() * 2 - 1


def _synth(dur, fn):
    return [fn(i / RATE) for i in range(int(RATE * dur))]


def _write_wav(path, samples):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(b"".join(struct.pack("<h", int(max(-1.0, min(1.0, s)) * 30000))
                               for s in samples))


def _cue_fn(t):
    return (0.9 * math.sin(2 * math.pi * 220 * t) * math.exp(-t * 45)
            + 0.6 * _noise() * math.exp(-t * 140)
            + 0.4 * math.sin(2 * math.pi * 1500 * t) * math.exp(-t * 90))


def _ball_fn(t):
    return (0.8 * math.sin(2 * math.pi * 2800 * t) * math.exp(-t * 110)
            + 0.5 * math.sin(2 * math.pi * 4100 * t) * math.exp(-t * 160)
            + 0.3 * math.sin(2 * math.pi * 1300 * t) * math.exp(-t * 60)
            + 0.3 * _noise() * math.exp(-t * 300))


def _cushion_fn(t):
    return (0.9 * math.sin(2 * math.pi * 120 * t) * math.exp(-t * 28)
            + 0.4 * math.sin(2 * math.pi * 260 * t) * math.exp(-t * 45)
            + 0.3 * _noise() * math.exp(-t * 100))


def _pocket_fn(t):
    v = 0.6 * math.sin(2 * math.pi * (180 - 250 * t) * t) * math.exp(-t * 6)
    v += 0.35 * _noise() * math.exp(-t * 14) * (0.5 + 0.5 * math.sin(2 * math.pi * 35 * t))
    t2 = t - 0.28
    if t2 > 0:
        v += 0.5 * math.sin(2 * math.pi * 2200 * t2) * math.exp(-t2 * 80)
    return v


class SoundBank:
    def __init__(self, base):
        self.sounds = {}
        self.index = {}
        try:
            folder = tempfile.mkdtemp(prefix="pool_sfx_")
            specs = {"cue": (0.15, _cue_fn, 2), "ball": (0.15, _ball_fn, 4),
                     "cushion": (0.2, _cushion_fn, 4), "pocket": (0.6, _pocket_fn, 2)}
            for name, (dur, fn, copies) in specs.items():
                path = os.path.join(folder, name + ".wav")
                _write_wav(path, _synth(dur, fn))
                fname = Filename.fromOsSpecific(path)
                self.sounds[name] = [base.loader.loadSfx(fname) for _ in range(copies)]
                self.index[name] = 0
        except Exception as exc:
            print("Sound disabled:", exc)
            self.sounds = {}

    def play(self, name, vol=1.0):
        lst = self.sounds.get(name)
        if not lst:
            return
        snd = lst[self.index[name] % len(lst)]
        self.index[name] += 1
        snd.setVolume(max(0.05, min(1.0, vol)))
        snd.setPlayRate(random.uniform(0.95, 1.05))
        snd.play()


# ============================================================================
# Replays: recording, saving to disk, and looping playback of best pots
# ============================================================================
REC_DT = 1.0 / 30.0           # recording rate for replays


class ReplayStore:
    """Keeps every saved game (gzip JSON) in a folder, with a small index file."""

    def __init__(self):
        self.dir = None
        for d in (os.path.join(os.path.dirname(os.path.abspath(__file__)), "pool_replays"),
                  os.path.join(os.path.expanduser("~"), ".pool_replays")):
            try:
                os.makedirs(d, exist_ok=True)
                probe = os.path.join(d, ".probe")
                with open(probe, "w") as f:
                    f.write("x")
                os.remove(probe)
                self.dir = d
                break
            except Exception:
                continue
        self.index = self._load_index()

    def _path(self, name):
        return os.path.join(self.dir, name)

    def _read(self, fname):
        with gzip.open(self._path(fname), "rt", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _entry(fname, rec):
        return {"file": fname, "date": rec.get("date", ""), "players": rec.get("players", ["?", "?"]),
                "scores": rec.get("scores", [0, 0]), "two_player": rec.get("two_player", True),
                "shots": len(rec.get("shots", []))}

    def _load_index(self):
        if not self.dir:
            return []
        try:
            with open(self._path("index.json"), "r", encoding="utf-8") as f:
                idx = [e for e in json.load(f) if os.path.exists(self._path(e["file"]))]
        except Exception:
            idx = []
        known = {e["file"] for e in idx}
        try:
            for fn in sorted(os.listdir(self.dir)):          # pick up games missing from the index
                if fn.startswith("game_") and fn.endswith(".json.gz") and fn not in known:
                    try:
                        idx.append(self._entry(fn, self._read(fn)))
                    except Exception:
                        pass
        except Exception:
            pass
        idx.sort(key=lambda e: e["file"])
        return idx

    def save(self, rec):
        if not self.dir:
            return None
        try:
            base = time.strftime("game_%Y%m%d_%H%M%S")
            fname, n = base + ".json.gz", 1
            while os.path.exists(self._path(fname)):
                n += 1
                fname = f"{base}_{n}.json.gz"
            with gzip.open(self._path(fname), "wt", encoding="utf-8") as f:
                json.dump(rec, f, separators=(",", ":"))
            self.index.append(self._entry(fname, rec))
            with open(self._path("index.json"), "w", encoding="utf-8") as f:
                json.dump(self.index, f)
            return fname
        except Exception as exc:
            print("Could not save replay:", exc)
            return None

    def load(self, entry):
        try:
            return self._read(entry["file"])
        except Exception as exc:
            print("Could not load replay:", exc)
            return None


def _nlerp(q0, q1, a):
    d = sum(q0[k] * q1[k] for k in range(4))
    if d < 0:
        q1 = [-v for v in q1]
    q = [q0[k] + (q1[k] - q0[k]) * a for k in range(4)]
    n = math.sqrt(sum(v * v for v in q)) or 1.0
    return [v / n for v in q]


class ReplayPlayer:
    """Plays recorded shots on the table, one after another, in a loop."""
    PRE, LUNGE, HOLD = 1.0, 0.12, 1.8

    def __init__(self, game):
        self.g = game
        self.active = False
        self.balls = {}
        self.shots = []
        self.i = 0
        self.t = 0.0

    def start(self, shots):
        self.stop()
        if not shots:
            return
        self.g.set_live_visible(False)
        self.shots = shots
        self.active = True
        self.load(0)

    def stop(self):
        if not self.active:
            return
        self.active = False
        for b in self.balls.values():
            b.destroy()
        self.balls = {}
        self.g.cue.np.hide()
        self.g.hl_text.setText("")
        self.g.hl_sub.setText("")
        self.g.set_live_visible(True)

    def load(self, i):
        self.i = i
        shot = self.shots[i]
        self.shot = shot
        self.order = shot["order"]
        self.frames = shot["frames"]
        self.dt = shot.get("dt", REC_DT)
        self.cue_idx = self.order.index(0) if 0 in self.order else None
        for n in self.order:
            if n not in self.balls:
                b = Ball(self.g, n, Point3(0, 0, -40))
                b.leave_world()                       # visual only, no physics
                self.balls[n] = b
        for n, b in self.balls.items():
            if n not in self.order:
                b.np.hide()
                b.shadow.hide()
        self.t = 0.0
        self.set_pose(0, 0, 0.0)
        self.g.hl_text.setText(f"Highlights: {shot.get('name', '')}")
        parts = ", ".join(shot.get("parts", []))
        self.g.hl_sub.setText(f"Shot {i + 1}/{len(self.shots)}   +{shot.get('points', 0)}   {parts}")

    def set_pose(self, i0, i1, a):
        high = self.g.high_quality
        r0s, r1s = self.frames[i0], self.frames[i1]
        for k, num in enumerate(self.order):
            b = self.balls[num]
            r0, r1 = r0s[k], r1s[k]
            if r0 is None:
                b.np.hide()
                b.shadow.hide()
                continue
            if r1 is None:
                r1 = r0
            b.np.show()
            x = r0[0] + (r1[0] - r0[0]) * a
            y = r0[1] + (r1[1] - r0[1]) * a
            z = r0[2] + (r1[2] - r0[2]) * a
            s = r0[3] + (r1[3] - r0[3]) * a
            b.np.setPos(x, y, z)
            b.np.setScale(s)
            q = _nlerp(r0[4:8], r1[4:8], a)
            b.np.setQuat(Quat(q[3], q[0], q[1], q[2]))
            if high or z < 0.2:
                b.shadow.hide()
            else:
                b.shadow.setPos(x + 0.12, y - 0.08, 0.012)
                b.shadow.show()

    def update(self, dt):
        if not self.active:
            return
        self.t += dt
        n = len(self.frames)
        play = max(0, n - 1) * self.dt
        pre, lunge = self.PRE, self.LUNGE
        t = self.t
        if t < pre + lunge:                                   # cue stick draws back and strikes
            self.set_pose(0, 0, 0.0)
            row = self.frames[0][self.cue_idx] if self.cue_idx is not None else None
            if row:
                pull = 0.35 + self.shot.get("power", 0.5) * 5.0
                if t < pre:
                    pull *= min(1.0, t / pre)
                else:
                    k = (t - pre) / lunge
                    pull *= (1 - k * k)
                h = math.radians(self.shot.get("aim_h", 0.0))
                dx, dy = -math.sin(h), math.cos(h)
                off = R + 0.04 + pull
                self.g.cue.place(Point3(row[0] - dx * off, row[1] - dy * off, R), self.shot.get("aim_h", 0.0))
        elif t < pre + lunge + play:
            self.g.cue.np.hide()
            f = (t - pre - lunge) / self.dt
            i0 = int(f)
            if i0 >= n - 1:
                self.set_pose(n - 1, n - 1, 0.0)
            else:
                self.set_pose(i0, i0 + 1, f - i0)
        elif t < pre + lunge + play + self.HOLD:
            self.g.cue.np.hide()
            self.set_pose(n - 1, n - 1, 0.0)
        else:
            self.load((self.i + 1) % len(self.shots))          # loop to the next highlight


class PoolGame(ShowBase):
    CAM_NAMES = ["Overview", "Player view", "Top-down"]

    def __init__(self):
        ShowBase.__init__(self)
        props = WindowProperties()
        props.setTitle("8-Ball Pool")
        self.win.requestProperties(props)
        self.setBackgroundColor(0.02, 0.02, 0.03)
        self.render.setAntialias(AntialiasAttrib.MAuto)
        self.disableMouse()

        self.world = bp.BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))

        self.balls = []
        self.cue_ball = None
        self.sinking = []
        self.guide = None
        try:   # re-apply rolling before every physics sub-step
            self.world.setTickCallback(CallbackObject.make(self.enforce_rolling), True)
        except Exception:
            pass
        self.dot_geom = sphere_node(1.0, 8, 12, (1.0, 0.15, 0.1, 1))

        self.two_player = True
        self.cam_mode = 0
        self.cam_pos = Vec3(0, -25, 45)
        self.player_anchor = Point3(0, -10, R)
        self.player_aim = Vec3(0, 1, 0)
        self.last_mouse = (0, 0)
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.cam_zoom = 1.0
        self.rmb = False
        self.lmb = False
        self.rmb_last = None
        self.touching = set()

        self.sfx = SoundBank(self)
        self.store = ReplayStore()
        self.replay = ReplayPlayer(self)
        self.names = ["Player 1", "Player 2"]
        self.menu_open = False
        self.game_shots = []
        self.game_saved = True
        self.rec_balls, self.rec_order, self.rec_frames, self.rec_acc = {}, [], [], 0.0
        self.rec_meta = {}
        self.table = Table(self)
        self.pub = Pub(self, self.table.wood_tex)
        self.cue = CueStick(self)
        self.setup_lights()
        self.setup_input()
        self.setup_ui()
        self.setup_menu()
        self.reset_game()
        self.camLens.setFov(55)
        self.set_quality(True)
        self.open_menu()

        self.taskMgr.add(self.update, "update")

    # ------------------------------------------------------------------ setup
    def setup_lights(self):
        a = AmbientLight("ambient")
        a.setColor(Vec4(0.32, 0.32, 0.38, 1))
        self.render.setLight(self.render.attachNewNode(a))

        d = DirectionalLight("sun")
        d.setColor(Vec4(0.35, 0.35, 0.33, 1))
        d.setSpecularColor(Vec4(0.6, 0.6, 0.6, 1))
        dnp = self.render.attachNewNode(d)
        dnp.setHpr(25, -55, 0)
        self.render.setLight(dnp)

        self.high_quality = True
        self.lamp_lights = []
        for y in (-10, 10):                        # two hanging lamps
            p = PointLight(f"lamp{y}")
            p.setColor(Vec4(0.75, 0.70, 0.60, 1))
            p.setSpecularColor(Vec4(1, 1, 1, 1))
            pnp = self.render.attachNewNode(p)
            pnp.setPos(0, y, 19.5)
            self.render.setLight(pnp)
            self.lamp_lights.append(p)

        # real shadows (High graphics): a shadow-casting sun just under the lamp hood
        sd = DirectionalLight("shadowsun")
        sd.setColor(Vec4(0.50, 0.50, 0.50, 1))
        sd.setSpecularColor(Vec4(0.6, 0.6, 0.6, 1))
        sd.setShadowCaster(True, 2048, 2048)
        lens = sd.getLens()
        lens.setFilmSize(31, 52)
        lens.setNearFar(0.5, 40)
        self.shadow_np = self.render.attachNewNode(sd)
        self.shadow_np.setPos(0, -2.9, 13.8)
        self.shadow_np.setHpr(0, -78, 0)

        # faint warm haze for depth
        fog = Fog("haze")
        fog.setColor(0.07, 0.05, 0.04)
        fog.setExpDensity(0.0016)
        self.render.setFog(fog)

    def setup_input(self):
        self.accept("mouse1", self.set_lmb, [True])
        self.accept("mouse1-up", self.set_lmb, [False])
        self.accept("mouse2", self.left_click)
        self.accept("mouse3", self.set_rmb, [True])
        self.accept("mouse3-up", self.set_rmb, [False])
        self.accept("wheel_up", self.wheel_power, [1])
        self.accept("wheel_down", self.wheel_power, [-1])
        for ev, f in (("arrow_up", 0.93), ("arrow_down", 1.07)):
            self.accept(ev, self.zoom_step, [f])
            self.accept(ev + "-repeat", self.zoom_step, [f])
        self.accept("space", self.start_charge)
        self.accept("space-up", self.release_charge)
        self.accept("c", self.cycle_camera)
        self.accept("m", self.toggle_mode)
        self.accept("g", self.toggle_quality)
        self.accept("r", self.new_game_request)
        self.accept("escape", self.escape_pressed)

        def ascii_key(c):
            try:
                return KeyboardButton.asciiKey(c)
            except TypeError:
                return KeyboardButton.asciiKey(c.encode())
        self.k_left = [KeyboardButton.left(), ascii_key("a")]
        self.k_right = [KeyboardButton.right(), ascii_key("d")]
        self.k_shift = KeyboardButton.shift()

    def key_down(self, handles):
        mw = self.mouseWatcherNode
        if not isinstance(handles, list):
            handles = [handles]
        return any(mw.isButtonDown(h) for h in handles)

    def setup_ui(self):
        fg = (0.9, 0.9, 0.9, 1)
        self.hud = DirectFrame(parent=self.a2dTopLeft, frameColor=(0, 0, 0, 0.55),
                               frameSize=(0, 1.2, -0.66, 0), pos=(0.03, 0, -0.03))
        self.name_txt, self.score_txt = [], []
        for i in range(2):
            y = -0.10 - i * 0.11
            self.name_txt.append(OnscreenText(parent=self.hud, text="", pos=(0.05, y),
                                              scale=0.065, align=TextNode.ALeft, fg=fg,
                                              mayChange=True))
            self.score_txt.append(OnscreenText(parent=self.hud, text="0", pos=(1.15, y),
                                               scale=0.065, align=TextNode.ARight, fg=fg,
                                               mayChange=True))
        self.left_txt = OnscreenText(parent=self.hud, text="", pos=(0.05, -0.34), scale=0.045,
                                     align=TextNode.ALeft, fg=(0.8, 0.9, 1, 1), mayChange=True)
        self.cam_txt = OnscreenText(parent=self.hud, text="", pos=(0.05, -0.45), scale=0.045,
                                    align=TextNode.ALeft, fg=(0.7, 0.7, 0.7, 1), mayChange=True)
        self.mode_txt = OnscreenText(parent=self.hud, text="", pos=(0.05, -0.55), scale=0.045,
                                     align=TextNode.ALeft, fg=(0.7, 0.7, 0.7, 1), mayChange=True)

        self.log_lines = []
        self.log_txt = OnscreenText(parent=self.a2dTopRight, text="", pos=(-0.05, -0.08),
                                    scale=0.045, align=TextNode.ARight, fg=(1, 1, 1, 1),
                                    shadow=(0, 0, 0, 1), mayChange=True)
        self.popup = OnscreenText(parent=self.aspect2d, text="", pos=(0, 0.55), scale=0.09,
                                  fg=(1, 0.85, 0.2, 1), shadow=(0, 0, 0, 1), mayChange=True)
        self.popup_t = 0.0
        self.banner = OnscreenText(parent=self.aspect2d, text="", pos=(0, 0.1), scale=0.13,
                                   fg=(1, 0.9, 0.3, 1), shadow=(0, 0, 0, 1), mayChange=True)
        self.help_txt = OnscreenText(
            parent=self.a2dBottomLeft, pos=(0.03, 0.04), scale=0.038, align=TextNode.ALeft,
            fg=(0.8, 0.8, 0.8, 1), shadow=(0, 0, 0, 1), mayChange=False,
            text="LMB+move / A-D: aim   Wheel: power   Wheel click: shoot   RMB+move: orbit camera   "
                 "Up/Down: zoom   C: camera   G: graphics   M: 1P/2P   R: new rack")
        self.power_bar = DirectWaitBar(parent=self.a2dBottomCenter, text="POWER", value=0,
                                       range=100, scale=0.55, pos=(0, 0, 0.16),
                                       text_scale=0.1, text_fg=(1, 1, 1, 1),
                                       text_pos=(0, 0.2), barColor=(0, 1, 0.1, 1),
                                       frameColor=(0, 0, 0, 0.6))

    # ------------------------------------------------------------- game state
    def reset_game(self):
        self.replay.stop()
        self.save_game()
        self.game_shots = []
        self.game_saved = False
        for b in self.balls:
            b.destroy()
        self.balls = []
        if self.cue_ball:
            self.cue_ball.destroy()
        self.sinking = []

        self.cue_ball = Ball(self, 0, CUE_START)

        numbers = [n for n in range(1, 16) if n != 8]
        random.shuffle(numbers)
        numbers.insert(4, 8)                      # 8-ball in the middle of the rack
        idx = 0
        gap = 1.03
        for row in range(5):
            for i in range(row + 1):
                x = RACK_APEX[0] + (i - row / 2.0) * gap
                y = RACK_APEX[1] + row * gap * 0.866
                self.balls.append(Ball(self, numbers[idx], Point3(x, y, R)))
                idx += 1

        self.scores = [0, 0]
        self.streak = [0, 0]
        self.current = 0
        self.first_shot = True
        self.eight_early = False
        self.shot_pots = []
        self.shot_scratch = False
        self.shot_start = {}
        self.aim_h = 0.0
        self.aim_dir = Vec3(0, 1, 0)
        self.power = 0.5
        self.charging = False
        self.state = "aiming"          # aiming | striking | rolling | over
        self.log_lines = []
        self.banner.setText("")
        self.popup.setText("")
        self.add_log(f"New game - {self.names[0]} breaks")
        self.refresh_hud()

    # ------------------------------------------------- names, menu, replays
    def pdisplay(self, i):
        return self.names[i] if self.two_player else self.names[0]

    def set_live_visible(self, vis):
        for b in self.balls:
            if vis:
                b.np.show()
            else:
                b.np.hide()
                b.shadow.hide()
        if self.cue_ball:
            if vis and self.cue_ball.alive:
                self.cue_ball.np.show()
            elif not vis:
                self.cue_ball.np.hide()
                self.cue_ball.shadow.hide()

    def capture_frame(self):
        row = []
        for n in self.rec_order:
            b = self.rec_balls.get(n)
            if b is None or b.np.isEmpty() or b.np.isHidden():
                row.append(None)
                continue
            p = b.np.getPos()
            q = b.np.getQuat()
            row.append([round(p.x, 2), round(p.y, 2), round(p.z, 2), round(b.np.getScale()[0], 2),
                        round(q.getI(), 3), round(q.getJ(), 3), round(q.getK(), 3), round(q.getR(), 3)])
        self.rec_frames.append(row)

    def save_game(self):
        """Store this game's best pots on disk (once per game)."""
        if self.game_saved or not self.game_shots:
            return None
        self.game_saved = True
        rec = {"version": 1, "date": time.strftime("%Y-%m-%d %H:%M"), "players": list(self.names),
               "two_player": self.two_player, "scores": list(self.scores),
               "shots": self.game_shots[:5]}
        self.store.save(rec)
        return rec

    def setup_menu(self):
        self.menu_two = True
        self.menu_page = 0
        gold = (1, 0.82, 0.3, 1)
        self.menu = DirectFrame(parent=self.aspect2d, frameColor=(0.04, 0.03, 0.02, 0.93),
                                frameSize=(-1.2, 1.2, -0.92, 0.92), pos=(0, 0, 0))
        OnscreenText(parent=self.menu, text="8-BALL POOL", pos=(0, 0.74), scale=0.12, fg=gold,
                     shadow=(0, 0, 0, 1))
        OnscreenText(parent=self.menu, text="Who is playing?", pos=(0, 0.63), scale=0.05,
                     fg=(0.8, 0.8, 0.8, 1))
        self.entries = []
        for i, (label, y) in enumerate((("Player 1", 0.48), ("Player 2", 0.34))):
            OnscreenText(parent=self.menu, text=f"{label} name:", pos=(-1.1, y), scale=0.055,
                         align=TextNode.ALeft, fg=(0.9, 0.9, 0.9, 1))
            e = DirectEntry(parent=self.menu, scale=0.06, width=18, numLines=1, initialText=self.names[i],
                            pos=(-0.45, 0, y), frameColor=(0.16, 0.11, 0.08, 1), text_fg=(1, 1, 1, 1),
                            suppressKeys=1, command=self.menu_entry_enter, extraArgs=[i])
            self.entries.append(e)
        self.mode_btn = DirectButton(parent=self.menu, text="Players: 2", scale=0.055, pos=(0.72, 0, 0.41),
                                     frameSize=(-5, 5, -0.7, 1.3), command=self.menu_toggle_mode,
                                     frameColor=(0.25, 0.18, 0.1, 1), text_fg=(1, 1, 1, 1))
        self.start_btn = DirectButton(parent=self.menu, text="START GAME", scale=0.07, pos=(0, 0, 0.19),
                                      frameSize=(-5.5, 5.5, -0.7, 1.4), command=self.menu_start,
                                      frameColor=(0.10, 0.45, 0.15, 1), text_fg=(1, 1, 1, 1))
        OnscreenText(parent=self.menu, text="Saved games - click one to watch its highlights",
                     pos=(0, 0.04), scale=0.05, fg=gold)
        self.menu_rows = []
        for k in range(5):
            btn = DirectButton(parent=self.menu, text="", text_align=TextNode.ALeft, scale=0.048,
                               pos=(-1.1, 0, -0.07 - k * 0.115), frameSize=(-0.3, 45, -0.6, 1.5),
                               frameColor=(0.18, 0.12, 0.08, 1), text_fg=(1, 1, 1, 1),
                               command=self.menu_play_row, extraArgs=[k])
            self.menu_rows.append(btn)
        self.prev_btn = DirectButton(parent=self.menu, text="< Prev", scale=0.05, pos=(-0.55, 0, -0.70),
                                     frameSize=(-3, 3, -0.6, 1.4), command=self.menu_page_step, extraArgs=[-1],
                                     frameColor=(0.25, 0.18, 0.1, 1), text_fg=(1, 1, 1, 1))
        self.next_btn = DirectButton(parent=self.menu, text="Next >", scale=0.05, pos=(0.55, 0, -0.70),
                                     frameSize=(-3, 3, -0.6, 1.4), command=self.menu_page_step, extraArgs=[1],
                                     frameColor=(0.25, 0.18, 0.1, 1), text_fg=(1, 1, 1, 1))
        self.page_txt = OnscreenText(parent=self.menu, text="", pos=(0, -0.71), scale=0.04,
                                     fg=(0.85, 0.85, 0.85, 1), mayChange=True)
        OnscreenText(parent=self.menu, text="Esc quits     In game: R opens this menu",
                     pos=(0, -0.85), scale=0.04, fg=(0.6, 0.6, 0.6, 1))
        self.back_btn = DirectButton(parent=self.a2dTopRight, text="Back to menu", scale=0.055,
                                     pos=(-0.4, 0, -0.12), frameSize=(-5.5, 5.5, -0.7, 1.4),
                                     command=self.replay_back, frameColor=(0.35, 0.1, 0.1, 1),
                                     text_fg=(1, 1, 1, 1))
        self.back_btn.hide()
        self.hl_text = OnscreenText(parent=self.a2dBottomCenter, text="", pos=(0, 0.17), scale=0.09,
                                    fg=(1, 0.9, 0.4, 1), shadow=(0, 0, 0, 1), mayChange=True)
        self.hl_sub = OnscreenText(parent=self.a2dBottomCenter, text="", pos=(0, 0.09), scale=0.045,
                                   fg=(1, 1, 1, 1), shadow=(0, 0, 0, 1), mayChange=True)
        self.menu.hide()

    def refresh_menu_list(self):
        games = list(reversed(self.store.index))
        pages = max(1, (len(games) + 4) // 5)
        self.menu_page = max(0, min(self.menu_page, pages - 1))
        for k, btn in enumerate(self.menu_rows):
            idx = self.menu_page * 5 + k
            if idx < len(games):
                g = games[idx]
                p = g["players"]
                if g.get("two_player", True):
                    who = f"{p[0][:12]} {g['scores'][0]} - {g['scores'][1]} {p[1][:12]}"
                else:
                    who = f"{p[0][:12]}  score {g['scores'][0]}"
                btn["text"] = f"{g['date']}   {who}   ({g['shots']} highlights)"
                btn.show()
            else:
                btn.hide()
        if self.store.dir is None:
            self.page_txt.setText("Saving is unavailable (no writable folder)")
        else:
            self.page_txt.setText(f"Page {self.menu_page + 1}/{pages}   -   {len(games)} saved game(s)"
                                  if games else "No saved games yet - play one!")

    def menu_page_step(self, d):
        self.menu_page += d
        self.refresh_menu_list()

    def menu_entry_enter(self, text, i):
        if i == 0:
            self.entries[1]["focus"] = 1
        else:
            self.menu_start()

    def menu_toggle_mode(self):
        self.menu_two = not self.menu_two
        self.mode_btn["text"] = "Players: 2" if self.menu_two else "Players: 1"

    def open_menu(self):
        self.replay.stop()
        self.save_game()
        self.back_btn.hide()
        self.banner.setText("")
        self.popup.setText("")
        self.charging = False
        self.menu_open = True
        self.state = "menu"
        for i, e in enumerate(self.entries):
            e.enterText(self.names[i])
        self.menu_two = self.two_player
        self.mode_btn["text"] = "Players: 2" if self.menu_two else "Players: 1"
        self.refresh_menu_list()
        self.menu.show()
        self.entries[0]["focus"] = 1

    def close_menu(self):
        for e in self.entries:
            e["focus"] = 0
        self.menu.hide()
        self.menu_open = False

    def menu_start(self):
        defaults = ("Player 1", "Player 2")
        self.names = [(e.get().strip()[:16] or d) for e, d in zip(self.entries, defaults)]
        self.two_player = self.menu_two
        self.close_menu()
        self.reset_game()

    def menu_play_row(self, k):
        games = list(reversed(self.store.index))
        idx = self.menu_page * 5 + k
        if idx >= len(games):
            return
        rec = self.store.load(games[idx])
        if not rec or not rec.get("shots"):
            return
        for e in self.entries:
            e["focus"] = 0
        self.menu.hide()
        self.back_btn.show()
        self.replay.start(rec["shots"])

    def replay_back(self):
        self.replay.stop()
        self.back_btn.hide()
        self.menu.show()
        self.entries[0]["focus"] = 1

    def new_game_request(self):
        if not self.menu_open:
            self.open_menu()

    def escape_pressed(self):
        if self.replay.active:
            if self.menu_open:
                self.replay_back()
            else:
                self.replay.stop()
            return
        sys.exit()

    def toggle_quality(self):
        self.set_quality(not self.high_quality)

    def set_quality(self, high):
        self.high_quality = high
        if high:
            self.render.setShaderAuto()
            self.render.setLight(self.shadow_np)
            set_env_enabled(False)
            for p in self.lamp_lights:
                p.setColor(Vec4(0.40, 0.37, 0.30, 1))
        else:
            self.render.setShaderOff()
            self.render.clearLight(self.shadow_np)
            set_env_enabled(True)
            for p in self.lamp_lights:
                p.setColor(Vec4(0.75, 0.70, 0.60, 1))
        self.refresh_hud()

    def toggle_mode(self):
        if self.menu_open:
            return
        self.two_player = not self.two_player
        self.reset_game()

    def cycle_camera(self):
        self.cam_mode = (self.cam_mode + 1) % 3
        self.cam_yaw = 0.0
        self.cam_pitch = 0.0
        self.camLens.setFov(60 if self.cam_mode == 1 else 55)
        self.refresh_hud()

    def add_log(self, msg):
        self.log_lines.append(msg)
        self.log_lines = self.log_lines[-7:]
        self.log_txt.setText("\n".join(self.log_lines))

    def show_popup(self, text):
        self.popup.setText(text)
        self.popup_t = 3.0

    # ------------------------------------------------------------------ input
    def set_lmb(self, down):
        self.lmb = down
        if down and self.mouseWatcherNode.hasMouse():
            m = self.mouseWatcherNode.getMouse()
            self.last_mouse = (m.getX(), m.getY())

    def left_click(self):
        """Middle mouse button: shoot with the selected power."""
        if self.state != "aiming" or self.charging:
            return
        self.strike_power = max(self.power, 0.05)
        self.strike_t = 0.0
        self.state = "striking"

    def wheel_power(self, direction):
        if self.state == "aiming" and not self.charging:
            self.power = max(0.05, min(1.0, self.power + direction * 0.05))

    def zoom_step(self, factor):
        self.cam_zoom = max(0.4, min(1.8, self.cam_zoom * factor))

    def set_rmb(self, down):
        self.rmb = down
        self.rmb_last = None

    def detect_contacts(self):
        """Play a sound whenever a moving ball newly touches a ball or a wall."""
        touching = set()
        for b in self.alive_balls():
            if not b.in_world:
                continue
            sp = b.speed()
            name = b.node.getName()
            for c in self.world.contactTest(b.node).getContacts():
                n0, n1 = c.getNode0(), c.getNode1()
                other = n1 if n0.getName() == name else n0
                oname = other.getName()
                if oname == "floor":
                    continue
                if oname.startswith("ball_"):
                    key, kind = frozenset((name, oname)), "ball"
                else:
                    key, kind = (name, "wall"), "cushion"
                touching.add(key)
                if key not in self.touching and sp > 0.8:
                    self.sfx.play(kind, 0.15 + sp / 25.0)
        self.touching = touching

    def start_charge(self):
        if self.state == "aiming" and not self.charging:
            self.charging = True
            self.power = 0.0

    def release_charge(self):
        if not self.charging:
            return
        self.charging = False
        if self.power < 0.03:
            self.power = 0.0
            return
        self.strike_power = self.power
        self.strike_t = 0.0
        self.state = "striking"

    def mouse_on_table(self):
        if not self.mouseWatcherNode.hasMouse():
            return None
        m = self.mouseWatcherNode.getMouse()
        near, far = Point3(), Point3()
        self.camLens.extrude(m, near, far)
        near = self.render.getRelativePoint(self.cam, near)
        far = self.render.getRelativePoint(self.cam, far)
        dz = far.getZ() - near.getZ()
        if abs(dz) < 1e-6:
            return None
        t = (R - near.getZ()) / dz
        return near + (far - near) * t

    # ------------------------------------------------------------------ rules
    def alive_balls(self):
        out = [b for b in self.balls if b.alive]
        if self.cue_ball.alive:
            out.append(self.cue_ball)
        return out

    def enforce_rolling(self, cbdata=None):
        """Pure rolling without slipping: contact point velocity = 0, so
        omega = (-vy/R, vx/R, 0) for a ball moving with velocity v on the felt."""
        for b in self.alive_balls():
            if not b.in_world:
                continue
            v = b.node.getLinearVelocity()
            b.node.setAngularVelocity(Vec3(-v.y / R, v.x / R, 0))

    def balls_moving(self):
        return any(b.speed() > 0.05 for b in self.alive_balls())

    def handle_pockets(self):
        for b in self.alive_balls():
            p = b.pos()
            if abs(p.x) > HW + 0.05 or abs(p.y) > HL + 0.05 or p.z < -0.8:
                self.pocket(b)

    def pocket(self, b):
        p = b.pos()
        target = self.table.nearest_pocket(p)
        b.leave_world()
        b.shadow.hide()
        self.sfx.play("pocket", 0.8)
        self.sinking.append([b, Vec3(p), target, 0.0])
        if b is self.cue_ball:
            self.shot_scratch = True
            return
        self.balls.remove(b)
        start = self.shot_start.get(b.number)
        dist = (start - target).length() if start is not None else 0
        self.shot_pots.append({"num": b.number, "dist": dist})
        if b.number == 8 and self.balls:
            self.eight_early = True

    def update_sinking(self, dt):
        for e in self.sinking[:]:
            b, start, target, t = e
            t += dt / 0.45
            e[3] = t
            k = min(t, 1.0)
            ease = k * k
            x = start.x + (target.x - start.x) * ease
            y = start.y + (target.y - start.y) * ease
            b.np.setPos(x, y, R - 1.8 * ease)
            b.np.setScale(1.0 - 0.35 * ease)
            if t >= 1.0:
                self.sinking.remove(e)
                if b is self.cue_ball:
                    b.np.hide()
                else:
                    b.destroy()

    def free_cue_spot(self):
        y = CUE_START.getY()
        while y < 18:
            spot = Point3(0, y, R)
            if all((b.pos() - spot).length() > 1.3 for b in self.balls):
                return spot
            y += 1.3
        return CUE_START

    def resolve_turn(self):
        pl = self.current
        pots = self.shot_pots
        nums = [p["num"] for p in pots]
        pname = self.pdisplay(pl)

        if self.eight_early or (self.shot_scratch and 8 in nums):
            self.finish_game(lost=True)
            return

        parts = []
        total = 0
        keep_turn = False
        if self.shot_scratch:
            pen = min(10, self.scores[pl])
            total = -pen
            parts.append(f"Scratch -{pen} (pots void)")
            self.streak[pl] = 0
        elif pots:
            keep_turn = True
            normal = [n for n in nums if n != 8]
            if normal:
                total += 10 * len(normal)
                parts.append(f"{len(normal)} ball{'s' if len(normal) > 1 else ''} +{10 * len(normal)}")
            if len(pots) >= 2:
                c = 10 * (len(pots) - 1)
                total += c
                parts.append(f"Combo x{len(pots)} +{c}")
            longs = [p for p in pots if p["dist"] > 22]
            if longs:
                c = 15 * len(longs)
                total += c
                parts.append(f"Long pot +{c}")
            if self.first_shot:
                total += 20
                parts.append("Break pot +20")
            self.streak[pl] += 1
            if self.streak[pl] >= 2:
                c = 5 * (self.streak[pl] - 1)
                total += c
                parts.append(f"Streak x{self.streak[pl]} +{c}")
            if 8 in nums:
                total += 30
                parts.append("8-ball finish +30")
        else:
            self.streak[pl] = 0

        self.scores[pl] += total
        self.first_shot = False
        if pots and not self.shot_scratch and total > 0:          # candidate highlight
            self.capture_frame()
            rec = {"player": pl, "name": pname, "points": total, "parts": list(parts), "pots": nums,
                   "dt": REC_DT, "order": list(self.rec_order), "frames": self.rec_frames}
            rec.update(self.rec_meta)
            self.game_shots.append(rec)
            self.game_shots.sort(key=lambda r: (-r["points"], -len(r["pots"])))
            del self.game_shots[8:]

        if not self.cue_ball.alive:
            self.cue_ball.respawn(self.free_cue_spot())

        if parts:
            self.add_log(f"{pname}: " + ", ".join(parts))
            self.show_popup(f"{total:+d}   " + "  |  ".join(parts))
        else:
            self.add_log(f"{pname}: miss")

        if 8 in nums:
            self.finish_game(lost=False)
            return

        if not keep_turn and self.two_player:
            self.current ^= 1
            self.add_log(f"{self.names[self.current]}'s turn")
        self.state = "aiming"
        self.refresh_hud()

    def finish_game(self, lost):
        self.state = "over"
        pl = self.current
        nm = self.names
        if lost:
            if self.two_player:
                msg = f"8-ball sunk too early!\n{nm[1 - pl].upper()} WINS"
            else:
                msg = "8-ball sunk too early!\nGAME OVER"
        else:
            if self.two_player:
                if self.scores[0] == self.scores[1]:
                    msg = f"DRAW  {self.scores[0]} - {self.scores[1]}"
                else:
                    w = 0 if self.scores[0] > self.scores[1] else 1
                    msg = f"{nm[w].upper()} WINS\n{self.scores[0]} - {self.scores[1]}"
            else:
                msg = f"TABLE CLEARED!\nScore: {self.scores[0]}"
        rec = self.save_game()
        note = "(R = new game, Esc stops highlights)" if rec else "(R = new game)"
        self.banner.setText(msg + "\n" + note)
        self.banner.setScale(0.07)
        self.banner.setPos(0, 0.66)
        self.add_log(msg.replace("\n", " "))
        self.refresh_hud()
        if rec:
            self.replay.start(rec["shots"])          # highlights loop until a new game starts

    def refresh_hud(self):
        for i in range(2):
            active = (i == self.current and self.state != "over")
            self.name_txt[i].setFg((1, 0.85, 0.2, 1) if active else (0.85, 0.85, 0.85, 1))
            self.score_txt[i].setFg((1, 0.85, 0.2, 1) if active else (0.85, 0.85, 0.85, 1))
            self.name_txt[i].setText(("> " if active else "  ") + self.names[i].upper())
            self.score_txt[i].setText(str(self.scores[i]))
            if i == 1 and not self.two_player:
                self.name_txt[i].hide()
                self.score_txt[i].hide()
            else:
                self.name_txt[i].show()
                self.score_txt[i].show()
        self.left_txt.setText(f"Balls left: {len(self.balls)}    "
                              f"Streak: {self.streak[self.current]}")
        self.cam_txt.setText(f"Camera: {self.CAM_NAMES[self.cam_mode]} [C]   Gfx: {'High' if self.high_quality else 'Standard'} [G]")
        self.mode_txt.setText(f"Mode: {'2 players' if self.two_player else '1 player'}   [M]")

    # ------------------------------------------------------------ aiming guide
    def trace(self, p, d):
        """Where would the cue ball be when it first touches something?"""
        best = 1e9
        kind, data = None, None
        for b in self.balls:
            if not b.alive:
                continue
            c = b.pos()
            fx, fy = c.x - p.x, c.y - p.y
            proj = fx * d.x + fy * d.y
            if proj <= 0:
                continue
            perp2 = fx * fx + fy * fy - proj * proj
            if perp2 >= 4 * R * R:
                continue
            t = proj - math.sqrt(4 * R * R - perp2)
            if 0 <= t < best:
                best, kind, data = t, "ball", c
        ex, ey = HW - R, HL - R
        for axis, lim in ((0, ex), (1, ey)):
            dv = d.x if axis == 0 else d.y
            pv = p.x if axis == 0 else p.y
            if abs(dv) < 1e-6:
                continue
            bound = lim if dv > 0 else -lim
            t = (bound - pv) / dv
            if 0 < t < best:
                n = Vec3(-1, 0, 0) if axis == 0 else Vec3(0, -1, 0)
                if dv < 0:
                    n = -n
                best, kind, data = t, "wall", n
        return best, kind, data

    def draw_guide(self):
        if self.guide:
            self.guide.removeNode()
            self.guide = None
        if self.state not in ("aiming", "striking") or not self.cue_ball.alive:
            return
        p = self.cue_ball.pos()
        d = self.aim_dir
        t, kind, data = self.trace(p, d)
        z = 0.06
        root = self.render.attachNewNode("guide")
        ls = LineSegs()

        def dashed(a, b, col, thick, dash=0.55, gap=0.4):
            ls.setColor(*col)
            ls.setThickness(thick)
            v = b - a
            length = v.length()
            if length < 1e-3:
                return
            v.normalize()
            s = 0.0
            while s < length:
                e = min(s + dash, length)
                ls.moveTo(a + v * s)
                ls.drawTo(a + v * e)
                s += dash + gap

        def circle(c, rad, col):
            ls.setColor(*col)
            ls.setThickness(2.5)
            for i in range(33):
                th = 2 * math.pi * i / 32
                pt = Point3(c.x + rad * math.cos(th), c.y + rad * math.sin(th), z)
                if i == 0:
                    ls.moveTo(pt)
                else:
                    ls.drawTo(pt)

        start = Point3(p.x + d.x * (R + 0.12), p.y + d.y * (R + 0.12), z)
        end = Point3(p.x + d.x * t, p.y + d.y * t, z)
        dashed(start, end, (1, 1, 1, 0.9), 3.0)
        circle(end, R, (1, 1, 1, 0.9))
        contact = None

        if kind == "ball":
            c = data
            n = Vec3(c.x - end.x, c.y - end.y, 0)
            n.normalize()
            contact = Point3(end.x + n.x * R, end.y + n.y * R, R)
            # object ball path
            a = Point3(c.x + n.x * R, c.y + n.y * R, z)
            b = Point3(c.x + n.x * (R + 5.0), c.y + n.y * (R + 5.0), z)
            dashed(a, b, (1, 0.85, 0.2, 0.95), 2.5, 0.35, 0.25)
            # cue ball deflection (tangent)
            dn = d.x * n.x + d.y * n.y
            tdir = Vec3(d.x - n.x * dn, d.y - n.y * dn, 0)
            if tdir.length() > 0.08:
                tdir.normalize()
                dashed(end, Point3(end.x + tdir.x * 3.0, end.y + tdir.y * 3.0, z),
                       (0.5, 0.9, 1, 0.9), 2.5, 0.35, 0.25)
        elif kind == "wall":
            n = data
            contact = Point3(end.x - n.x * R, end.y - n.y * R, R)
            dn = d.x * n.x + d.y * n.y
            refl = Vec3(d.x - 2 * dn * n.x, d.y - 2 * dn * n.y, 0)
            dashed(end, Point3(end.x + refl.x * 6, end.y + refl.y * 6, z),
                   (1, 1, 1, 0.4), 2.0, 0.4, 0.35)

        root.attachNewNode(ls.create())
        root.setLightOff()
        root.setTransparency(TransparencyAttrib.MAlpha)

        if contact is not None:                 # the tiny impact dot
            dot = root.attachNewNode(self.dot_geom)
            dot.setScale(0.11)
            dot.setPos(contact)
            dot.setLightOff()
            dot.setDepthTest(False)
            dot.setBin("fixed", 100)
        self.guide = root

    # ----------------------------------------------------------------- update
    def update_aim(self, dt):
        if self.state != "aiming":
            return
        fine = self.key_down(self.k_shift)
        rate = (10.0 if fine else 70.0) * dt
        if self.key_down(self.k_left):
            self.aim_h += rate
        if self.key_down(self.k_right):
            self.aim_h -= rate

        # Aiming with the mouse only happens while the LEFT button is held down.
        if self.mouseWatcherNode.hasMouse():
            m = self.mouseWatcherNode.getMouse()
            cur = (m.getX(), m.getY())
            if self.lmb and not self.rmb and not self.charging:
                if self.cam_mode != 1:
                    pt = self.mouse_on_table()          # aim towards the pointer
                    if pt is not None:
                        cp = self.cue_ball.pos()
                        v = Vec3(pt.x - cp.x, pt.y - cp.y, 0)
                        if v.length() > 0.3:
                            self.aim_h = math.degrees(math.atan2(-v.x, v.y))
                else:                                   # player view: turn with mouse X
                    self.aim_h -= (cur[0] - self.last_mouse[0]) * (40.0 if fine else 110.0)
            self.last_mouse = cur
        h = math.radians(self.aim_h)
        self.aim_dir = Vec3(-math.sin(h), math.cos(h), 0)

        if self.charging:
            self.power = min(1.0, self.power + dt * 0.75)

    def update_state(self, dt):
        if self.state == "rolling":
            self.roll_t += dt
            self.rec_acc += dt
            while self.rec_acc >= REC_DT:
                self.rec_acc -= REC_DT
                if len(self.rec_frames) < 1200:
                    self.capture_frame()
            if self.roll_t > 0.4 and not self.balls_moving() and not self.sinking:
                self.resolve_turn()
        elif self.state == "striking":
            self.strike_t += dt / 0.10
            if self.strike_t >= 1.0:
                self.fire()

    def fire(self):
        self.shot_pots = []
        self.shot_scratch = False
        self.eight_early = False
        self.shot_start = {b.number: Point3(b.pos()) for b in self.balls}
        self.rec_balls = {0: self.cue_ball}
        self.rec_balls.update({b.number: b for b in self.balls})
        self.rec_order = sorted(self.rec_balls)
        self.rec_frames, self.rec_acc = [], 0.0
        self.rec_meta = {"aim_h": round(self.aim_h, 2), "power": round(self.strike_power, 3)}
        self.capture_frame()
        impulse = self.aim_dir * (self.strike_power * MAX_SPEED * BALL_MASS)
        self.cue_ball.node.setActive(True)
        self.cue_ball.node.applyCentralImpulse(impulse)
        self.sfx.play("cue", 0.35 + 0.65 * self.strike_power)
        self.roll_t = 0.0
        self.state = "rolling"
        self.cue.np.hide()

    def update_visuals(self):
        if self.replay.active:                       # live balls are hidden during a replay
            if self.guide:
                self.guide.removeNode()
                self.guide = None
            return
        for b in self.alive_balls():
            p = b.pos()
            b.shadow.setPos(p.x + 0.12, p.y - 0.08, 0.012)
            if self.high_quality:
                b.shadow.hide()
            else:
                b.shadow.show()
        if self.state in ("aiming", "striking") and self.cue_ball.alive:
            if self.state == "striking":
                k = min(self.strike_t, 1.0)
                pull = (0.35 + self.strike_power * 5.0) * (1 - k * k)
            else:
                pull = 0.35 + self.power * 5.0
            cp = self.cue_ball.pos()
            d = self.aim_dir
            off = R + 0.04 + pull
            self.cue.place(Point3(cp.x - d.x * off, cp.y - d.y * off, R), self.aim_h)
        else:
            self.cue.np.hide()
        self.draw_guide()

    def update_camera(self, dt):
        # right mouse button + move = orbit
        if self.rmb and self.mouseWatcherNode.hasMouse():
            m = self.mouseWatcherNode.getMouse()
            cur = (m.getX(), m.getY())
            if self.rmb_last is not None:
                self.cam_yaw -= (cur[0] - self.rmb_last[0]) * 140.0
                self.cam_pitch = max(-70.0, min(70.0, self.cam_pitch + (cur[1] - self.rmb_last[1]) * 70.0))
            self.rmb_last = cur
        else:
            self.rmb_last = None

        vf = math.radians(self.camLens.getVfov())
        hf = math.radians(self.camLens.getHfov())
        fit = max(27.0 / math.tan(vf / 2), 17.0 / math.tan(hf / 2)) * self.cam_zoom
        az = math.radians(self.cam_yaw)
        if self.cam_mode in (0, 2):
            base_e = 62.0 if self.cam_mode == 0 else 89.0
            e = math.radians(max(8.0, min(89.6, base_e + self.cam_pitch)))
            look = Point3(0, -1, 0) if self.cam_mode == 0 else Point3(0, 0, 0)
            direction = Vec3(math.sin(az) * math.cos(e), -math.cos(az) * math.cos(e), math.sin(e))
            desired = Vec3(look.x, look.y, look.z) + direction * fit
        else:
            if self.state in ("aiming", "striking") and self.cue_ball.alive:
                self.player_anchor = Point3(self.cue_ball.pos())
                self.player_aim = Vec3(self.aim_dir)
            d = self.player_aim
            dx = d.x * math.cos(az) - d.y * math.sin(az)
            dy = d.x * math.sin(az) + d.y * math.cos(az)
            e = math.radians(max(6.0, min(80.0, 27.0 + self.cam_pitch)))
            dist = 10.1 * self.cam_zoom
            cp = self.player_anchor
            desired = Vec3(cp.x - dx * dist * math.cos(e), cp.y - dy * dist * math.cos(e),
                           dist * math.sin(e))
            look = Point3(cp.x + dx * 14.0, cp.y + dy * 14.0, 0.2)
        k = 1.0 - math.exp(-dt * 9.0)
        self.cam_pos += (desired - self.cam_pos) * k
        self.camera.setPos(self.cam_pos)
        self.camera.lookAt(look)

    def update_hud(self, dt):
        p = self.power if self.state in ("aiming", "striking") else 0.0
        self.power_bar["value"] = p * 100
        if self.replay.active or self.menu_open or self.state == "over":
            self.power_bar.hide()
        else:
            self.power_bar.show()
        self.power_bar["barColor"] = (min(1.0, p * 2), min(1.0, 2 - p * 2), 0.1, 1)
        if self.popup_t > 0:
            self.popup_t -= dt
            self.popup.setFg((1, 0.85, 0.2, max(0.0, min(1.0, self.popup_t))))
            if self.popup_t <= 0:
                self.popup.setText("")
        self.left_txt.setText(f"Balls left: {len(self.balls)}    "
                              f"Streak: {self.streak[self.current]}")
        for i in range(2):
            self.score_txt[i].setText(str(self.scores[i]))

    def update(self, task):
        dt = min(ClockObject.getGlobalClock().getDt(), 1 / 20.0)
        self.world.doPhysics(dt, 10, 1.0 / 240.0)
        self.pub.update(dt, self.camera.getZ(), self.cam_mode == 1)
        self.enforce_rolling()
        self.detect_contacts()
        self.handle_pockets()
        self.update_sinking(dt)
        for b in self.alive_balls():
            if 0 < b.speed() < 0.12:
                b.stop()
        self.update_aim(dt)
        self.update_state(dt)
        self.replay.update(dt)
        self.update_visuals()
        self.update_camera(dt)
        self.update_hud(dt)
        return task.cont


if __name__ == "__main__":
    PoolGame().run()
