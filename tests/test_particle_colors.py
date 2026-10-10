import subprocess
import sys


def test_usd_vertex_colors_shader_and_actual_stage_readback():
    script = """
from types import SimpleNamespace
from pxr import Usd, UsdGeom, UsdShade, Sdf, Vt, Gf
from unirobosim import EntityPath
from unirobosim_isaaclab.native import IsaacLabNativeWorld
stage=Usd.Stage.CreateInMemory()
points=UsdGeom.Points.Define(stage,'/particles')
points.CreatePointsAttr(Vt.Vec3fArray([(0,0,0),(1,0,0)]))
w=object.__new__(IsaacLabNativeWorld)
w._m=SimpleNamespace(UsdGeom=UsdGeom,UsdShade=UsdShade,Sdf=Sdf,Vt=Vt,Gf=Gf)
w._fluids={EntityPath('/fluid'):[SimpleNamespace(points=points)]}
w._write_particle_colors(points,[(1,0,0,1),(0,1,0,.5)])
w._bind_particle_material(points)
assert points.GetDisplayColorPrimvar().GetInterpolation()=='vertex'
assert points.GetDisplayOpacityPrimvar().GetInterpolation()=='vertex'
assert w.read_particle_colors(EntityPath('/fluid'))[0][1]==(0.,1.,0.,.5)
points.GetDisplayColorPrimvar().Set(Vt.Vec3fArray([(0,0,1),(1,0,0)]))
assert w.read_particle_colors(EntityPath('/fluid'))[0][0]==(0.,0.,1.,1.)
material,_=UsdShade.MaterialBindingAPI(points).ComputeBoundMaterial()
shader=UsdShade.Shader.Get(stage,str(material.GetPath())+'/surface')
assert shader.GetInput('diffuseColor').GetConnectedSource()[1]=='result'
assert shader.GetInput('opacity').GetConnectedSource()[1]=='result'
w._sync_particle_visual(points,.015)
inst=UsdGeom.PointInstancer.Get(stage,'/particle_visual')
assert list(inst.GetProtoIndicesAttr().Get())==[0,1]
assert list(inst.GetPositionsAttr().Get())==list(points.GetPointsAttr().Get())
assert points.GetVisibilityAttr().Get()=='invisible'
for i,color in enumerate(w.read_particle_colors(EntityPath('/fluid'))[0]):
    path=inst.GetPrototypesRel().GetTargets()[i]
    sphere=UsdGeom.Sphere.Get(stage,path)
    assert sphere.GetRadiusAttr().Get()==.015
    material,_=UsdShade.MaterialBindingAPI(sphere).ComputeBoundMaterial()
    surf=UsdShade.Shader.Get(stage,str(material.GetPath())+'/surface')
    assert tuple(surf.GetInput('diffuseColor').Get())==color[:3]
    assert surf.GetInput('opacity').Get()==color[3]
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
